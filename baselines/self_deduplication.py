import pdb

import numpy as np

from utils.parse_args import parse_args
from utils.blocker import block_model_1d
from utils.common import (
    load_models_info,
    load_model,
    set_model_args,
)


def run():
    model_args, data_args, training_args = parse_args()
    models_info = load_models_info(model_args)

    total_new_blocks = 0
    max_acc_drop = 0.0
    for model_info in models_info:
        print(f"Model info: {model_info}")
        model, eval_fn, train_fn, sensitivity_fn = load_model(model_info, model_args)
        model_storage = block_model_1d(model_args.block_size, model)

        set_model_args(model_args, model, model_storage)
        model_storage["model_range"] = [0, model_storage["blocks"].shape[0]]
        model_storage["untouch_weights"] = [model_storage["untouch_weights"]]

        model_constitution, acc_drop = deduplicate_blocks(
            model_args,
            data_args,
            training_args,
            model_storage,
            model_info,
            0,
            eval_fn,
            train_fn,
            sensitivity_fn,
        )
        max_acc_drop = max(max_acc_drop, acc_drop)
        total_new_blocks += len(set(model_constitution))
        print(f"Remaining blocks: {list(set(model_constitution))}")

    n_models = len(models_info)
    untouched_weights = model_args.untouched_weights * n_models
    n_original_weights = model_args.n_original_weights * n_models
    remaining_weights = total_new_blocks * model_args.block_size + untouched_weights
    cr = remaining_weights / n_original_weights
    print(f"{total_new_blocks=} | {cr=:.6f} | {max_acc_drop=:.6f}")


def deduplicate_blocks(
    model_args,
    data_args,
    training_args,
    model_storage,
    model_info,
    model_id,
    eval_fn,
    train_fn,
    sensitivity_fn,
    distance_metric="l1",
):
    # Set parameters
    acc_threshold = model_info["original_acc"] - model_info["acc_drop_threshold"]
    data_args.task_name = model_info["task_name"]
    model_args.model_name_or_path = model_info["model_path"]
    model_range_start = model_storage["model_range"][model_id]
    model_range_end = model_storage["model_range"][model_id + 1]
    candidate_range = model_range_end

    # Get initial model constitution
    model_constitution = list(range(model_range_start, model_range_end))

    # Prepare the candidate blocks
    candidate_blocks = model_storage["blocks"][:candidate_range]

    # The order the blocks are iterated through
    interate_seq = np.arange(model_range_start, model_range_end)

    return_n_embed_blocks = training_args.sensitivity_measure == "wanda"
    measures, n_embed_blocks = sensitivity_fn(
        model_info,
        training_args.sensitivity_measure,
        skip_embeds=False,
        return_n_embed_blocks=return_n_embed_blocks,
    )

    all_zeros = np.all(measures == 0, axis=1)
    non_all_zeros = ~all_zeros
    allzero_indices = np.where(all_zeros != 0)[0]
    nonzero_indices = np.nonzero(non_all_zeros)[0]

    allzerograd_seq = interate_seq[allzero_indices]
    seq_to_order = interate_seq[nonzero_indices]
    measures = measures[nonzero_indices]
    print(f"Size of all zero sensitivity blocks : {len(allzero_indices)}")
    print(f"All zero sensitivity blocks : {allzerograd_seq}")
    allzero_indices_set = set(allzerograd_seq)

    if training_args.orderby == "l1_norm":
        block_sens = np.sum(np.abs(measures), axis=1)
    elif training_args.orderby == "l2_norm":
        block_sens = np.linalg.norm(measures, axis=1)
    elif training_args.orderby == "l_inf_norm":
        block_sens = np.max(np.abs(measures), axis=1)
    elif training_args.orderby == "3rd_quantile":
        block_sens = np.quantile(measures, 0.75, axis=1)
    else:
        raise ValueError(f"Invalid orderby: {training_args.orderby}")

    # Because Wanda only applies to linear layers, we deduplicate them at last
    ordered_indices = np.argsort(block_sens)
    if training_args.sensitivity_measure == "wanda":
        embed_seq = interate_seq[:n_embed_blocks]
        other_seq = interate_seq[n_embed_blocks:]
        other_seq = other_seq[ordered_indices]
        interate_seq = np.concatenate((embed_seq, other_seq))
    else:
        # interate_seq = interate_seq[ordered_indices]
        interate_seq = seq_to_order[ordered_indices]

    # Print the block magnitude
    # ordered_sensitivity = [round(m, 6) for m in block_sens[ordered_indices]]
    # measures = measures[ordered_indices]

    # Initialize variables
    # Current iteration, evaluate every n iterations
    curr_iter = 0

    # Deduplicated indices
    dedup_indices = set()
    tobe_dedup_indices = set()
    # Used all-zero sensitivity indices
    used_allzero_indices = set()
    used_allzero_indices_temp = set()
    # Constitution to be evaluated
    temp_constitution = model_constitution.copy()

    # for i, sens, measure in zip(interate_seq, ordered_sensitivity, measures):
    for i in interate_seq:
        curr_iter += 1
        tobe_dedup_indices.add(i)

        block_2b_replaced = model_storage["blocks"][i]
        diff = candidate_blocks - block_2b_replaced
        # Sort by some metrics: l1, l2, cosine
        if distance_metric == "l1":
            dist = np.sum(np.abs(diff), axis=1)
        elif distance_metric == "l2":
            dist = np.linalg.norm(diff, axis=1)
        elif distance_metric == "cosine":
            dist = np.dot(candidate_blocks, block_2b_replaced) / (
                np.linalg.norm(candidate_blocks) * np.linalg.norm(block_2b_replaced)
            )
        else:
            raise ValueError(f"Invalid distance metric: {distance_metric}")

        # Find the most similar block j
        j = -1
        for idx in dist.argsort():
            if idx == i or idx in dedup_indices or idx in tobe_dedup_indices:
                continue
            j = idx
            break

        if j in allzero_indices_set:
            used_allzero_indices_temp.add(j)

        # avg_distance = round(dist[j] / len(block_2b_replaced), 4)
        # total_diff = np.dot(measure, diff[j])
        print(f"{model_info['model_path']} block {i} -> {j}")
        # print(f"{avg_distance=}, {sens=}, {total_diff=}")

        # Replace the current block with the most similar block
        temp_constitution = [j if c == i else c for c in temp_constitution]

        if curr_iter % training_args.every_n == 0 or curr_iter == len(interate_seq):
            acc = eval_fn(
                data_args,
                model_args,
                training_args,
                model_info,
                temp_constitution,
                model_storage,
                model_id,
            )

            if acc >= acc_threshold:
                dedup_indices |= tobe_dedup_indices
                used_allzero_indices |= used_allzero_indices_temp
                model_constitution = temp_constitution
                if training_args.do_finetune:
                    train_fn(
                        data_args,
                        model_args,
                        training_args,
                        model_info,
                        temp_constitution,
                        model_storage,
                        model_id,
                    )
            tobe_dedup_indices = set()
            used_allzero_indices_temp = set()
            temp_constitution = model_constitution.copy()
            print(f"acc: {acc:.4f}, dedup success: {acc >= acc_threshold}")
            print(f"Model constitution after dedup: {model_constitution}\n")

    # Deduplicate all-zero sensitivity blocks
    print(f"Used all-zero indices: {list(used_allzero_indices)}")
    allzerograd_seq = [i for i in allzerograd_seq if i not in used_allzero_indices]

    # Start deduplication
    temp_constitution = model_constitution.copy()
    for i in allzerograd_seq:
        block_2b_replaced = model_storage["blocks"][i]
        diff = candidate_blocks - block_2b_replaced
        # Sort by some metrics: l1, l2, cosine
        if distance_metric == "l1":
            dist = np.sum(np.abs(diff), axis=1)
        elif distance_metric == "l2":
            dist = np.linalg.norm(diff, axis=1)
        elif distance_metric == "cosine":
            dist = np.dot(candidate_blocks, block_2b_replaced) / (
                np.linalg.norm(candidate_blocks) * np.linalg.norm(block_2b_replaced)
            )
        else:
            raise ValueError(f"Invalid distance metric: {distance_metric}")

        for j in dist.argsort():
            if j in allzero_indices_set or j in dedup_indices:
                continue
            # Replace the current block with the most similar block
            print(f"{model_info['model_path']} block {i} -> {j}")
            temp_constitution = [j if c == i else c for c in temp_constitution]
            break

    acc = eval_fn(
        data_args,
        model_args,
        training_args,
        model_info,
        temp_constitution,
        model_storage,
        model_id,
    )
    if acc >= acc_threshold:
        model_constitution = temp_constitution

    acc_drop = model_info["original_acc"] - acc

    if len(allzerograd_seq) == 0:
        print(
            f"{model_info['model_path']} dedup zero sensitivity blocks acc: {acc:.4f}, dedup success: {acc >= acc_threshold}"
        )
        print(f"Model constitution after dedup: {model_constitution}\n")

    return model_constitution, acc_drop