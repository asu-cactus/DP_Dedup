import pdb

import numpy as np

from utils.parse_args import parse_args
from utils.blocker import block_model_1d
from utils.common import (
    load_models_info,
    load_model,
    merge_model_storage,
    merge_base_model_storage,
    separate_blocks,
    compute_compression_ratio,
    set_model_args,
    set_val_epsilon,
)


def run():
    model_args, data_args, training_args = parse_args()
    models_info = load_models_info(model_args)

    blockss_from_base, accs, n_new_blockss = [], [], []
    n_evalss, n_failss, crs = [], [], []

    n_base_blocks = 0
    total_untouched_weights, total_original_weights = 0, 0
    base_model_storage = {"blocks": None, "untouched_weights": None}
    for i in range(model_args.n_base_models):
        base_model = load_model(models_info[i], model_args)[0]
        base_storage = block_model_1d(model_args.block_size, base_model)
        n_blocks = base_storage["blocks"].shape[0]
        n_base_blocks += n_blocks
        base_model_storage = merge_base_model_storage(base_model_storage, base_storage)

        set_model_args(model_args, base_model, base_storage)
        total_untouched_weights += model_args.untouched_weights
        total_original_weights += model_args.n_original_weights

        blockss_from_base.append(set())
        accs.append(models_info[i]["original_acc"])
        n_new_blockss.append(n_blocks)
        n_evalss.append(0)
        n_failss.append(0)
        crs.append(1.0)

    acc = accs[-1]
    for model_info in models_info[model_args.n_base_models :]:
        print(f"Model info: {model_info}")
        set_val_epsilon(training_args, model_info["budget"], models_info[0]["budget"])
        model, eval_fn, sensitivity_fn = load_model(model_info, model_args)
        curr_model_storage = block_model_1d(model_args.block_size, model)

        set_model_args(model_args, model, curr_model_storage)
        total_untouched_weights += model_args.untouched_weights
        total_original_weights += model_args.n_original_weights

        model_storage = merge_model_storage(base_model_storage, curr_model_storage)
        # The following line guarantees the fairness rule
        set_acc = model_info["original_acc"] - model_info["acc_drop_threshold"]
        if training_args.enforce_fairness:
            model_info["acc_threshold"] = max(set_acc, acc)
        else:
            model_info["acc_threshold"] = set_acc
        model_constitution, acc, n_fails, n_evals = deduplicate_blocks(
            model_args,
            data_args,
            training_args,
            model_storage,
            model_info,
            eval_fn,
            sensitivity_fn,
        )

        # Separate blocks
        n_new_blocks, blocks_from_base = separate_blocks(
            model_constitution, n_base_blocks
        )
        n_new_blockss.append(n_new_blocks)
        blockss_from_base.append(blocks_from_base)

        cr = compute_compression_ratio(
            n_new_blocks,
            model_args.block_size,
            model_args.untouched_weights,
            model_args.n_original_weights,
        )
        print(f"Current model {n_new_blocks=}, {cr=}, {acc=}, {n_fails=}, {n_evals=}")
        # Record the results
        accs.append(acc)
        n_evalss.append(n_evals)
        n_failss.append(n_fails)
        crs.append(cr)

    if model_args.dummy_base_model >= 0:
        accs = accs[1:]
        blockss_from_base = blockss_from_base[1:]
        n_new_blockss = n_new_blockss[1:]
        n_evalss = n_evalss[1:]
        n_failss = n_failss[1:]
        crs = crs[1:]

    # Accuracies
    acc_drops = [
        model_info["original_acc"] - acc for acc, model_info in zip(accs, models_info)
    ]
    max_acc_drop = max(acc_drops)
    # Total new blocks
    total_new_blocks = sum(n_new_blockss)
    # Number of models
    n_models = len(models_info)
    # Blocks from base model
    blockss_from_base = set.union(*blockss_from_base)
    if model_args.dummy_base_model >= 0:
        total_new_blocks += len(blockss_from_base)
    # Compression ratios
    crs = [round(cr, 4) for cr in crs]

    cr = compute_compression_ratio(
        total_new_blocks,
        model_args.block_size,
        total_untouched_weights,
        total_original_weights,
    )

    print(f"{n_models=} | {total_new_blocks=} | {cr=} | {max_acc_drop=}")
    print(f"Compression ratios: {crs}")
    print(f"Accuracies: {accs}")
    print(f"Number of evaluations: {n_evalss}")
    print(f"Number of fails: {n_failss}")
    print(f"All blocks from base: {blockss_from_base}")
    print(f"Number of blocks from base: {len(blockss_from_base)}")


def deduplicate_blocks(
    model_args,
    data_args,
    training_args,
    model_storage,
    model_info,
    eval_fn,
    sensitivity_fn,
    distance_metric="l1",
):
    # Set parameters
    acc_threshold = model_info["acc_threshold"]
    data_args.task_name = model_info["task_name"]
    model_args.model_name_or_path = model_info["model_path"]
    model_range_start = model_storage["model_range"][1]
    model_range_end = model_storage["model_range"][2]
    candidate_range = model_range_end

    # Get initial model constitution
    model_constitution = list(range(model_range_start, model_range_end))

    # Prepare the candidate blocks
    all_blocks = model_storage["blocks"]
    candidate_blocks = all_blocks[:candidate_range]

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
    # Number of fails
    remain_fails = training_args.max_fails
    # Number of evaluations
    n_evals = 0

    if not hasattr(data_args, "noise") and training_args.extra_val_eps >= 0:
        scale = 2 / (data_args.val_size * training_args.val_epsilon1)
        data_args.noise = np.random.laplace(loc=0, scale=scale)
        print(f"Noise to threshold: {data_args.noise}")
        acc_threshold += data_args.noise

    # for i, sens, measure in zip(interate_seq, ordered_sensitivity, measures):
    for i in interate_seq:
        curr_iter += 1
        tobe_dedup_indices.add(i)

        block_2b_replaced = all_blocks[i]
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
            )
            n_evals += 1

            if training_args.extra_val_eps >= 0:
                scale = (
                    4
                    * training_args.max_fails
                    / (data_args.val_size * training_args.val_epsilon2)
                )
                noise = np.random.laplace(loc=0, scale=scale)
                print(f"Noise to acc: {noise}")
                acc += noise
            if acc > acc_threshold:
                dedup_indices |= tobe_dedup_indices
                used_allzero_indices |= used_allzero_indices_temp
                model_constitution = temp_constitution
            else:
                remain_fails -= 1

            tobe_dedup_indices = set()
            used_allzero_indices_temp = set()
            temp_constitution = model_constitution.copy()
            print(f"acc: {acc:.4f}, dedup success: {acc > acc_threshold}")
            print(f"Model constitution after dedup: {model_constitution}\n")
            if remain_fails == 0:
                break

    # Deduplicate all-zero sensitivity blocks
    print(f"Used all-zero indices: {list(used_allzero_indices)}")
    allzerograd_seq = [i for i in allzerograd_seq if i not in used_allzero_indices]

    # Start deduplication
    temp_constitution = model_constitution.copy()
    for i in allzerograd_seq:
        block_2b_replaced = all_blocks[i]
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
    )

    if acc > acc_threshold:
        model_constitution = temp_constitution

    if len(allzerograd_seq) == 0:
        print(
            f"{model_info['model_path']} dedup zero sensitivity blocks acc: {acc:.4f}, dedup success: {acc >= acc_threshold}"
        )
        print(f"Model constitution after dedup: {model_constitution}\n")

    n_fails = training_args.max_fails - remain_fails
    return model_constitution, acc, n_fails, n_evals
