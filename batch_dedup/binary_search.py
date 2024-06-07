import pdb
from copy import deepcopy
import numpy as np

from utils.parse_args import parse_args
from utils.blocker import block_model_1d
from utils.common import (
    load_models_info,
    load_model,
    merge_model_storage,
    separate_blocks,
    compute_compression_ratio,
    set_model_args,
    save_model_storage,
)
from utils.quantization import quant_and_dequant_model
from utils.pruning import prune_model, sparsify_model_storage


def run():
    model_args, data_args, training_args = parse_args()
    models_info = load_models_info(model_args)

    base_model = load_model(models_info[0], model_args)[0]
    # Block model
    if model_args.quantize and model_args.prune:
        raise ValueError(
            "Current implementation doesn't support quantization and pruning at the same time"
        )
    if model_args.quantize:
        deq_model, quant_data, _ = quant_and_dequant_model(base_model)
        base_model_storage = block_model_1d(model_args.block_size, deq_model)
    elif model_args.prune:
        p_model = prune_model(base_model)
        base_model_storage = block_model_1d(model_args.block_size, p_model)
    else:
        base_model_storage = block_model_1d(model_args.block_size, base_model)

    n_base_blocks = base_model_storage["blocks"].shape[0]
    set_model_args(model_args, base_model, base_model_storage)

    # Deduplicate blocks for each model
    total_new_blocks = 0
    blockss_from_base = set()
    max_acc_drop = 0.0
    for model_info in models_info[1:]:
        print(f"Model info: {model_info}")
        model, eval_fn, train_fn, sensitivity_fn = load_model(model_info, model_args)
        curr_model_storage = block_model_1d(model_args.block_size, model)
        if model_args.quantize:
            save_path = model_info["model_path"].replace(".pt", "_wo_quant.npz")
            save_model_storage(curr_model_storage, save_path)
            deq_model, _, q_model = quant_and_dequant_model(model, quant_data)

            # Measure the accuracy of the model after quantization before deduplication
            deq_model_storage = block_model_1d(model_args.block_size, deq_model)
            nblocks = deq_model_storage["blocks"].shape[0]
            deq_model_storage["model_range"] = [0, nblocks - 1]
            deq_model_storage["untouch_weights"] = [
                deq_model_storage["untouch_weights"]
            ]
            eval_fn(
                data_args,
                model_args,
                training_args,
                model_info,
                list(range(nblocks - 1)),
                deq_model_storage,
                0,
            )

            # Make sure blocks are integers.
            q_model_storage = block_model_1d(model_args.block_size, q_model)
            q_model_storage["blocks"] = q_model_storage["blocks"].astype(np.int8)
            for name, params in q_model_storage["untouch_weights"].items():
                q_model_storage["untouch_weights"][name] = params.astype(np.int8)
            save_path = model_info["model_path"].replace(".pt", "_w_quant.npz")
            save_model_storage(q_model_storage, save_path)
            curr_model_storage = block_model_1d(model_args.block_size, deq_model)
        if model_args.prune:
            save_path = model_info["model_path"].replace(".pt", "_wo_prune.npz")
            save_model_storage(curr_model_storage, save_path)
            p_model = prune_model(model)

            # Measure the accuracy of the model after quantization before deduplication
            p_model_storage = block_model_1d(model_args.block_size, p_model)
            nblocks = p_model_storage["blocks"].shape[0]
            p_model_storage["model_range"] = [0, nblocks - 1]
            p_model_storage["untouch_weights"] = [p_model_storage["untouch_weights"]]
            eval_fn(
                data_args,
                model_args,
                training_args,
                model_info,
                list(range(nblocks - 1)),
                p_model_storage,
                0,
            )

            curr_model_storage = deepcopy(p_model_storage)

            sparsify_model_storage(p_model_storage)
            save_path = model_info["model_path"].replace(".pt", "_w_prune.npz")
            save_model_storage(p_model_storage, save_path)

        model_storage = merge_model_storage(base_model_storage, curr_model_storage)
        model_constitution, acc_drop = deduplicate_blocks(
            model_args,
            data_args,
            training_args,
            model_storage,
            model_info,
            1,
            eval_fn,
            train_fn,
            sensitivity_fn,
        )
        max_acc_drop = max(max_acc_drop, acc_drop)
        n_new_blocks, blocks_from_base, blocks_from_current = separate_blocks(
            model_constitution, n_base_blocks, return_new_blocks=True
        )
        total_new_blocks += n_new_blocks
        blockss_from_base |= blocks_from_base

        if model_args.quantize:
            blocks_from_current = list(blocks_from_current)
            curr_model_storage["blocks"] = model_storage["blocks"][blocks_from_current]
            save_path = model_info["model_path"].replace(".pt", "_quant_dedup.npz")
            save_model_storage(curr_model_storage, save_path)
        if model_args.prune:
            blocks_from_current = list(blocks_from_current)
            curr_model_storage["blocks"] = model_storage["blocks"][blocks_from_current]
            sparsify_model_storage(curr_model_storage)
            save_path = model_info["model_path"].replace(".pt", "_prune_dedup.npz")
            save_model_storage(curr_model_storage, save_path)

    n_models = len(models_info) - 1
    cr = compute_compression_ratio(
        total_new_blocks,
        model_args.block_size,
        model_args.untouched_weights,
        model_args.n_original_weights,
        n_models,
    )
    print(f"{total_new_blocks=} | {cr=} | {max_acc_drop=}")
    print(f"All blocks from base: {blockss_from_base}")
    print(f"Number of blocks from base: {len(blockss_from_base)}")


def get_used_allzero_indices(
    model_constitution, allzero_indices_set, model_range_start
):
    used_allzero_indices = set()
    for i, block in enumerate(model_constitution):
        if block in allzero_indices_set and block - model_range_start != i:
            used_allzero_indices.add(block)
    return used_allzero_indices


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

    # Deduplicate non-all-zero sensitivity blocks
    dedup_indices = set()
    n_dedup, model_constitution = recursive_deduplicate(
        model_args,
        data_args,
        training_args,
        interate_seq,
        model_storage,
        candidate_blocks,
        model_constitution,
        model_info,
        distance_metric,
        acc_threshold,
        dedup_indices,
        eval_fn,
        train_fn,
    )

    # Deduplicate all-zero sensitivity blocks
    used_allzero_indices = get_used_allzero_indices(
        model_constitution, allzero_indices_set, model_range_start
    )
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

    if len(allzerograd_seq) > 0:
        print(
            f"{model_info['model_path']} dedup zero sensitivity blocks acc: {acc:.4f}"
        )
        print(f"Model constitution after dedup: {model_constitution}")

    return model_constitution, acc_drop


def recursive_deduplicate(
    model_args,
    data_args,
    training_args,
    interate_seq,
    model_storage,
    candidate_blocks,
    model_constitution,
    model_info,
    distance_metric,
    acc_threshold,
    dedup_indices,
    eval_fn,
    train_fn,
):

    # Base case
    if len(interate_seq) < training_args.min_dedup_len:
        return 0, model_constitution

    mid_point = len(interate_seq) // 2
    left_seq = interate_seq[:mid_point]
    right_seq = interate_seq[mid_point:]

    # Initialize variables
    #  Deduplicated indices
    tobe_dedup_indices = set()
    # Constitution to be evaluated
    temp_constitution = model_constitution.copy()

    for i in left_seq:
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

        # Replace the current block with the most similar block
        print(f"{model_info['model_path']} block {i} -> {j}")
        temp_constitution = [j if c == i else c for c in temp_constitution]
    acc = eval_fn(
        data_args,
        model_args,
        training_args,
        model_info,
        temp_constitution,
        model_storage,
        1,
    )

    success = False
    if acc >= acc_threshold:
        dedup_indices |= tobe_dedup_indices
        model_constitution = temp_constitution
        success = True
        if training_args.do_finetune:
            train_fn(
                data_args,
                model_args,
                training_args,
                model_info,
                temp_constitution,
                model_storage,
                1,
            )

    print(f"acc: {acc:.4f}, dedup success: {success}")
    print(f"Model constitution after dedup: {model_constitution}")

    if not success:
        n_dedup_left, model_constitution = recursive_deduplicate(
            model_args,
            data_args,
            training_args,
            left_seq,
            model_storage,
            candidate_blocks,
            model_constitution,
            model_info,
            distance_metric,
            acc_threshold,
            dedup_indices,
            eval_fn,
            train_fn,
        )

    if success or n_dedup_left > len(left_seq) // 2:
        n_dedup_right, model_constitution = recursive_deduplicate(
            model_args,
            data_args,
            training_args,
            right_seq,
            model_storage,
            candidate_blocks,
            model_constitution,
            model_info,
            distance_metric,
            acc_threshold,
            dedup_indices,
            eval_fn,
            train_fn,
        )
        if success:
            return len(left_seq) + n_dedup_right, model_constitution
        else:
            return n_dedup_left + n_dedup_right, model_constitution
    else:
        return n_dedup_left, model_constitution
