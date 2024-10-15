import pdb
from copy import deepcopy
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
    save_model_storage,
    collect_storage,
    set_val_epsilon,
)
from utils.quantization import quant_and_dequant_model
from utils.pruning import prune_model, sparsify_model_storage

n_evals = 0


def run():
    model_args, data_args, training_args = parse_args()
    models_info = load_models_info(model_args)

    blockss_from_base, accs, n_new_blockss = [], [], []
    n_evalss, n_failss, crs = [], [], []

    n_base_blocks = 0
    total_untouched_weights, total_original_weights = 0, 0
    base_model_storage = {"blocks": None, "untouched_weights": None}
    for i in range(model_args.n_base_models):
        base_model = load_model(models_info[i])[0]

        # Block model
        if model_args.quantize:
            deq_model, quant_data, _ = quant_and_dequant_model(base_model)
            base_storage = block_model_1d(model_args.block_size, deq_model)
        elif model_args.prune:
            p_model = prune_model(base_model)
            base_storage = block_model_1d(model_args.block_size, p_model)
        else:
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

    if model_args.save_combined_storage:
        total_model_storage = {
            "blocks": base_storage["blocks"],
            "untouched_weights": [base_storage["untouched_weights"]],
            "model_constitution": [np.arange(n_blocks)],
        }

    # Deduplicate blocks for each model
    acc = accs[-1]
    for model_info in models_info[model_args.n_base_models :]:
        print(f"Model info: {model_info}")
        set_val_epsilon(training_args, model_info["budget"], models_info[0]["budget"])
        model, eval_fn, sensitivity_fn = load_model(model_info)
        curr_model_storage = block_model_1d(model_args.block_size, model)

        set_model_args(model_args, model, curr_model_storage)
        total_untouched_weights += model_args.untouched_weights
        total_original_weights += model_args.n_original_weights

        if model_args.quantize:
            save_path = model_info["model_path"].replace(".pt", "_wo_quant.npz")
            save_model_storage(curr_model_storage, save_path)
            deq_model, _, q_model = quant_and_dequant_model(model, quant_data)

            # Measure the accuracy of the model after quantization before deduplication
            deq_model_storage = block_model_1d(model_args.block_size, deq_model)
            nblocks = deq_model_storage["blocks"].shape[0]
            deq_model_storage["model_range"] = [0, nblocks - 1]
            deq_model_storage["untouched_weights"] = [
                deq_model_storage["untouched_weights"]
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
            for name, params in q_model_storage["untouched_weights"].items():
                q_model_storage["untouched_weights"][name] = params.astype(np.int8)
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
            p_model_storage["untouched_weights"] = [
                p_model_storage["untouched_weights"]
            ]
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
        # The following line guarantees the fairness rule
        acc_drop_threshold = (
            model_info["acc_drop_threshold"] - 0.003
            if training_args.extra_val_eps >= 0
            and model_args.task_type.startswith("vision")
            else model_info["acc_drop_threshold"]
        )
        set_acc = model_info["original_acc"] - acc_drop_threshold
        if training_args.enforce_fairness:
            model_info["acc_threshold"] = max(set_acc, acc)
        else:
            model_info["acc_threshold"] = set_acc
        model_constitution, acc, n_fails = deduplicate_blocks(
            model_args,
            data_args,
            training_args,
            model_storage,
            model_info,
            eval_fn,
            sensitivity_fn,
        )

        n_new_blocks, blocks_from_base, blocks_from_current = separate_blocks(
            model_constitution, n_base_blocks, return_new_blocks=True
        )
        n_new_blockss.append(n_new_blocks)
        blockss_from_base.append(blocks_from_base)

        cr = compute_compression_ratio(
            n_new_blocks,
            model_args.block_size,
            model_args.untouched_weights,
            model_args.n_original_weights,
        )
        global n_evals
        print(f"Current model {n_new_blocks=}, {cr=}, {acc=}, {n_fails=}, {n_evals=}")
        # Record the results
        accs.append(acc)
        n_evalss.append(n_evals)
        n_failss.append(n_fails)
        crs.append(cr)
        n_evals = 0

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
        if model_args.save_combined_storage:
            collect_storage(total_model_storage, curr_model_storage, model_constitution)

    if model_args.save_combined_storage:
        n_models = len(models_info)
        save_path = f"../models/{model_args.task_type}_{n_models}models_storage.npz"
        np.savez(save_path, **total_model_storage)

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

    n_dedup, model_constitution, remain_fails = recursive_deduplicate(
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
        training_args.max_fails,
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
    )

    if acc > acc_threshold:
        model_constitution = temp_constitution

    if len(allzerograd_seq) > 0:
        print(
            f"{model_info['model_path']} dedup zero sensitivity blocks acc: {acc:.4f}"
        )
        print(f"Model constitution after dedup: {model_constitution}")

    n_fails = training_args.max_fails - remain_fails
    return model_constitution, acc, n_fails


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
    remain_fails,
):

    # Base case
    if len(interate_seq) < training_args.min_dedup_len or (
        training_args.extra_val_eps >= 0 and remain_fails == 0
    ):
        return 0, model_constitution, remain_fails

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
    global n_evals
    n_evals += 1

    if training_args.extra_val_eps >= 0:
        if not hasattr(data_args, "noise"):
            scale = 2 / (data_args.val_size * training_args.val_epsilon1)
            data_args.noise = np.random.laplace(loc=0, scale=scale)
            print(f"Noise to threshold: {data_args.noise}")
            acc_threshold += data_args.noise

        scale = (
            4
            * training_args.max_fails
            / (data_args.val_size * training_args.val_epsilon2)
        )
        noise = np.random.laplace(loc=0, scale=scale)
        print(f"Noise to acc: {noise}")
        acc += noise

    success = False
    if acc > acc_threshold:
        dedup_indices |= tobe_dedup_indices
        model_constitution = temp_constitution
        success = True
    else:
        remain_fails -= 1

    print(f"acc: {acc:.4f}, dedup success: {success}")
    print(f"Model constitution after dedup: {model_constitution}")

    if not success:
        n_dedup_left, model_constitution, remain_fails = recursive_deduplicate(
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
            remain_fails,
        )

    if success or n_dedup_left > len(left_seq) // 2:
        n_dedup_right, model_constitution, remain_fails = recursive_deduplicate(
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
            remain_fails,
        )

        if success:
            return len(left_seq) + n_dedup_right, model_constitution, remain_fails
        else:
            return n_dedup_left + n_dedup_right, model_constitution, remain_fails
    else:
        return n_dedup_left, model_constitution, remain_fails
