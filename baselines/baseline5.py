import pdb
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np

from utils.parse_args import parse_args
from utils.blocker import get_blocks
from utils import load_models_info
from text_task_utils.evaluate import evaluate
from sensitivity_measure import get_block_sensitivity


def run():
    """
    This is a hard-coded version of the baseline5 method:
    Order by sensitivity, and leave the all-zero sensitivity blocks to the end.
    """
    model_args, data_args, training_args = parse_args()
    models_info = load_models_info()
    model_paths = [info["model_path"] for info in models_info]
    model_storage = get_blocks(model_paths=model_paths)

    dedup_indices = set()
    # dedup_dict = defaultdict(list)

    dedup_indices = deduplicate_blocks(
        model_args,
        data_args,
        training_args,
        model_storage,
        models_info,
        dedup_indices,
        # dedup_dict,
        0,
    )
    print(f"Number of changes: {len(dedup_indices)}")

    dedup_indices = deduplicate_blocks(
        model_args,
        data_args,
        training_args,
        model_storage,
        models_info,
        dedup_indices,
        # dedup_dict,
        1,
    )
    print(f"Number of changes: {len(dedup_indices)}")


def deduplicate_blocks(
    model_args,
    data_args,
    training_args,
    model_storage,
    models_info,
    dedup_indices,
    model_id,
    distance_metric="l1",
):
    # Set parameters
    model_info = models_info[model_id]
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

    task_name = model_info["task_name"]
    eps = model_info["budget"]
    return_n_embed_blocks = training_args.sensitivity_measure == "wanda"
    measures, n_embed_blocks = get_block_sensitivity(
        task_name,
        training_args.sensitivity_measure,
        eps,
        skip_embeds=False,
        return_n_embed_blocks=return_n_embed_blocks,
    )

    all_zeros = np.all(measures == 0, axis=1)
    non_all_zeros = ~all_zeros
    allzero_indices = np.where(all_zeros != 0)[0]
    nonzero_indices = np.nonzero(non_all_zeros)[0]
    allzero_indices_set = set(allzero_indices)
    used_allzero_indices = set()
    print(f"Size of all zero sensitivity blocks : {len(allzero_indices)}")
    print(f"All zero sensitivity blocks : {allzero_indices}")

    seq_to_order = interate_seq[nonzero_indices]
    allzerograd_seq = interate_seq[allzero_indices]
    measures = measures[nonzero_indices]

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
    ordered_sensitivity = [round(m, 6) for m in block_sens[ordered_indices]]

    measures = measures[ordered_indices]

    avg_distances = []
    total_diffs = []
    # search_range = model_storage["search_range"]
    for i, sens, measure in zip(interate_seq, ordered_sensitivity, measures):
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

        # # This part is just for experiment. Will not be used in the final version.
        # if model_id == 0:
        #     dist[0 : search_range[i, 0]] = np.inf
        #     dist[search_range[i, 1] :] = np.inf
        # else:
        #     dist[0 : search_range[i - 833, 0]] = np.inf
        #     dist[search_range[i - 833, 1] : search_range[i - 833, 2]] = np.inf
        #     dist[search_range[i - 833, 3] :] = np.inf

        ind = dist.argsort()
        for j in ind:
            if j == i or j in dedup_indices:
                continue
            if j in allzero_indices_set:
                used_allzero_indices.add(j)

            avg_distance = round(dist[j] / len(block_2b_replaced), 4)
            total_diff = np.dot(measure, diff[j])
            avg_distances.append(avg_distance)
            total_diffs.append(total_diff)

            # Replace the current block with the most similar block
            temp_constitution = [j if c == i else c for c in model_constitution]

            acc = evaluate(
                model_storage,
                model_id,
                temp_constitution,
                data_args,
                model_args,
                training_args,
            )
            if acc >= acc_threshold:
                model_constitution = temp_constitution
                dedup_indices.add(i)
            print(f"Model {model_id} block {i} -> {j} acc: {acc:.4f}")
            print(f"{avg_distance=}, {sens=}, {total_diff=}")
            print(f"Model constitution after dedup: {model_constitution}")
            break

    # Deduplicate all-zero sensitivity blocks
    print(f"Used all-zero indices: {list(used_allzero_indices)}")
    allzerograd_seq = [i for i in allzerograd_seq if i not in used_allzero_indices]

    # Start deduplication
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

        ind = dist.argsort()
        for j in ind:
            if j in allzero_indices_set or j in dedup_indices:
                continue
            # Replace the current block with the most similar block
            temp_constitution = [j if c == i else c for c in model_constitution]
            break

    acc = evaluate(
        model_storage,
        model_id,
        temp_constitution,
        data_args,
        model_args,
        training_args,
    )
    if acc >= acc_threshold:
        model_constitution = temp_constitution
        dedup_indices |= set(allzerograd_seq)
    print(f"Model {model_id} dedup zero sensitivity blocks acc: {acc:.4f}")
    print(f"Model constitution after dedup: {model_constitution}")

    print(f"Block senstivity {training_args.orderby}:\n{ordered_sensitivity}")
    print(f"Average distances: {avg_distances}")
    print(f"Total diffs: {total_diffs}")
    return dedup_indices
