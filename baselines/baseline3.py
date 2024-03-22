import pdb
import os
from collections import defaultdict

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np

from utils.parse_args import parse_args
from utils.blocker import get_blocks, block_model_1d
from utils import load_models_info
from text_task_utils.evaluate import evaluate
from sensitivity_measure import get_block_sensitivity


def run():
    """
    This is a hard-coded version of the baseline2 method:
    Self deduplicate lower budget model, and used as a reference,
    and then deduplcaite higher budget model (also allowing for self deduplication).
    """
    model_args, data_args, training_args = parse_args()
    models_info = load_models_info()
    model_paths = [info["model_path"] for info in models_info]
    model_storage = get_blocks(model_paths=model_paths)

    dedup_indices = set()
    dedup_dict = defaultdict(list)

    dedup_indices, dedup_dict = deduplicate_blocks(
        model_args,
        data_args,
        training_args,
        model_storage,
        models_info,
        dedup_indices,
        dedup_dict,
        0,
    )
    print(f"Number of changes: {len(dedup_indices)}")

    dedup_indices, _ = deduplicate_blocks(
        model_args,
        data_args,
        training_args,
        model_storage,
        models_info,
        dedup_indices,
        dedup_dict,
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
    dedup_dict,
    model_id,
    sort_by_magnitude=True,
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
    if sort_by_magnitude:
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
        # measures = model_storage["blocks"][model_range_start:model_range_end]

        if training_args.orderby == "l1_norm":
            block_magnitude = np.sum(np.abs(measures), axis=1)
        elif training_args.orderby == "l2_norm":
            block_magnitude = np.linalg.norm(measures, axis=1)
        elif training_args.orderby == "l_inf_norm":
            block_magnitude = np.max(np.abs(measures), axis=1)
        elif training_args.orderby == "3rd_quantile":
            block_magnitude = np.quantile(measures, 0.75, axis=1)

        # Because Wanda only applies to linear layers, we deduplicate them at last
        if training_args.sensitivity_measure == "wanda":
            embed_seq = interate_seq[:n_embed_blocks]
            other_seq = interate_seq[n_embed_blocks:]
            other_seq = other_seq[np.argsort(block_magnitude)]
            interate_seq = np.concatenate((other_seq, embed_seq))
        else:
            interate_seq = interate_seq[np.argsort(block_magnitude)]

    # search_range = model_storage["search_range"]
    for i in interate_seq:
        block_2b_replaced = model_storage["blocks"][i]
        # Sort by some metrics: l1, l2, cosine
        if distance_metric == "l1":
            diff = np.sum(np.abs(candidate_blocks - block_2b_replaced), axis=1)
        elif distance_metric == "l2":
            diff = np.linalg.norm(candidate_blocks - block_2b_replaced, axis=1)
        elif distance_metric == "cosine":
            diff = np.dot(candidate_blocks, block_2b_replaced) / (
                np.linalg.norm(candidate_blocks) * np.linalg.norm(block_2b_replaced)
            )
        else:
            raise ValueError(f"Invalid distance metric: {distance_metric}")

        # # This part is just for experiment. Will not be used in the final version.
        # if model_id == 0:
        #     diff[0 : search_range[i, 0]] = np.inf
        #     diff[search_range[i, 1] :] = np.inf
        # else:
        #     diff[0 : search_range[i - 833, 0]] = np.inf
        #     diff[search_range[i - 833, 1] : search_range[i - 833, 2]] = np.inf
        #     diff[search_range[i - 833, 3] :] = np.inf

        ind = diff.argsort()
        for j in ind:
            # j += model_range_start
            if j != i and j not in dedup_indices:
                temp_constitution = model_constitution.copy()
                # Replace the current block with the most similar block
                temp_constitution[i - model_range_start] = j
                # If the current block was used to replace other blocks,
                # replace those blocks with the most similar block j
                # TODO: Correct only if all models do not use blocks from later models.
                # Otherwise, all model that are changed should be tested.
                if i in dedup_dict:
                    for k in dedup_dict[i]:
                        idx = k - model_range_start
                        if idx >= 0 and idx < len(temp_constitution):
                            temp_constitution[idx] = j

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
                    dedup_dict[j].append(i)
                    if i in dedup_dict:
                        dedup_dict[j].extend(dedup_dict[i])
                        del dedup_dict[i]
                print(f"Model {model_id} block {i} -> {j} acc: {acc:.4f}")
                print(f"Model constitution after dedup: {model_constitution}")
                break
    return dedup_indices, dedup_dict
