import pdb
import os
from collections import defaultdict

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np

from utils.parse_args import parse_args
from utils.blocker import get_blocks
from utils import load_models_info
from text_task_utils.evaluate import evaluate


def run(acc_drop_threshold=0.02, original_acc=0.89, sort_by_magnitude=True):
    """
    This is a hard-coded version of the baseline2 method:
    Self deduplicate lower budget model, and used as a reference,
    and then deduplcaite higher budget model (also allowing for self deduplication).
    """
    acc_threshold = original_acc - acc_drop_threshold
    model_args, data_args, training_args = parse_args()
    models_info = load_models_info(model_args)
    model_paths = [info["model_path"] for info in models_info]
    model_storage = get_blocks(model_paths=model_paths)

    model0_range_start = model_storage["model_range"][0]
    model0_range_end = model_storage["model_range"][1]
    model1_range_start = model_storage["model_range"][1]
    model1_range_end = model_storage["model_range"][2]

    dedup_indices = set()
    dedup_dict = defaultdict(list)

    # candidate_blocks = model_storage["blocks"][model0_range_start:model0_range_end]
    candidate_range = model0_range_end
    dedup_indices, dedup_dict = deduplicate_blocks(
        model_args,
        data_args,
        training_args,
        model_storage,
        acc_threshold,
        dedup_indices,
        dedup_dict,
        0,
        model0_range_start,
        model0_range_end,
        candidate_range,
        sort_by_magnitude,
    )

    # candidate_blocks = model_storage["blocks"]
    candidate_range = model1_range_end
    dedup_indices, _ = deduplicate_blocks(
        model_args,
        data_args,
        training_args,
        model_storage,
        acc_threshold,
        dedup_indices,
        dedup_dict,
        1,
        model1_range_start,
        model1_range_end,
        candidate_range,
        sort_by_magnitude,
    )

    print(f"Number of changes: {len(dedup_indices)}")


def deduplicate_blocks(
    model_args,
    data_args,
    training_args,
    model_storage,
    acc_threshold,
    dedup_indices,
    dedup_dict,
    model_id,
    model_range_start,
    model_range_end,
    candidate_range,
    sort_by_magnitude,
):
    model_constitution = list(range(model_range_start, model_range_end))

    # Prepare the candidate blocks
    candidate_blocks = model_storage["blocks"][:candidate_range]
    # overlapped = False if candidate_range <= model_range_start else True

    # The order the blocks are iterated through
    interate_seq = np.arange(model_range_start, model_range_end)
    if sort_by_magnitude:
        if training_args.orderby == "l2_norm":
            block_magnitude = np.linalg.norm(
                model_storage["blocks"][model_range_start:model_range_end], axis=1
            )
        elif training_args.orderby == "3rd_quantile":
            block_magnitude = np.quantile(
                model_storage["blocks"][model_range_start:model_range_end], 0.75, axis=1
            )
        interate_seq = interate_seq[np.argsort(block_magnitude)]

    for i in interate_seq:
        block_2b_replaced = model_storage["blocks"][i]
        # Sort by l1 distance
        diff = np.sum(
            np.abs(candidate_blocks - block_2b_replaced), axis=1, keepdims=False
        )
        # ind = np.argpartition(diff, 20)[:20]
        # # The most similar block is the block itself, so get rid of it
        # ind = ind[np.argsort(diff[ind])][1:]

        # ind = diff.argsort()[1:] if overlapped else [np.argmin(diff)]
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
