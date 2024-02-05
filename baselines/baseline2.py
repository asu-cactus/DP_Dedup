import json
import pdb
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np

from parse_args import parse_args
from blocker import get_blocks
from text_task_utils.evaluate import evaluate


def load_models_info(model_args) -> list[dict]:
    with open("models/model_info.json", "r") as f:
        models_info = json.load(f)

    model_ids = model_args.model_ids.split(",")
    models_info = [models_info[idx] for idx in model_ids]
    return models_info


def run(acc_drop_threshold=0.02, original_acc=0.89, sort_by_magnitude=False):
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

    # candidate_blocks = model_storage["blocks"][model0_range_start:model0_range_end]
    candidate_range = model0_range_end
    dedup_indices = deduplicate_blocks(
        model_args,
        data_args,
        training_args,
        model_storage,
        acc_threshold,
        dedup_indices,
        0,
        model0_range_start,
        model0_range_end,
        candidate_range,
        sort_by_magnitude,
    )

    # candidate_blocks = model_storage["blocks"]
    candidate_range = model1_range_end
    dedup_indices = deduplicate_blocks(
        model_args,
        data_args,
        training_args,
        model_storage,
        acc_threshold,
        dedup_indices,
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
    model_id,
    model_range_start,
    model_range_end,
    candidate_range,
    sort_by_magnitude=False,
):
    model_constitution = list(range(model_range_start, model_range_end))

    # Prepare the candidate blocks
    candidate_blocks = model_storage["blocks"][:candidate_range]
    # overlapped = False if candidate_range <= model_range_start else True

    # The order the blocks are iterated through
    interate_seq = np.arange(model_range_start, model_range_end)
    if sort_by_magnitude:
        # Sort the iterations by the magnitude of the block
        block_magnitude = np.linalg.norm(
            model_storage["blocks"][model_range_start:model_range_end],
            axis=1,
            keepdims=False,
        )
        interate_seq = interate_seq[np.argsort(block_magnitude)]

    for i in interate_seq:
        block_2b_replaced = model_storage["blocks"][i]
        diff = np.sum(
            np.abs(candidate_blocks - block_2b_replaced), axis=1, keepdims=False
        )
        # ind = np.argpartition(diff, 20)[:20]
        # # The most similar block is the block itself, so get rid of it
        # ind = ind[np.argsort(diff[ind])][1:]

        # ind = diff.argsort()[1:] if overlapped else [np.argmin(diff)]
        ind = diff.argsort()[1:]
        for j in ind:
            # j += model_range_start
            if j != i and j not in dedup_indices:
                model_constitution[i - model_range_start] = j
                acc = evaluate(
                    model_storage,
                    model_id,
                    model_constitution,
                    data_args,
                    model_args,
                    training_args,
                )
                if acc < acc_threshold:
                    # Revert the change
                    model_constitution[i - model_range_start] = i
                else:
                    dedup_indices.add(i)
                break
    return dedup_indices
