import pdb
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np

from utils.parse_args import parse_args
from utils.blocker import get_blocks
from utils import load_models_info
from text_task_utils.evaluate import evaluate


# def run(acc_drop_threshold=0.02, original_acc=0.89):
#     """
#     This is a hard-coded version of the baseline1 method:
#     Fix low budget model weights as the reference model, deduplicate the highe budget model.
#     """
#     model_args, data_args, training_args = parse_args()
#     models_info = load_models_info(model_args)
#     model_paths = [info["model_path"] for info in models_info]
#     model_storage = get_blocks(model_paths=model_paths)

#     model0_range_start = model_storage["model_range"][0]
#     model0_range_end = model_storage["model_range"][1]
#     model1_range_start = model_storage["model_range"][1]
#     model1_range_end = model_storage["model_range"][2]

#     model0_blocks = model_storage["blocks"][model0_range_start:model0_range_end]
#     model1_constitution = list(range(model1_range_start, model1_range_end))

#     n_changes = 0

#     for i in range(model1_range_start, model1_range_end):
#         block_2b_replaced = model_storage["blocks"][i]
#         diff = np.sum(np.abs(model0_blocks - block_2b_replaced), axis=1, keepdims=False)
#         most_similar_idx = np.argmin(diff)

#         model1_constitution[i - model1_range_start] = most_similar_idx
#         acc = evaluate(
#             model_storage,
#             1,
#             model1_constitution,
#             data_args,
#             model_args,
#             training_args,
#         )
#         if acc < original_acc - acc_drop_threshold:
#             # Revert the change
#             model1_constitution[i - model1_range_start] = i
#         else:
#             n_changes += 1

#     print(f"Number of changes: {n_changes}")


def run(acc_drop_threshold=0.02, original_acc=0.89):
    model_args, data_args, training_args = parse_args()
    models_info = load_models_info(model_args)
    model_paths = [info["model_path"] for info in models_info]
    model_storage = get_blocks(model_paths=model_paths)

    model0_range_start = model_storage["model_range"][0]
    model0_range_end = model_storage["model_range"][1]
    model1_range_start = model_storage["model_range"][1]
    model1_range_end = model_storage["model_range"][2]

    acc_threshold = original_acc - acc_drop_threshold
    model_id = 1
    n_change = deduplicate_blocks(
        model_args,
        data_args,
        training_args,
        model_storage,
        acc_threshold,
        model_id,
        model1_range_start,
        model1_range_end,
        model0_range_end,
        sort_by_magnitude=False,
    )
    print(f"Number of changes: {n_change}")


def deduplicate_blocks(
    model_args,
    data_args,
    training_args,
    model_storage,
    acc_threshold,
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

    n_change = 0
    for i in interate_seq:
        block_2b_replaced = model_storage["blocks"][i]
        diff = np.sum(
            np.abs(candidate_blocks - block_2b_replaced), axis=1, keepdims=False
        )

        # Replace the current block with the most similar block
        temp_constitution = model_constitution.copy()
        temp_constitution[i - model_range_start] = np.argmin(diff)

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
            n_change += 1
    return n_change
