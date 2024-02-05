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


def run(acc_drop_threshold=0.02, original_acc=0.89):
    """
    This is a hard-coded version of the baseline1 method:
    Fix low budget model weights as the reference model, deduplicate the highe budget model.
    """
    model_args, data_args, training_args = parse_args()
    models_info = load_models_info(model_args)
    model_paths = [info["model_path"] for info in models_info]
    model_storage = get_blocks(model_paths=model_paths)

    model0_range_start = model_storage["model_range"][0]
    model0_range_end = model_storage["model_range"][1]
    model1_range_start = model_storage["model_range"][1]
    model1_range_end = model_storage["model_range"][2]

    model0_blocks = model_storage["blocks"][model0_range_start:model0_range_end]
    model1_constitution = list(range(model1_range_start, model1_range_end))

    n_changes = 0
    # For each block in the first model, find the most similar block in the second model according to absolute error
    for i in range(model1_range_start, model1_range_end):
        block_2b_replaced = model_storage["blocks"][i]
        diff = np.sum(np.abs(model0_blocks - block_2b_replaced), axis=1, keepdims=False)
        most_similar_idx = np.argmin(diff)

        model1_constitution[i - model1_range_start] = most_similar_idx
        acc = evaluate(
            model_storage,
            1,
            model1_constitution,
            data_args,
            model_args,
            training_args,
        )
        if acc < original_acc - acc_drop_threshold:
            # Revert the change
            model1_constitution[i - model1_range_start] = i
        else:
            n_changes += 1

    print(f"Number of changes: {n_changes}")
