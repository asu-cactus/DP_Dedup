import pdb
import json
import pickle
from copy import copy

import numpy as np

from utils.parse_args import parse_args
from utils.blocker import get_blocks
from utils import load_models_info
from text_task_utils.evaluate import evaluate


def get_heuristics_dict():
    """
    This is a hard-coded version of the baseline2 method:
    Self deduplicate lower budget model, and used as a reference,
    and then deduplcaite higher budget model (also allowing for self deduplication).
    """

    model_args, data_args, training_args = parse_args()
    models_info = load_models_info(model_args)
    model_paths = [info["model_path"] for info in models_info]
    model_storage = get_blocks(model_paths=model_paths)

    heuristics_dict = {}
    last_legal_model = find_last_legal_model(models_info)

    for model_id, last_model_id in last_legal_model.items():
        acc_dict = get_acc_after_dedup(
            model_args,
            data_args,
            training_args,
            model_storage,
            model_id,
            last_model_id,
        )
        heuristics_dict.update(acc_dict)
    # Save the heuristics_dict with pickle
    with open("heuristics_dict.pkl", "wb") as f:
        pickle.dump(heuristics_dict, f)
    # Save the heuristics_dict with json
    with open("heuristics_dict.json", "w") as f:
        json.dump(heuristics_dict, f)
    return heuristics_dict


def find_last_legal_model(models_info):
    """Find the legal models to deduplicate.
    For example, if the budget is [1, 2, 3, 4, 5], then the legal model range is: {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}.
    If the budget is [1, 1, 2, 3, 3, 3, 4], then the legal model range is: {0: 1, 1: 1, 2: 2, 3: 5, 4: 5, 5: 5, 6: 6}.
    """
    budgets = [info["budget"] for info in models_info]

    legal_model_action = {}
    current_budget = budgets[0]
    idx = 0
    for i, budget in enumerate(budgets):
        if budget > current_budget:
            for j in range(idx, i):
                legal_model_action[j] = i - 1
            current_budget = budget
            idx = i
    for j in range(idx, i + 1):
        legal_model_action[j] = i
    print(f"legal_model_action:\n{legal_model_action}")
    return legal_model_action


def get_acc_after_dedup(
    model_args,
    data_args,
    training_args,
    model_storage,
    model_id,
    last_model_id,
    top_k=5,
):
    """
    Get the accuracy after deduplication.
    The result is a dictionary, where the key is the block index, and the value is another dictionary,
    where the key is the block index of the candidate block, and the value is the accuracy after deduplication.
    """
    models_range = model_storage["model_range"]
    blocks = model_storage["blocks"]

    model_range_start = models_range[model_id]
    model_range_end = models_range[model_id + 1]
    model_constitution = list(range(model_range_start, model_range_end))

    result_dict = {}
    for i in range(model_range_start, model_range_end):
        block_2b_replaced = blocks[i]
        for target_model_id in range(last_model_id):
            target_model_range_start = models_range[target_model_id]
            target_model_range_end = models_range[target_model_id + 1]
            candidate_blocks = blocks[target_model_range_start:target_model_range_end]

            diff = np.sum(
                np.abs(candidate_blocks - block_2b_replaced), axis=1, keepdims=False
            )
            ind = np.argpartition(diff, top_k + 1)[: top_k + 1]
            ind = ind[np.argsort(diff[ind])]
            ind = ind[1:] if model_id == target_model_id else ind[:top_k]
            ind = [i + target_model_range_start for i in ind]

            action_to_acc_dict = {}
            for j in ind:

                temp_constitution = copy(model_constitution)
                temp_constitution[i - model_range_start] = j

                acc = evaluate(
                    model_storage,
                    model_id,
                    temp_constitution,
                    data_args,
                    model_args,
                    training_args,
                )
                action_to_acc_dict[j] = acc

            result_dict[i] = action_to_acc_dict
    return result_dict
