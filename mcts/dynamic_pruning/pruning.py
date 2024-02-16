import pdb
import pickle
import os

import numpy as np

from text_task_utils.evaluate import evaluate


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
    models_storage,
    model_id,
    last_model_id,
):
    """
    Get the accuracy after deduplication.
    The result is a dictionary, where the key is the block index, and the value is another dictionary,
    where the key is the block index of the candidate block, and the value is the accuracy after deduplication.
    """
    models_range = models_storage["model_range"]
    blocks = models_storage["blocks"]
    # top_k = training_args.top_k

    model_range_start = models_range[model_id]
    model_range_end = models_range[model_id + 1]
    model_constitution = list(range(model_range_start, model_range_end))

    heuristics_dict = {}
    for i in range(model_range_start, model_range_end):
        action_to_acc_dict = {}
        block_2b_replaced = blocks[i]
        for target_model_id in range(last_model_id + 1):
            target_model_range_start = models_range[target_model_id]
            target_model_range_end = models_range[target_model_id + 1]
            candidate_blocks = blocks[target_model_range_start:target_model_range_end]

            diff = np.sum(
                np.abs(candidate_blocks - block_2b_replaced), axis=1, keepdims=False
            )
            ind = np.argpartition(diff, 2)[:2]
            ind = ind[np.argsort(diff[ind])]
            most_similar_block = ind[0] if ind[0] != i else ind[1]
            most_similar_block += target_model_range_start
            # ind = ind[1:] if model_id == target_model_id else ind[:top_k]
            # ind = [i + target_model_range_start for i in ind]

            temp_constitution = model_constitution.copy()
            temp_constitution[i - model_range_start] = most_similar_block

            acc = evaluate(
                models_storage,
                model_id,
                temp_constitution,
                data_args,
                model_args,
                training_args,
            )
            action_to_acc_dict[most_similar_block] = acc

        heuristics_dict[i] = action_to_acc_dict
    return heuristics_dict


def get_heuristics_dict(
    model_args,
    data_args,
    training_args,
    models_info,
    models_storage,
) -> dict[int, dict[int, float]]:
    """
    This is a hard-coded version of the baseline2 method:
    Self deduplicate lower budget model, and used as a reference,
    and then deduplcaite higher budget model (also allowing for self deduplication).
    """

    # Load heursitics_dict from pickle if it exists
    if os.path.exists("heuristics_dict.pkl"):
        with open("heuristics_dict.pkl", "rb") as f:
            heuristics_dict = pickle.load(f)
        print("Loaded heuristics_dict from pickle")
        return heuristics_dict

    heuristics_dict = {}
    last_legal_model = find_last_legal_model(models_info)

    for model_id, last_model_id in last_legal_model.items():
        acc_dict = get_acc_after_dedup(
            model_args,
            data_args,
            training_args,
            models_storage,
            model_id,
            last_model_id,
        )
        heuristics_dict.update(acc_dict)
    # Save the heuristics_dict with pickle
    with open("heuristics_dict.pkl", "wb") as f:
        pickle.dump(heuristics_dict, f)
    print("Saved heuristics_dict with pickle")
    return heuristics_dict


def get_heuristic_info(
    model_args,
    data_args,
    training_args,
    models_info,
    models_storage,
) -> dict[int, dict[int, tuple[int, float]]]:
    """Get the heuristic information for the MCTS."""
    heuristics_dict = get_heuristics_dict(
        model_args, data_args, training_args, models_info, models_storage
    )
    acc_thresholds = [
        info["original_acc"] - info["acc_drop_threshold"] for info in models_info
    ]

    all_legal_actions = {}
    for block_2b_replaced, value in heuristics_dict.items():
        n_target_models = len(value) // 5
        for model_id in range(n_target_models):
            to_sort = list(value.items())[model_id * 5 : (model_id + 1) * 5]
            block_to_replace, acc = max(to_sort, key=lambda x: x[1])
            # In the future use the following line to replace the above two lines
            # block_to_replace, acc = value.items()[model_id]
            if acc >= acc_thresholds[0]:
                if block_2b_replaced not in all_legal_actions:
                    all_legal_actions[block_2b_replaced] = {}
                all_legal_actions[block_2b_replaced][model_id] = (block_to_replace, acc)

    print(f"all legal 1st sub actions: {list(all_legal_actions.keys())}")
    print(f"all_legal_actions:\n{all_legal_actions}")
    return all_legal_actions
