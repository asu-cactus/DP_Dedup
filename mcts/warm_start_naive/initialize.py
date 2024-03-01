import pdb
import pickle
import os

import numpy as np

from text_task_utils.evaluate import evaluate
import final_constitution_quantile as fcq


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


def _print_all_action_space(all_legal_actions):
    for model_id, value in all_legal_actions.items():
        print(f"Model {model_id} action space:")
        print(f"Original action space width: {len(value)}")
        for block_2b_replaced, blocks_to_replace in value.items():
            print(f"{block_2b_replaced}: {blocks_to_replace[:10]}")


def get_heuristic_info(models_storage) -> dict[int, dict[int, list[int]]]:
    """Get the heuristic information for the MCTS."""
    file_name = "all_legal_actions_self_dedup.pkl"

    # Load all_legal_actions from pickle if it exists
    if os.path.exists(file_name):
        with open(file_name, "rb") as f:
            all_legal_actions = pickle.load(f)
        print("Loaded all_legal_actions from pickle")
        _print_all_action_space(all_legal_actions)
        return all_legal_actions

    blocks = models_storage["blocks"]
    models_range = models_storage["model_range"]
    all_legal_actions = {0: {}, 1: {}}

    for model_id, model_constitute in enumerate([fcq.model0, fcq.model1]):
        model_range_start = models_range[model_id]
        model_range_end = models_range[model_id + 1]
        unique_block_ids = list(set(model_constitute))
        # Filter blocks that are belong to the current model with model_id
        blocks_2b_replaced_id = [
            block_id
            for block_id in unique_block_ids
            if block_id >= model_range_start and block_id < model_range_end
        ]
        candidate_blocks = blocks[unique_block_ids]

        for block_2b_replaced_id in blocks_2b_replaced_id:
            block_2b_replaced = blocks[block_2b_replaced_id]

            # Compute l1 distance
            diff = np.sum(
                np.abs(candidate_blocks - block_2b_replaced),
                axis=1,
                keepdims=False,
            )

            blocks_to_replace = [unique_block_ids[i] for i in np.argsort(diff)]
            blocks_to_replace.remove(block_2b_replaced_id)
            all_legal_actions[model_id][block_2b_replaced_id] = blocks_to_replace

    _print_all_action_space(all_legal_actions)

    # Save all_legal_actions with pickle
    with open(file_name, "wb") as f:
        pickle.dump(all_legal_actions, f)
    return all_legal_actions
