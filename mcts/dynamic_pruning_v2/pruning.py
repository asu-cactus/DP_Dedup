import pdb
import pickle
from dataclasses import dataclass
from bisect import bisect
import os
from time import time

import numpy as np


# @dataclass
# class ActionInfo:
#     block_to_replace: int
#     acc: float
#     block_2b_replaced_norm: float
#     distance: float


@dataclass
class ActionInfo:
    block_to_replace: int
    model_id: int


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
    if model_args.task_type == "text":
        from text_task_utils.evaluate import evaluate
    elif model_args.task_type.startswith("vision"):
        from vision_task_utils.evaluate import evaluate
    else:
        raise ValueError(f"Unknown task_type: {model_args.task_type}")

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
        model_info = models_info[model_id]
        data_args.task_name = model_info["task_name"]
        model_args.model_name_or_path = model_info["model_path"]
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


def _block_id_to_model_id(model_range: list[int], block_id: int) -> int:
    return bisect(model_range[1:], block_id)


def _print_all_action_space(all_legal_actions):
    for block_2b_replaced, action_infos in all_legal_actions.items():
        print(f"{block_2b_replaced}: {action_infos[:10]}")
    print(f"Original action space width: {len(all_legal_actions)}")


def get_heuristic_info(
    model_args,
    data_args,
    training_args,
    models_info,
    models_storage,
    order_by_magnitude: bool = True,
) -> dict[int, dict[int, tuple[int, float]]]:
    """Get the heuristic information for the MCTS."""

    if model_args.task_type == "text":
        from utils.text_model_sensitivity import get_block_sensitivity
    elif model_args.task_type.startswith("vision"):
        from utils.vision_model_sensitivity import get_block_sensitivity
    else:
        raise ValueError(f"Unknown task_type: {model_args.task_type}")
    # Get file name
    if training_args.orderby == "l2_norm":
        file_name = f"all_legal_actions_l2_{model_args.task_type}.pkl"
    elif training_args.orderby == "3rd_quantile":
        file_name = f"all_legal_actions_3rd_{model_args.task_type}.pkl"
    else:
        raise ValueError(f"Unknown training_args.orderby: {training_args.orderby}")

    # Load all_legal_actions from pickle if it exists
    if os.path.exists(file_name):
        with open(file_name, "rb") as f:
            all_legal_actions = pickle.load(f)
        print("Loaded all_legal_actions from pickle")
        _print_all_action_space(all_legal_actions)
        return all_legal_actions

    heuristics_dict = get_heuristics_dict(
        model_args, data_args, training_args, models_info, models_storage
    )
    last_legal_model = find_last_legal_model(models_info)

    acc_thresholds = [
        info["original_acc"] - info["acc_drop_threshold"] for info in models_info
    ]
    models_range = models_storage["model_range"]
    blocks = models_storage["blocks"]
    all_legal_actions = {}

    if order_by_magnitude:
        # # Note that this is different from the baseline method, which order in each model, starting from the smallest privacy budget model.
        # if training_args.orderby == "l2_norm":
        #     block_magnitude = np.linalg.norm(blocks, axis=1)
        # else:
        #     block_magnitude = np.quantile(blocks, 0.75, axis=1)
        # interate_seq = np.argsort(block_magnitude)
        # heuristics_dict = {key: heuristics_dict[key] for key in interate_seq}

        # The following code order blocks in each model and put them together
        temp_dict = {}
        for i, model_start in enumerate(models_range[:-1]):

            # model_end = models_range[i + 1]
            # model_blocks = blocks[model_start:model_end]

            # The following lines order the blocks by sensitivity measure
            model_info = models_info[i]

            model_blocks, _ = get_block_sensitivity(
                model_info["task_name"],
                training_args.sensitivity_measure,
                model_info["budget"],
                skip_embeds=False,
                return_n_embed_blocks=False,
            )

            if training_args.orderby == "l2_norm":
                block_magnitude = np.linalg.norm(model_blocks, axis=1)
            elif training_args.orderby == "3rd_quantile":
                block_magnitude = np.quantile(model_blocks, 0.75, axis=1)
            else:
                raise ValueError(f"Unknown orderby: {training_args.orderby}")
            interate_seq = np.argsort(block_magnitude)
            interate_seq = [j + model_start for j in interate_seq]
            temp_dict.update({key: heuristics_dict[key] for key in interate_seq})
        heuristics_dict = temp_dict

    for block_2b_replaced_id, value in heuristics_dict.items():
        block_2b_replaced = blocks[block_2b_replaced_id]
        curr_model_id = _block_id_to_model_id(models_range, block_2b_replaced_id)

        delete_indices = []
        acc_threshold = acc_thresholds[curr_model_id]
        pass_test = False
        for block_to_replace, acc in value.items():
            if acc < acc_threshold:
                delete_indices.append(block_to_replace)
            else:
                pass_test = True
                break
        if not pass_test:
            continue

        # Sort the legal actions by l1 distance, from low to high
        candidate_end = models_range[last_legal_model[curr_model_id] + 1]
        candidate_blocks = blocks[:candidate_end]

        diff = np.sum(
            np.abs(candidate_blocks - block_2b_replaced),
            axis=1,
            keepdims=False,
        )
        ind = np.argsort(diff).tolist()
        ind.remove(block_2b_replaced_id)
        for i in delete_indices:
            ind.remove(i)

        action_infos = [
            ActionInfo(i, _block_id_to_model_id(models_range, i)) for i in ind
        ]
        all_legal_actions[block_2b_replaced_id] = action_infos

    # Save all_legal_actions with pickle
    with open(file_name, "wb") as f:
        pickle.dump(all_legal_actions, f)

    _print_all_action_space(all_legal_actions)
    return all_legal_actions
