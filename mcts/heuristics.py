import pdb
import pickle
from collections import defaultdict
from dataclasses import dataclass
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
    top_k = training_args.top_k

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
            ind = np.argpartition(diff, top_k + 1)[: top_k + 1]
            ind = ind[np.argsort(diff[ind])]
            ind = ind[1:] if model_id == target_model_id else ind[:top_k]
            ind = [i + target_model_range_start for i in ind]

            for j in ind:
                temp_constitution = model_constitution.copy()
                temp_constitution[i - model_range_start] = j

                acc = evaluate(
                    models_storage,
                    model_id,
                    temp_constitution,
                    data_args,
                    model_args,
                    training_args,
                )
                action_to_acc_dict[j] = acc

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
):
    """Get the heuristic information for the MCTS."""
    heuristics_dict = get_heuristics_dict(
        model_args, data_args, training_args, models_info, models_storage
    )
    acc_thresholds = [
        info["original_acc"] - info["acc_drop_threshold"] for info in models_info
    ]
    H1a_to_C1a = {}  # type: dict[int, int]
    H2aa_to_C2aa = defaultdict(dict)  # type: dict[int, dict[int, int]]

    legal_actions_2 = {}

    dedup_counts = 0
    for block_2b_replaced, value in heuristics_dict.items():
        n_can_be_placed = 0
        for i, (block_to_replace, acc) in enumerate(value.items()):
            model_id = i // training_args.top_k
            if acc >= acc_thresholds[model_id]:
                n_can_be_placed += 1

        if n_can_be_placed > 0:
            # Custom heuristic function for H1a_to_C1a
            H1a_to_C1a[block_2b_replaced] = int(50 * n_can_be_placed / len(value))
            # Custom heuristic function for H2aa_to_C2aa
            all_pass = True
            for i, (block_to_replace, acc) in enumerate(value.items()):
                model_id = i // training_args.top_k
                if acc >= acc_thresholds[model_id]:
                    H2aa_to_C2aa[block_2b_replaced][block_to_replace] = int(
                        1000
                        * len(value)
                        / n_can_be_placed
                        * (acc - acc_thresholds[model_id])
                    )
                    if block_2b_replaced not in legal_actions_2:
                        legal_actions_2[block_2b_replaced] = defaultdict(list)
                    legal_actions_2[block_2b_replaced][model_id].append(
                        block_to_replace
                    )
                else:
                    all_pass = False
            if all_pass:
                dedup_counts += 1

    all_legal_actions = legal_actions_2
    # legal_actions_reverse = defaultdict(list)
    # for block_2b_replaced, value in all_legal_actions.items():
    #     for model_id, block_to_replace in value.items():
    #         legal_actions_reverse[block_to_replace].append(
    #             ReverseDictValue(block_2b_replaced, model_id)
    #         )

    # Make some conversions
    H2aa_to_C2aa = dict(H2aa_to_C2aa)
    # legal_actions_1 = list(H1a_to_C1a.keys())
    all_legal_actions = {k: dict(v) for k, v in all_legal_actions.items()}
    # legal_actions_reverse = dict(legal_actions_reverse)
    heuristic_constant = 1 - dedup_counts / models_storage["model_range"][-1]

    print(f"H1a_to_C1a:\n{H1a_to_C1a}")
    print(f"H2aa_to_C2aa:\n{H2aa_to_C2aa}")
    print(f"dedup_counts: {dedup_counts}")
    print(f"H: {heuristic_constant}")
    # print(f"legal_actions_1:\n{legal_actions_1}")
    print(f"all legal 1st sub actions: {list(all_legal_actions.keys())}")
    print(f"all_legal_actions:\n{all_legal_actions}")
    # print(f"legal_actions_reverse:\n{legal_actions_reverse}")
    return (
        heuristic_constant,
        H1a_to_C1a,
        H2aa_to_C2aa,
        # legal_actions_1,
        all_legal_actions,
        # legal_actions_reverse,
    )


@dataclass
class ReverseDictValue:
    block_2b_replaced: int
    model_id: int
