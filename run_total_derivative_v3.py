"""
In version 1, candidate blocks do not include the blocks with all zero gradients.
In version 2, all the exisiting blocks are the candidate blocks.
In this version 3, the second sub step is determined by l1 or l2 distance, and only 
order the sequence of blocks to be deduplicated.
"""

import os
from dataclasses import dataclass
import pdb

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from sensitivity_measure import get_model_and_dateset, gradient_sensitity
from utils import load_models_info
from utils.parse_args import parse_args
from utils.blocker import block_model_1d
from text_task_utils.evaluate import evaluate

import numpy as np


@dataclass
class ModelInfo:
    blocks: np.ndarray
    n_original_blocks: int
    block_original_pos: list
    model_constitution: list


def deduplicate_blocks(
    model_args,
    data_args,
    training_args,
    models_info,
    base_model_info=None,
    distance_metric="l1",
):

    # Set parameters
    model_id = 0 if base_model_info is None else 1
    model_info = models_info[model_id]
    acc_threshold = model_info["original_acc"] - model_info["acc_drop_threshold"]
    data_args.task_name = model_info["task_name"]
    model_args.model_name_or_path = model_info["model_path"]

    # Get the model and dataset
    model, dataset = get_model_and_dateset(
        data_args,
        model_args,
        training_args,
    )

    # Get weight blocks and gradient blocks
    original_wblocks = block_model_1d(model)["blocks"]
    grad_blocks = gradient_sensitity(model, dataset, correct_only=True)

    # Compute the indices of the gradient blocks that are (not) all zeros
    all_zeros = np.all(grad_blocks == 0, axis=1)
    non_all_zeros = ~all_zeros
    allzero_indices = np.where(all_zeros != 0)[0]
    nonzero_indices = np.nonzero(non_all_zeros)[0]
    nonzero_indices_set = set(nonzero_indices)

    # Get the blocks to search
    weight_blocks = original_wblocks[non_all_zeros]
    grad_blocks = grad_blocks[non_all_zeros]
    assert len(nonzero_indices) == len(weight_blocks)

    if model_id == 0:
        candidate_blocks = original_wblocks
    else:
        candidate_blocks = np.concatenate(
            [original_wblocks, base_model_info.blocks], axis=0
        )

    # For each block, get the minimum the distance and the index to all the candidate blocks, and compute D
    if distance_metric == "l1":
        ord = 1
    elif distance_metric == "l2":
        ord = 2
    else:
        raise ValueError(f"Invalid distance metric: {distance_metric}")

    D = np.empty(len(grad_blocks))
    to_replace_indices = []
    for i, (wblock, gblock) in enumerate(zip(weight_blocks, grad_blocks)):
        diff = candidate_blocks - wblock
        # Find the minimum distance, excluding the current block itself
        distances = np.linalg.norm(diff, ord=ord, axis=1)
        min_distance_ind_top2 = np.argsort(distances)[:2]
        min_distance_index = (
            min_distance_ind_top2[1]
            if nonzero_indices[i] == min_distance_ind_top2[0]
            else min_distance_ind_top2[0]
        )
        # Compute D (total derivative)
        D[i] = np.dot(diff[min_distance_index], gblock)
        to_replace_indices.append(min_distance_index)

    # Sort the blocks by the distance
    sorted_indices = np.argsort(D)
    D = D[sorted_indices]
    dedup_map = {}
    for index in sorted_indices:
        dedup_map[nonzero_indices[index]] = to_replace_indices[index]

    # Start deduplication
    dedup_indices = set()
    used_allzerograd_indices = set()
    blocks = (
        original_wblocks
        if model_id == 0
        else np.concatenate([original_wblocks, base_model_info.blocks], axis=0)
    )
    model_constitution = list(range(len(original_wblocks)))
    n_eval = 0
    for i, (be_replaced, to_replace) in enumerate(dedup_map.items()):
        if be_replaced in dedup_indices:
            continue
        n_eval += 1

        # temp_constitution = model_constitution.copy()
        # temp_constitution[be_replaced] = to_replace
        temp_constitution = [
            to_replace if b == be_replaced else b for b in model_constitution
        ]
        acc = evaluate(
            None, None, temp_constitution, data_args, model_args, training_args, blocks
        )
        if acc >= acc_threshold:
            model_constitution = temp_constitution
            dedup_indices.add(be_replaced)
            if (
                to_replace < len(original_wblocks)
                and to_replace not in nonzero_indices_set
            ):
                # These are blocks that the gradients are all zeros.
                used_allzerograd_indices.add(to_replace)

        # Print information
        printed_constitution = get_printed_constitution(
            model_constitution, base_model_info
        )
        if model_id == 1:
            be_replaced += base_model_info.n_original_blocks
            if to_replace >= len(original_wblocks):
                to_replace = base_model_info.block_original_pos[
                    to_replace - len(original_wblocks)
                ]
            else:
                to_replace += base_model_info.n_original_blocks
        print(
            f"Model {model_id} block {be_replaced} -> {to_replace} acc: {acc:.4f}, D value: {D[i]:.4f}"
        )
        print(f"Model constitution after dedup: {printed_constitution}")

    # Deduplicate blocks with all zero gradients
    print(f"Use n gradient zero blocks: {len(used_allzerograd_indices)}")
    # Get remaining blocks that gradients are not all zeros
    remaining_block_indices = sorted(list(nonzero_indices_set - dedup_indices))
    remaining_blocks = original_wblocks[remaining_block_indices]
    # Get all blocks that gradients are all zeros
    allzero_indices = sorted(list(set(allzero_indices) - used_allzerograd_indices))
    allzerograd_blocks = original_wblocks[allzero_indices]
    # For each block in allzerograd_blocks, deduplicate it by the most similar block in remaining_blocks
    temp_constitution = model_constitution.copy()

    for i, allzerograd_block in enumerate(allzerograd_blocks):
        diff = remaining_blocks - allzerograd_block
        distances = np.linalg.norm(diff, ord=ord, axis=1)
        min_distance_index = np.argmin(distances)
        to_replace = remaining_block_indices[min_distance_index]
        be_replaced = allzero_indices[i]
        try:
            temp_constitution[be_replaced] = to_replace
        except:
            pdb.set_trace()

    # Evaluate the model with the deduplicated blocks
    acc = evaluate(
        None, None, temp_constitution, data_args, model_args, training_args, blocks
    )
    if acc >= acc_threshold:
        model_constitution = temp_constitution
    printed_constitution = get_printed_constitution(model_constitution, base_model_info)
    n_remaining_blocks = get_n_remaining_blocks(model_constitution)
    print(f"Model {model_id} dedup all zero grad blocks, acc: {acc:.4f}")
    print(f"Model constitution after dedup: {printed_constitution}")
    print(f"Number of remaining blocks: {len(n_remaining_blocks)}")

    # Return base model info
    if model_id == 0:
        block_original_pos = sorted(list(set(model_constitution)))
        blocks = original_wblocks[block_original_pos]
        return ModelInfo(
            blocks=blocks,
            n_original_blocks=len(original_wblocks),
            block_original_pos=block_original_pos,
            model_constitution=model_constitution,
        )


def get_printed_constitution(model_constitution, base_model_info):
    if base_model_info is None:
        return model_constitution
    printed_constitution = []
    for c in model_constitution:
        if c < len(model_constitution):
            printed_constitution.append(c + base_model_info.n_original_blocks)
        else:
            c = base_model_info.block_original_pos[c - len(model_constitution)]
            printed_constitution.append(c)
    return printed_constitution


def get_n_remaining_blocks(model_constitution):
    return set([c for c in model_constitution if c < len(model_constitution)])


def run():
    model_args, data_args, training_args = parse_args()
    models_info = load_models_info()

    base_model_info = deduplicate_blocks(
        model_args, data_args, training_args, models_info
    )

    deduplicate_blocks(
        model_args, data_args, training_args, models_info, base_model_info
    )


if __name__ == "__main__":
    run()
