import os
from collections import defaultdict
from dataclasses import dataclass
import pdb

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
    nonzero_indices = np.nonzero(non_all_zeros)[0]

    # Get the blocks to search
    weight_blocks = original_wblocks[non_all_zeros]
    grad_blocks = grad_blocks[non_all_zeros]
    assert len(nonzero_indices) == len(weight_blocks)

    if model_id == 0:
        candidate_blocks = weight_blocks
    else:
        candidate_blocks = np.concatenate(
            [weight_blocks, base_model_info.blocks], axis=0
        )

    n_curr_blocks = len(grad_blocks)
    n_cand_blocks = len(candidate_blocks)

    # Compute the change matrix
    A = np.empty((n_curr_blocks, n_cand_blocks))
    for i, (wblock, gblock) in enumerate(zip(weight_blocks, grad_blocks)):
        A[i] = np.dot(candidate_blocks - wblock, gblock)

    # Sort A
    sorted_indices = np.argsort(A, axis=None)
    row_indices, col_indices = np.unravel_index(sorted_indices, A.shape)

    # sorted_A = np.sort(A, axis=None)
    # print(A)
    # print(sorted_A)
    # pdb.set_trace()

    # Start deduplication
    dedup_indices = set()
    blocks = (
        original_wblocks
        if model_id == 0
        else np.concatenate([original_wblocks, base_model_info.blocks], axis=0)
    )
    model_constitution = list(range(n_curr_blocks))
    for row, col in zip(row_indices, col_indices):
        if row == col or col in dedup_indices or row in dedup_indices:
            continue

        row_idx = nonzero_indices[row]
        if col < n_curr_blocks:
            col_idx = nonzero_indices[col]
        else:
            col_idx = col + len(original_wblocks) - n_curr_blocks
        temp_constitution = model_constitution.copy()
        temp_constitution[row_idx] = col_idx
        acc = evaluate(
            None, None, temp_constitution, data_args, model_args, training_args, blocks
        )
        if acc >= acc_threshold:
            model_constitution = temp_constitution
            dedup_indices.add(row)
            dedup_indices.add(col)

        # Print information
        printed_constitution = model_constitution
        if model_id == 1:
            printed_constitution = [
                i + base_model_info.n_original_blocks for i in model_constitution
            ]
            row_idx += base_model_info.n_original_blocks
            col_idx = base_model_info.block_original_pos[col - n_curr_blocks]

        print(
            f"Model {model_id} block {row_idx} -> {col_idx} acc: {acc:.4f}, A value: {A[row, col]}"
        )
        print(f"Model constitution after dedup: {printed_constitution}")

    # Return model info
    block_original_pos = sorted(list(set(model_constitution)))
    blocks = original_wblocks[block_original_pos]
    return ModelInfo(
        blocks=blocks,
        n_original_blocks=len(original_wblocks),
        block_original_pos=block_original_pos,
        model_constitution=model_constitution,
    )


def run():
    model_args, data_args, training_args = parse_args()
    models_info = load_models_info()

    base_model_info = deduplicate_blocks(
        model_args, data_args, training_args, models_info
    )
    print(f"Remaining blocks: {len(base_model_info.blocks)}")

    model_info = deduplicate_blocks(
        model_args, data_args, training_args, models_info, base_model_info
    )
    print(f"Remaining blocks: {len(model_info.blocks)}")


if __name__ == "__main__":
    run()
