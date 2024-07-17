import timm
import torch
from opacus.validators import ModuleValidator
import numpy as np

import json
import pdb


def load_models_info(model_args) -> list[dict]:
    """
    Load model information from model_info.json.
    If model_ids is not specified, load all models. Otherwise, load the specified models.
    """
    if model_args.task_type == "text":
        model_info_path = "models/text.json"
    elif model_args.task_type == "vision_vit":
        if model_args.heter:
            model_info_path = "models/vision_vit_heter.json"
        elif model_args.big_batch:
            model_info_path = "models/vision_vit_20models.json"
        elif model_args.dummy_base_model >= 0:
            model_info_path = "models/vision_vit_dummy.json"
        elif model_args.inter_data_mode == "cifar100_celeba":
            model_info_path = "models/vision_vit_cifar100_celeba.json"
        elif model_args.inter_data_mode == "celeba_cifar100":
            model_info_path = "models/vision_vit_celeba_cifar100.json"
        else:
            model_info_path = "models/vision_vit.json"

    elif model_args.task_type == "vision_resnet":
        if model_args.prune:
            model_info_path = "models/vision_resnet_pruned.json"
        elif model_args.quantize:
            model_info_path = "models/vision_resnet_quantized.json"
        elif model_args.heter:
            model_info_path = "models/vision_resnet_heter.json"
        else:
            model_info_path = "models/vision_resnet.json"
    elif model_args.task_type == "recommendation":
        model_info_path = "models/recommendation.json"
    else:
        raise ValueError(f"Invalid task type: {model_args.task_type}")

    with open(model_info_path, "r") as f:
        models_info = json.load(f)
    models_info = list(models_info.values())
    if model_args.dummy_base_model >= 0:
        model_ids = [model_args.dummy_base_model, 4, 5, 6, 7, 8]
        models_info = [models_info[i] for i in model_ids]
    for info in models_info:
        print(info)
    return models_info


def set_model_args(model_args, model, model_storage):
    untouched_weight_count = 0
    for weight in model_storage["untouch_weights"].values():
        untouched_weight_count += weight.size
    model_args.untouched_weights = untouched_weight_count
    print(f"Number of untouched weights: {untouched_weight_count}")

    params_count = 0
    for params in model.parameters():
        params_count += params.numel()
    model_args.n_original_weights = params_count
    print(f"Number of original weights: {params_count}")


def compute_compression_ratio(
    remaining_blocks: int,
    block_size: int,
    untouched_weights: int,
    n_original_weights: int,
    n_models: int,
) -> float:
    n_before = remaining_blocks * block_size + untouched_weights * n_models
    n_after = n_original_weights * n_models
    return n_before / n_after
    # if n_models == 1:
    #     return (remaining_blocks * block_size + untouched_weights) / n_original_weights

    # return (
    #     remaining_blocks * block_size
    #     + untouched_weights * n_models
    #     + n_original_weights
    # ) / (n_original_weights * (n_models + 1))


def merge_model_storage(base_model_storage, curr_model_storage):
    base_blocks = base_model_storage["blocks"]
    curr_blocks = curr_model_storage["blocks"]
    blocks = np.concatenate([base_blocks, curr_blocks], axis=0)
    model_range = [0, base_blocks.shape[0], base_blocks.shape[0] + curr_blocks.shape[0]]
    return {
        "blocks": blocks,
        "model_range": model_range,
        "untouch_weights": [
            base_model_storage["untouch_weights"],
            curr_model_storage["untouch_weights"],
        ],
    }


def collect_storage(model_storage, curr_model_storage, model_constitution):
    """
    Collects the extra storage from the current model storage and adds it to the extra storage.
    This function assumes that the current model has the same architecture as the base model.
    model_storage is a dictionary with the following keys:
    - blocks: np.array, shape: (n_blocks, block_size)
    - untouch_weights: list of untouch weights
    - model_constitution: list of model constitution
    """
    n_base_blocks = len(model_constitution)
    new_blocks = [block for block in model_constitution if block >= n_base_blocks]
    new_blocks = list(set(new_blocks))

    # Modify and collect model_constitution
    start_index = model_storage["blocks"].shape[0]
    new_indices = {block: start_index + i for i, block in enumerate(new_blocks)}
    model_constitution = [new_indices.get(block, block) for block in model_constitution]
    model_constitution = np.array(model_constitution, dtype=int)
    model_storage["model_constitution"].append(model_constitution)

    # Collect blocks
    blocks = model_storage["blocks"]
    new_block_indices = [block - n_base_blocks for block in new_blocks]
    new_block_storage = curr_model_storage["blocks"][new_block_indices, :]
    model_storage["blocks"] = np.concatenate([blocks, new_block_storage], axis=0)

    # Collect untouch_weights
    model_storage["untouch_weights"].append(curr_model_storage["untouch_weights"])

    return model_storage


def separate_blocks(model_constitution, n_base_blocks, return_new_blocks=False):
    new_blocks = set()
    blocks_from_base = set()
    for block in model_constitution:
        if block < n_base_blocks:
            blocks_from_base.add(block)
        else:
            new_blocks.add(block)
    print(f"New blocks: {new_blocks}")
    print(f"Blocks from base: {blocks_from_base}")

    if return_new_blocks:
        return len(new_blocks), blocks_from_base, new_blocks
    return len(new_blocks), blocks_from_base


def print_params(model):
    total_params = 0
    for name, param in model.named_parameters():
        print(name, param.size())
        total_params += param.numel()

    print(f"Number of total parameters: {total_params}")

    pdb.set_trace()


def load_model(model_info, model_args):
    if model_args.task_type == "text":
        from text_task_utils.models import RobertaForPromptFinetuning
        from text_task_utils.evaluate import evaluate as eval_fn
        from text_task_utils.train import train as train_fn
        from utils.text_model_sensitivity import get_block_sensitivity as sensitivity_fn

        model = RobertaForPromptFinetuning.from_pretrained(model_info["model_path"])

    elif "vision" in model_args.task_type:
        if model_info["task_name"] == "CIFAR100":
            num_classes = 100
        elif model_info["task_name"] == "CelebA":
            num_classes = 40
        from vision_task_utils.evaluate import evaluate as eval_fn
        from vision_task_utils.train import train as train_fn
        from utils.vision_model_sensitivity import (
            get_block_sensitivity as sensitivity_fn,
        )

        # model = timm.create_model(model_args.model, num_classes=num_classes, pretrained=True)
        model_name = (
            "resnet152.tv2_in1k"
            if "in1k" in model_info["model_path"]
            else "vit_large_patch16_224"
        )
        model = timm.create_model(model_name, num_classes=num_classes)
        model = ModuleValidator.fix(model)
        model.load_state_dict(torch.load(model_info["model_path"], map_location="cpu"))

    elif model_args.task_type == "recommendation":
        from recommendation_task_utils.evaluate import load_model
        from recommendation_task_utils.evaluate import evaluate as eval_fn
        from recommendation_task_utils.train import train as train_fn
        from utils.recommender_sensitivity import (
            get_block_sensitivity as sensitivity_fn,
        )

        model = load_model(model_info["model_path"])
    else:
        raise ValueError(f"Invalid task name: {model_args.task_type}")

    return model, eval_fn, train_fn, sensitivity_fn


def save_model_storage(model_storage, save_path):
    np.savez(
        save_path,
        blocks=model_storage["blocks"],
        untouched_weights=model_storage["untouch_weights"],
    )


def longest_increasing_subsequence(arr):
    if not arr:
        return []

    n = len(arr)
    lis = [1] * n  # Initialize LIS values for all indexes as 1
    prev_index = [-1] * n  # To track the previous index in the LIS

    # Compute LIS values in a bottom-up manner
    for i in range(1, n):
        for j in range(0, i):
            if arr[i] > arr[j] and lis[i] < lis[j] + 1:
                lis[i] = lis[j] + 1
                prev_index[i] = j

    # Find the maximum value in lis[] and its index
    max_len = max(lis)
    max_index = lis.index(max_len)

    # Reconstruct the longest increasing subsequence
    lis_index = []
    # lis_sequence = []
    while max_index != -1:
        lis_index.append(max_index)
        # lis_sequence.append(arr[max_index])
        max_index = prev_index[max_index]

    # Reverse the lis_sequence since we built it backwards
    lis_index.reverse()
    # lis_sequence = [arr[i] for i in lis_index]

    # return lis_index, lis_sequence
    return lis_index


def set_val_epsilon(training_args, curr_budget, base_budget):
    if training_args.extra_val_eps >= 0:
        eps_ratio = (2 * training_args.max_fails) ** (2 / 3)
        val_eps = curr_budget + base_budget + training_args.extra_val_eps
        training_args.val_epsilon1 = val_eps / (1 + eps_ratio)
        training_args.val_epsilon2 = training_args.val_epsilon1 * eps_ratio
