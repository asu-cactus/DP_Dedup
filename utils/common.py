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
    if model_args.task_type == "text_qnli":
        if (
            model_args.dummy_base_model >= 0
            or model_args.big_batch
            or model_args.in_group_n_base
            or model_args.base_model_selection
        ):
            model_info_path = "models/text_10models.json"
        elif model_args.inter_data_mode == "qnli_sst2":
            model_info_path = "models/text_qnli_sst2.json"
        elif model_args.inter_data_mode == "sst2_qnli":
            model_info_path = "models/text_sst2_qnli.json"
        else:
            model_info_path = "models/text_qnli.json"
    elif model_args.task_type == "text_mnli":
        model_info_path = "models/text_mnli.json"
    elif model_args.task_type == "text_mnli_sst2":
        model_info_path = "models/text_mnli_sst2.json"
    elif model_args.task_type == "text_sst2_mnli":
        model_info_path = "models/text_sst2_mnli.json"
    elif model_args.task_type == "text_qnli_mnli":
        model_info_path = "models/text_qnli_mnli.json"
    elif model_args.task_type == "cifar100_qnli":
        model_info_path = "models/cifar100_qnli.json"
    elif model_args.task_type == "cifar100_sst2":
        model_info_path = "models/cifar100_sst2.json"

    elif model_args.task_type == "vision_vit":
        if model_args.heter:
            model_info_path = "models/vision_vit_heter.json"
        elif model_args.big_batch or model_args.base_model_selection:
            model_info_path = "models/vision_vit_10models.json"
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
        elif model_args.big_batch or model_args.base_model_selection:
            model_info_path = "models/vision_resnet_20models.json"
        elif model_args.dummy_base_model >= 0:
            model_info_path = "models/vision_resnet_dummy.json"
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
    if model_args.in_group_n_base > 0:
        assert model_args.n_base_models <= 4
        n_models = model_args.n_base_models + 4
        models_info = models_info[:8]
        models_info = models_info[-n_models:]
    for info in models_info:
        print(info)
    return models_info


def set_model_args(model_args, model, model_storage):
    untouched_weight_count = 0
    for weight in model_storage["untouched_weights"].values():
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
    total_original_weights: int,
) -> float:
    n_after = remaining_blocks * block_size + untouched_weights
    return n_after / total_original_weights


def merge_base_model_storage(base_storage, new_base_storage):
    if base_storage["blocks"] is None:
        return {
            "blocks": new_base_storage["blocks"],
            "untouched_weights": [new_base_storage["untouched_weights"]],
        }
    base_blocks = base_storage["blocks"]
    new_blocks = new_base_storage["blocks"]
    blocks = np.concatenate([base_blocks, new_blocks], axis=0)
    return {
        "blocks": blocks,
        "untouched_weights": base_storage["untouched_weights"]
        + [new_base_storage["untouched_weights"]],
    }


def merge_model_storage(base_model_storage, curr_model_storage):
    base_blocks = base_model_storage["blocks"]
    curr_blocks = curr_model_storage["blocks"]
    blocks = np.concatenate([base_blocks, curr_blocks], axis=0)
    model_range = [0, base_blocks.shape[0], base_blocks.shape[0] + curr_blocks.shape[0]]
    return {
        "blocks": blocks,
        "model_range": model_range,
        "untouched_weights": [
            base_model_storage["untouched_weights"],
            curr_model_storage["untouched_weights"],
        ],
    }


def collect_storage(model_storage, curr_model_storage, model_constitution):
    """
    Collects the extra storage from the current model storage and adds it to the extra storage.
    This function assumes that the current model has the same architecture as the base model.
    model_storage is a dictionary with the following keys:
    - blocks: np.array, shape: (n_blocks, block_size)
    - untouched_weights: list of untouch weights
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

    # Collect untouched_weights
    model_storage["untouched_weights"].append(curr_model_storage["untouched_weights"])

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


def load_model(model_info):
    if model_info["task_name"] in ("qnli", "mnli", "sst-2"):
        from text_task_utils.models import RobertaForPromptFinetuning
        from text_task_utils.evaluate import evaluate as eval_fn
        from utils.text_model_sensitivity import get_block_sensitivity as sensitivity_fn

        model = RobertaForPromptFinetuning.from_pretrained(model_info["model_path"])

    elif model_info["task_name"] in ("CIFAR100", "CelebA"):
        if model_info["task_name"] == "CIFAR100":
            num_classes = 100
        elif model_info["task_name"] == "CelebA":
            num_classes = 40
        from vision_task_utils.evaluate import evaluate as eval_fn
        from utils.vision_model_sensitivity import (
            get_block_sensitivity as sensitivity_fn,
        )

        model_name = (
            "resnet152.tv2_in1k"
            if "in1k" in model_info["model_path"]
            else "vit_large_patch16_224"
        )
        model = timm.create_model(model_name, num_classes=num_classes)
        model = ModuleValidator.fix(model)
        model.load_state_dict(torch.load(model_info["model_path"], map_location="cpu"))
    else:
        raise ValueError(f"Invalid task name: {model_info['task_name']}")
    return model, eval_fn, sensitivity_fn


def save_model_storage(model_storage, save_path):
    np.savez(
        save_path,
        blocks=model_storage["blocks"],
        untouched_weights=model_storage["untouched_weights"],
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


def set_val_epsilon(training_args, curr_budget, base_budget, is_same_data):
    if training_args.extra_val_eps >= 0:
        eps_ratio = (2 * training_args.max_fails) ** (2 / 3)
        if is_same_data:
            val_eps = curr_budget + base_budget
        else:
            val_eps = max(curr_budget, base_budget)
        val_eps += training_args.extra_val_eps
        training_args.val_epsilon1 = val_eps / (1 + eps_ratio)
        training_args.val_epsilon2 = training_args.val_epsilon1 * eps_ratio
