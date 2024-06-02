import timm
import torch
from opacus.validators import ModuleValidator
import numpy as np

import json
import pdb


def load_models_info(task_type) -> list[dict]:
    """
    Load model information from model_info.json.
    If model_ids is not specified, load all models. Otherwise, load the specified models.
    """
    if task_type == "text":
        model_info_path = "models/model_info_text.json"
    elif task_type == "vision_vit":
        model_info_path = "models/model_info_vision_vit.json"
    elif task_type == "vision_resnet":
        model_info_path = "models/model_info_vision_resnet.json"
    elif task_type == "recommendation":
        model_info_path = "models/model_info_recommendation.json"
    else:
        raise ValueError(f"Invalid task type: {task_type}")

    with open(model_info_path, "r") as f:
        models_info = json.load(f)
    models_info = list(models_info.values())
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
    n_models: int = 4,
) -> float:
    return (
        remaining_blocks * block_size
        + untouched_weights * n_models
        + n_original_weights
    ) / (n_original_weights * (n_models + 1))


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


def separate_blocks(model_constitution, n_base_blocks):
    new_blocks = []
    blocks_from_base = set()
    for block in model_constitution:
        if block < n_base_blocks:
            blocks_from_base.add(block)
        else:
            new_blocks.append(block)
    print(f"New blocks: {new_blocks}")
    print(f"Blocks from base: {blocks_from_base}")
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

        model = timm.create_model(
            model_args.model, pretrained=True, num_classes=num_classes
        )
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
