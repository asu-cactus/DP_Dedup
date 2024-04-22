import timm
import torch
import numpy as np
import pdb


def merge_model_storage(base_model_storage, curr_model_storage):
    base_blocks = base_model_storage["blocks"]
    curr_blocks = curr_model_storage["blocks"]
    blocks = np.concatenate([base_blocks, curr_blocks], axis=0)
    model_range = [0, base_blocks.shape[0], base_blocks.shape[0] + curr_blocks.shape[0]]
    return {
        "blocks": blocks,
        "model_range": model_range,
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
        from utils.text_model_sensitivity import get_block_sensitivity as sensitivity_fn

    elif model_args.task_type == "vision":
        if model_info["task_name"] == "CIFAR100":
            num_classes = 100
        elif model_info["task_name"] == "CelebA":
            num_classes = 40
        from vision_task_utils.evaluate import evaluate as eval_fn
        from utils.vision_model_sensitivity import (
            get_block_sensitivity as sensitivity_fn,
        )
    else:
        raise ValueError(f"Invalid task name: {model_args.task_type}")

    if model_args.task_type == "text":
        model = RobertaForPromptFinetuning.from_pretrained(model_info["model_path"])
    elif model_args.task_type == "vision":
        model = timm.create_model(
            model_args.model, pretrained=True, num_classes=num_classes
        )
        model.load_state_dict(torch.load(model_info["model_path"]))

    return model, eval_fn, sensitivity_fn
