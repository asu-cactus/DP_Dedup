import timm
import torch


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
