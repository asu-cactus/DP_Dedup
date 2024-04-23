import timm
import torch
from opacus.validators import ModuleValidator
import numpy as np
import torchvision


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

    elif "vision" in model_args.task_type:
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
    elif "vision" in model_args.task_type:
        model = timm.create_model(
            model_args.model, pretrained=True, num_classes=num_classes
        )
        model = ModuleValidator.fix(model)
        model.load_state_dict(torch.load(model_info["model_path"], map_location="cpu"))
    else:
        raise ValueError(f"Invalid task type: {model_args.task_type}")

    return model, eval_fn, sensitivity_fn


def load_vision_dateset(data_args):
    transformation = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(
                (224, 224)
            ),  # https://discuss.pytorch.org/t/runtimeerror-stack-expects-each-tensor-to-be-equal-size-but-got-3-224-224-at-entry-0-and-3-224-336-at-entry-3/87211/10
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    # if data_args.dataset_name in ["SVHN", "CIFAR10"]:
    #     num_classes = 10
    # elif data_args.dataset_name in ["CIFAR100", "FGVCAircraft"]:
    #     num_classes = 100
    # elif data_args.dataset_name in ["Food101"]:
    #     num_classes = 101
    # elif data_args.dataset_name in ["GTSRB"]:
    #     num_classes = 43
    # elif data_args.dataset_name in ["CelebA"]:
    #     num_classes = 40
    # elif data_args.dataset_name in ["Places365"]:
    #     num_classes = 365
    # elif data_args.dataset_name in ["ImageNet"]:
    #     num_classes = 1000
    # elif data_args.dataset_name in ["INaturalist"]:
    #     num_classes = 10000

    if data_args.dataset_name in ["SVHN", "Food101", "GTSRB", "FGVCAircraft"]:
        # trainset = getattr(torchvision.datasets, data_args.dataset_name)(
        #     root="data/", split="train", download=True, transform=transformation
        # )
        testset = getattr(torchvision.datasets, data_args.dataset_name)(
            root="data/", split="test", download=True, transform=transformation
        )
    elif data_args.dataset_name in ["CIFAR10", "CIFAR100"]:
        # trainset = getattr(torchvision.datasets, data_args.dataset_name)(
        #     root="data/", train=True, download=True, transform=transformation
        # )
        testset = getattr(torchvision.datasets, data_args.dataset_name)(
            root="data/", train=False, download=True, transform=transformation
        )
    elif data_args.dataset_name == "CelebA":
        # trainset = getattr(torchvision.datasets, data_args.dataset_name)(
        #     root="data/",
        #     split="train",
        #     download=False,
        #     target_type="attr",
        #     transform=transformation,
        # )
        testset = getattr(torchvision.datasets, data_args.dataset_name)(
            root="data/",
            split="test",
            download=False,
            target_type="attr",
            transform=transformation,
        )
    elif data_args.dataset_name == "Places365":
        # trainset = getattr(torchvision.datasets, data_args.dataset_name)(
        #     root="data/",
        #     split="train-standard",
        #     small=True,
        #     download=False,
        #     transform=transformation,
        # )
        testset = getattr(torchvision.datasets, data_args.dataset_name)(
            root="data/",
            split="val",
            small=True,
            download=False,
            transform=transformation,
        )
    elif data_args.dataset_name == "INaturalist":
        # trainset = getattr(torchvision.datasets, data_args.dataset_name)(
        #     root="data/",
        #     version="2021_train_mini",
        #     download=False,
        #     transform=transformation,
        # )
        testset = getattr(torchvision.datasets, data_args.dataset_name)(
            root="data/", version="2021_valid", download=False, transform=transformation
        )
    elif data_args.dataset_name == "ImageNet":
        # trainset = getattr(torchvision.datasets, data_args.dataset_name)(
        #     root="data/", split="train", transform=transformation
        # )
        testset = getattr(torchvision.datasets, data_args.dataset_name)(
            root="data/", split="val", transform=transformation
        )
    return testset
