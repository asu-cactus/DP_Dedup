import pdb

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import timm

from utils.blocker import block_model_1d
from utils.parse_args import parse_args


def get_model_and_dateset(model_info, data_args, model_args, training_args):
    # Load model
    if model_args.task_type == "text":
        from text_task_utils.models import RobertaForPromptFinetuning
    elif model_args.task_type == "vision":
        if model_info["task_name"] == "CIFAR100":
            num_classes = 100
        elif model_info["task_name"] == "CelebA":
            num_classes = 40
    else:
        raise ValueError(f"Invalid task name: {model_args.task_type}")

    if model_args.task_type == "text":
        model = RobertaForPromptFinetuning.from_pretrained(model_info["model_path"])
    elif model_args.task_type == "vision":
        model = timm.create_model(
            model_args.model, pretrained=True, num_classes=num_classes
        )
        model.load_state_dict(torch.load(model_info["model_path"]))

    # Load data
    transformation = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(
                (224, 224)
            ),  # https://discuss.pytorch.org/t/runtimeerror-stack-expects-each-tensor-to-be-equal-size-but-got-3-224-224-at-entry-0-and-3-224-336-at-entry-3/87211/10
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    if data_args.dataset_name in ["SVHN", "CIFAR10"]:
        num_classes = 10
    elif data_args.dataset_name in ["CIFAR100", "FGVCAircraft"]:
        num_classes = 100
    elif data_args.dataset_name in ["Food101"]:
        num_classes = 101
    elif data_args.dataset_name in ["GTSRB"]:
        num_classes = 43
    elif data_args.dataset_name in ["CelebA"]:
        num_classes = 40
    elif data_args.dataset_name in ["Places365"]:
        num_classes = 365
    elif data_args.dataset_name in ["ImageNet"]:
        num_classes = 1000
    elif data_args.dataset_name in ["INaturalist"]:
        num_classes = 10000

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
    return model, testset


def get_block_sensitivity(
    model_info, measure, skip_embeds=False, return_n_embed_blocks=False
):
    model_args, data_args, training_args = parse_args()
    data_args.task_name = model_info["task_name"]
    model_args.model_name_or_path = model_info["model_path"]
    # Get the model and dataset
    model, dataset = get_model_and_dateset(
        model_info,
        data_args,
        model_args,
        training_args,
    )

    if measure == "magnitude":
        blocks = magnitute_sensitivity(model)
    elif measure == "fisher":
        if data_args.dataset_name == "CelebA":
            raise ValueError("Fisher sensitivity is not implemented for CelebA.")
        blocks = fisher_sensitity(model, dataset, data_args.dataset_name)
    elif measure == "gradient":
        blocks = gradient_sensitity(model, dataset, data_args.dataset_name)
    else:
        raise ValueError(f"Unknown sensitivity measure: {measure}")

    return blocks, None


def magnitute_sensitivity(model):
    model_storage = block_model_1d(model)
    blocks = model_storage["blocks"]
    return blocks


def fisher_sensitity(model, dataset, batch_size=16):
    model.eval()
    model.cuda()

    sample_size = len(dataset)

    testloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    for batch_idx, (inputs, labels) in enumerate(testloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        logits = model(inputs)
        probs = F.softmax(logits, dim=1)
        logprobs = F.log_softmax(logits, dim=1)

        probs = probs.gather(1, labels).squeeze()
        logprobs = logprobs.gather(1, labels).squeeze()

        # Initialize Fisher information matrix
        fim = {}
        no_grad_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                fim[name] = torch.zeros_like(param, device="cpu")
            else:
                no_grad_params[name] = param
        assert len(no_grad_params) == 0, "There are some parameters without grad."

        # Compute Fisher information
        for logprob, prob in zip(logprobs, probs):
            model.zero_grad()
            torch.autograd.backward(logprob, retain_graph=True)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    fim[name] += (param.grad.square() * prob).detach().cpu()

    fim = {k: grad2 / sample_size for k, grad2 in fim.items()}

    # Block the Fisher information corresponding to each parameter
    blocks = block_model_1d(fim)["blocks"]

    return blocks


def gradient_sensitity(
    model,
    dataset,
    dataset_name,
    batch_size=16,
    sample_size=None,
):
    model.eval()
    model.cuda()

    if sample_size is None:
        sample_size = len(dataset)

    testloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    if dataset_name == "CelebA":
        criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    else:
        criterion = torch.nn.CrossEntropyLoss()
    accum_iter = sample_size / batch_size

    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)

        if dataset_name == "CelebA":
            loss = criterion(outputs, targets.float()).sum(dim=1).mean()
        else:
            loss = criterion(outputs, targets)
        loss = loss / accum_iter
        loss.backward()

    grads = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            grads[name] = param.grad

    # Block the Gradients corresponding to each parameter
    blocks = block_model_1d(grads)["blocks"]

    return blocks
