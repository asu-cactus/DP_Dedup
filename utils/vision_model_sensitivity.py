import pdb

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import timm

from utils.blocker import block_model_1d
from utils.parse_args import parse_args
from utils.common import load_model, load_vision_dateset


def get_block_sensitivity(
    model_info, measure, skip_embeds=False, return_n_embed_blocks=False
):
    model_args, data_args, training_args = parse_args()
    data_args.task_name = model_info["task_name"]
    model_args.model_name_or_path = model_info["model_path"]
    # Get the model and dataset
    dataset = load_vision_dateset(data_args)
    model = load_model(model_info, model_args)[0]

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
