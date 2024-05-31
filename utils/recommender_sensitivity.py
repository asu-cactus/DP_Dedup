import pdb

import torch
import torch.nn.functional as F

from utils.blocker import block_model_1d
from utils.parse_args import parse_args
from utils.common import load_model
from recommendation_task_utils.evaluate import load_dataset


def get_block_sensitivity(model_info, measure, skip_embeds, return_n_embed_blocks):
    model_args, data_args, training_args = parse_args()
    data_args.task_name = model_info["task_name"]
    model_args.model_name_or_path = model_info["model_path"]
    # Get the model and dataset
    val_loader, val_size = load_dataset(training_args)
    model = load_model(model_info, model_args)[0]

    block_size = model_args.block_size
    if measure == "magnitude":
        blocks = magnitute_sensitivity(model, block_size)
    elif measure == "fisher":
        blocks = fisher_sensitity(model, val_loader, val_size, block_size)
    elif measure == "gradient":
        blocks = gradient_sensitity(model, val_loader, val_size, block_size)
    else:
        raise ValueError(f"Unknown sensitivity measure: {measure}")

    return blocks, None


def magnitute_sensitivity(model, block_size):
    model_storage = block_model_1d(block_size, model)
    blocks = model_storage["blocks"]
    return blocks


def fisher_sensitity(model, val_loader, val_size, block_size):
    model.eval()
    model.cuda()

    sample_size = val_size
    testloader = val_loader

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
    blocks = block_model_1d(block_size, fim)["blocks"]

    return blocks


def gradient_sensitity(model, val_loader, val_size, block_size):
    model.eval()
    model.cuda()

    sample_size = val_size
    accum_iter = sample_size / val_loader.batch_size

    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    # correct, total = 0, 0
    for inputs, targets in val_loader:
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, targets.float()).sum(dim=1).mean()
        loss = loss / accum_iter
        loss.backward()

    grads = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            grads[name] = param.grad

    # Block the Gradients corresponding to each parameter
    blocks = block_model_1d(block_size, grads)["blocks"]

    return blocks
