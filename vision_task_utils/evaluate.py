"""Evaluate CV with PyTorch."""

import pdb
from time import time

from utils.blocker import reconstruct_weight, reconstruct_weight_helper


import torch
from tqdm import tqdm
import numpy as np
from utils.common import load_model
from vision_task_utils.dataset import load_vision_dataset


def evaluate(
    data_args,
    model_args,
    training_args,
    model_info,
    model_constitution=None,
    model_storage=None,
    model_id=None,
    blocks: np.array = None,
):
    device = torch.device("cuda")

    testset = load_vision_dataset(data_args.dataset_name)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=16, shuffle=False, num_workers=4
    )
    data_args.val_size = len(testset)

    model = load_model(model_info, model_args)[0]
    model.to(device)

    # Reconstruct the parameters using the model constitution
    if model_constitution:
        if blocks is None:
            reconstruct_weight(model_storage, model, model_constitution)
        else:
            reconstruct_weight_helper(model, blocks, 0, model_constitution)

    if data_args.dataset_name == "CelebA":
        criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    else:
        criterion = torch.nn.CrossEntropyLoss()

    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        tic = time()
        for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            if data_args.dataset_name == "CelebA":
                loss = criterion(outputs, targets.float()).sum(dim=1).mean()
            else:
                loss = criterion(outputs, targets)

            test_loss += loss.item()
            total += targets.size(0)
            if data_args.dataset_name == "CelebA":
                correct += ((outputs > 0) == targets).sum(dim=0).float().mean().item()
            else:
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
        acc = correct / total
        loss = test_loss / (batch_idx + 1)
        tok = time()

        print(f"Test Loss: {loss:.3f} | Test Acc: {acc:.5f} | Time: {tok-tic:.3f}s")

    return acc
