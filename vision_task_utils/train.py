"""Train CV with PyTorch."""

import pdb
from time import time

from utils.blocker import reconstruct_weight
from utils.common import load_vision_dateset, load_model
from utils.blocker import BLOCKSIZE

import torch
from tqdm import tqdm
import numpy as np
from opacus.accountants.utils import get_noise_multiplier
from fastDP import PrivacyEngine


def train(
    data_args,
    model_args,
    training_args,
    model_info,
    model_constitution,
    model_storage,
    model_id,
):
    device = torch.device("cuda")

    # Load the dataset
    trainset = load_vision_dateset(data_args)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=training_args.mini_bs, shuffle=True, num_workers=4
    )

    # Reconstruct the parameters using the model constitution
    model = load_model(model_info, model_args)[0]
    reconstruct_weight(model_storage, model, model_id, model_constitution)
    model.to(device)

    # Train only untouched weights
    for name, params in model.named_parameters():
        if params.squeeze().dim() == 1 or params.numel() < BLOCKSIZE:
            params.requires_grad_(True)
        else:
            params.requires_grad_(False)

    # Set the criterion and optimizer
    if data_args.dataset_name == "CelebA":
        criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    else:
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    # Apply Privacy Engine
    epochs = 2
    target_epsilon = 0.1
    sigma = get_noise_multiplier(
        target_epsilon=target_epsilon,
        target_delta=1 / len(trainset),
        sample_rate=training_args.bs / len(trainset),
        epochs=epochs,
    )
    privacy_engine = PrivacyEngine(
        model,
        batch_size=training_args.bs,
        sample_size=len(trainset),
        noise_multiplier=sigma,
        epochs=epochs,
        clipping_mode="MixOpt",
        clipping_style="all-layer",
    )
    privacy_engine.attach(optimizer)

    # Start training
    model.train()
    n_acc_steps = training_args.bs // training_args.mini_bs
    for epoch in range(epochs):
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            if data_args.dataset_name == "CelebA":
                loss = criterion(outputs, targets.float()).sum(dim=1).mean()
            else:
                loss = criterion(outputs, targets)
                # print(loss.item())

            loss.backward()
            if ((batch_idx + 1) % n_acc_steps == 0) or (
                (batch_idx + 1) == len(trainloader)
            ):
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item()
            total += targets.size(0)
            if data_args.dataset_name == "CelebA":
                correct += ((outputs > 0) == targets).sum(dim=0).float().mean()
            else:
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()

        print(
            "Epoch: ",
            epoch + 1,
            "Train Loss: %.3f | Acc: %.3f%% (%d/%d)"
            % (train_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
        )

    # Update the model storage
    untouch_weights = model_storage["untouch_weights"][model_id]
    for name, params in model.named_parameters():
        if params.squeeze().dim() == 1 or params.numel() < BLOCKSIZE:
            untouch_weights[name] = params.detach().cpu().numpy()
