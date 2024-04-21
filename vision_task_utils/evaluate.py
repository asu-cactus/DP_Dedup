"""Train CV with PyTorch."""

import pdb
from time import time

from utils.blocker import reconstruct_weight, reconstruct_weight_helper


import torch
import torchvision
from tqdm import tqdm
import timm
import numpy as np


def evaluate(
    model_storage,
    model_id,
    model_constitution,
    data_args,
    model_args,
    training_args,
    blocks: np.array = None,
):

    device = torch.device("cuda")

    # Data
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

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=16, shuffle=False, num_workers=4
    )

    # Model
    model = timm.create_model(
        model_args.model, pretrained=True, num_classes=num_classes
    )
    model.cuda()

    # Reconstruct the parameters using the model constitution
    if model_constitution:
        if blocks is None:
            reconstruct_weight(model_storage, model, model_id, model_constitution)
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
                correct += ((outputs > 0) == targets).sum(dim=0).float().mean()
            else:
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
        acc = 100.0 * correct / total
        loss = test_loss / (batch_idx + 1)
        tok = time()

        print(f"Test Loss: {loss:.3f} | Test Acc: {acc:.4f} | Time: {tok-tic:.3f}s")

    return acc


# if __name__ == "__main__":
# import os

# os.environ["TQDM_DISABLE"] = "1"

# import argparse

# parser = argparse.ArgumentParser(description="PyTorch CIFAR Training")
# parser.add_argument("--lr", default=5e-4, type=float, help="learning rate")
# parser.add_argument("--epochs", default=5, type=int, help="numter of epochs")
# parser.add_argument("--bs", default=1000, type=int, help="batch size")
# parser.add_argument("--mini_bs", type=int, default=100)
# parser.add_argument("--epsilon", default=8, type=float, help="target epsilon")
# parser.add_argument(
#     "--dataset_name",
#     type=str,
#     default="CelebA",
#     help="https://pytorch.org/vision/stable/datasets.html",
# )
# parser.add_argument(
#     "--clipping_mode",
#     type=str,
#     default="MixOpt",
#     choices=["BiTFiT", "MixOpt", "nonDP", "nonDP-BiTFiT"],
# )
# parser.add_argument(
#     "--model", default="vit_small_patch16_224", type=str, help="model name"
# )
# parser.add_argument("--gpu_id", type=int, default=0)

# args = parser.parse_args()
# print(args)

# from fastDP import PrivacyEngine

# import torch
# import torchvision

# torch.manual_seed(2)
# import torch.nn as nn
# import torch.optim as optim
# import timm
# from opacus.validators import ModuleValidator
# from opacus.accountants.utils import get_noise_multiplier
# from tqdm import tqdm
# import warnings

# warnings.filterwarnings("ignore")
# evaluate()
