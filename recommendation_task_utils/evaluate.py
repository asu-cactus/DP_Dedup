from fastDP import PrivacyEngine
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from opacus.accountants.utils import get_noise_multiplier

import warnings
import os
import time
import pdb

from utils.blocker import reconstruct_weight, reconstruct_weight_helper

warnings.filterwarnings("ignore")


class RecommendationDataset(Dataset):
    def __init__(self, path):
        with open(path, "r") as f:
            self.lines = f.readlines()[1:]
        self.num_features = 47236
        self.num_labels = 5
        self.nclasses = 500

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        elements = self.lines[idx].split(" ")
        label_part = elements[0]
        feature_parts = elements[1:]

        # Get labels
        index_set = {int(val) // self.nclasses for val in label_part.split(",")}
        index = torch.tensor(list(index_set))
        labels = torch.zeros(self.num_labels, dtype=torch.long).scatter_(0, index, 1)

        # Get features
        features = torch.zeros(self.num_features, dtype=torch.float32)
        for ft_pairs in feature_parts:
            ft, ft_val = ft_pairs.split(":")
            features[int(ft)] = float(ft_val)

        # Return features and labels
        return features, labels


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(47236, 5000),
            nn.ReLU(),
            nn.Linear(5000, 5),
        )

    def forward(self, x):
        return self.model(x)


# def obtain_dataset_info(path_to_train, path_to_test):
#     """Obtain the dataset info from the given path, the data info is stored in the
#     first line

#     Args:
#         path_to_train (str): path to the training data
#         path_to_test (str): path to the testing data

#     Returns:
#         (int, int, int, int): number of training data, number of testing data,
#                               number of features, number of labels
#     """
#     with open(path_to_train) as f:
#         line = f.readline()
#     num_train, num_features, num_labels = line.split(" ")
#     num_labels = math.ceil(int(num_labels) / CLASS_SIZE)

#     with open(path_to_test) as f:
#         line = f.readline()
#     num_test, _, _ = line.split(" ")

#     return int(num_train), int(num_test), int(num_features), num_labels


def load_dataset(training_args):
    data_root = "data"
    path_to_train = os.path.join(data_root, "RCV1-2K", "train.txt")
    path_to_test = os.path.join(data_root, "RCV1-2K", "test.txt")

    # ds_train = RecommendationDataset(path_to_train)
    # trainloader = DataLoader(
    #     ds_train, batch_size=training_args.mini_bs, shuffle=True, num_workers=4
    # )
    ds_test = RecommendationDataset(path_to_test)
    testloader = DataLoader(
        ds_test, batch_size=training_args.mini_bs, shuffle=False, num_workers=4
    )
    test_size = len(ds_test)
    # return (trainloader, testloader, val_size)
    return testloader, test_size


def load_model(model_path):
    """Load a extreme classification model"""

    model = LinearModel()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    return model
    # optimizer = torch.optim.Adam(model.parameters(), lr=training_args.lr)

    # # Privacy engine
    # if "nonDP" not in training_args.clipping_mode:
    #     sigma = get_noise_multiplier(
    #         target_epsilon=training_args.epsilon,
    #         target_delta=1 / datasize,
    #         sample_rate=training_args.bs / datasize,
    #         epochs=2,
    #     )
    #     print(f"adding noise level {sigma}")
    #     privacy_engine = PrivacyEngine(
    #         model,
    #         batch_size=training_args.bs,
    #         sample_size=training_args.num_train,
    #         noise_multiplier=sigma,
    #         epochs=2,
    #         clipping_mode="MixOpt",
    #         clipping_style="all-layer",
    #     )
    #     privacy_engine.attach(optimizer)

    # return model, optimizer


def evaluate(
    data_args,
    model_args,
    training_args,
    model_info,
    model_constitution=None,
    model_storage=None,
    model_id=None,
    blocks=None,
):
    testloader, test_size = load_dataset(training_args)
    model = load_model(model_info["model_path"])

    if model_constitution:
        if blocks is None:
            reconstruct_weight(model_storage, model, model_id, model_constitution)
        else:
            reconstruct_weight_helper(model, blocks, 0, model_constitution)

    device = torch.device(f"cuda")
    model.to(device)
    model.eval()

    correct, total = 0, 0
    tic = time.time()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            # Record loss
            total += targets.size(0)
            indices = torch.argmax(outputs, dim=1, keepdim=True)
            correct += torch.gather(targets, 1, indices).sum()
    toc = time.time()
    top1_prec = 100.0 * correct / total
    print(f"Top1 Precision: {top1_prec:.3f}% | Time: {toc-tic:.3f}s")
    return top1_prec
