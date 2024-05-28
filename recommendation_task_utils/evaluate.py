from fastDP import PrivacyEngine
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from opacus.accountants.utils import get_noise_multiplier

from utils.blocker import reconstruct_weight, reconstruct_weight_helper

import warnings
import time
from collections import defaultdict
import pdb

torch.manual_seed(2)
warnings.filterwarnings("ignore")


class MovieLensDataset(Dataset):
    """
    The Movie Lens Dataset class. This class prepares the dataset for training and validation.
    """

    def __init__(self, users, movies, ratings):
        """
        Initializes the dataset object with user, movie, and rating data.
        """
        self.users = users
        self.movies = movies
        self.ratings = ratings

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.users)

    def __getitem__(self, item):
        """
        Retrieves a sample from the dataset at the specified index.
        """
        users = self.users[item]
        movies = self.movies[item]
        ratings = self.ratings[item]

        return {
            "users": torch.tensor([users], dtype=torch.long),
            "movies": torch.tensor([movies], dtype=torch.long),
            "ratings": torch.tensor(ratings, dtype=torch.float),
        }


def prepare_data(training_args):
    # df_train = pd.read_csv("data/ml-latest/ratings_train.csv", skiprows=1)
    df_val = pd.read_csv("data/ml-latest/ratings_test.csv", skiprows=1)

    # trainset = MovieLensDataset(
    #     df_train.userId.values, df_train.movieId.values, df_train.rating.values
    # )
    validset = MovieLensDataset(
        df_val.userId.values, df_val.movieId.values, df_val.rating.values
    )

    # train_loader = DataLoader(
    #     trainset, batch_size=args.mini_bs, shuffle=True, num_workers=4
    # )
    val_loader = DataLoader(
        validset, batch_size=training_args.mini_bs, shuffle=False, num_workers=4
    )

    val_size = len(validset)
    # train_size = len(trainset)

    print(f" Validation size: {val_size}")
    return val_loader, val_size


class RecommendationSystemModel(nn.Module):
    def __init__(
        self,
        num_users,
        num_movies,
        embedding_size,
        hidden_dim,
        dropout_rate,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size
        self.hidden_dim = hidden_dim

        # Embedding layers
        self.user_embedding = nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.embedding_size
        )
        self.movie_embedding = nn.Embedding(
            num_embeddings=self.num_movies, embedding_dim=self.embedding_size
        )

        # Hidden layers
        self.linear = nn.Sequential(
            nn.Linear(self.embedding_size, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, users, movies):
        # Embeddings
        user_embedded = self.user_embedding(users)
        movie_embedded = self.movie_embedding(movies)

        # Concatenate user and movie embeddings
        # combined = torch.cat([user_embedded, movie_embedded], dim=2)
        combined = user_embedded + movie_embedded

        # Pass through hidden layers with ReLU activation and dropout
        output = self.linear(combined)

        return output


def calculate_precision_recall(user_ratings, k, threshold):
    user_ratings.sort(key=lambda x: x[0], reverse=True)
    n_rel = sum(true_r >= threshold for _, true_r in user_ratings)
    n_rec_k = sum(est >= threshold for est, _ in user_ratings[:k])
    n_rel_and_rec_k = sum(
        (true_r >= threshold) and (est >= threshold) for est, true_r in user_ratings[:k]
    )

    precision = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
    recall = n_rel_and_rec_k / n_rel if n_rel != 0 else 1
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if precision + recall != 0
        else 0
    )
    return f1


def load_model(model_path):
    with open("data/ml-latest/ratings_test.csv", "r") as f:
        line = f.readline()
        segments = line.split(",")
        num_users = int(segments[0])
        num_movies = int(segments[1])

    model = RecommendationSystemModel(
        num_users=num_users,
        num_movies=num_movies,
        embedding_size=256,
        hidden_dim=256,
        dropout_rate=0.1,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    return model


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
    val_loader, val_size = prepare_data(training_args)
    model = load_model(model_info["model_path"])

    if model_constitution:
        if blocks is None:
            reconstruct_weight(model_storage, model, model_id, model_constitution)
        else:
            reconstruct_weight_helper(model, blocks, 0, model_constitution)

    device = torch.device("cuda")
    model.to(device)
    model.eval()

    user_ratings_comparison = defaultdict(list)
    with torch.no_grad():
        tik = time.time()
        for valid_data in val_loader:
            users = valid_data["users"].to(device)
            movies = valid_data["movies"].to(device)
            ratings = valid_data["ratings"].to(device)
            output = model(users, movies)

            for user, pred, true in zip(users, output, ratings):
                user_ratings_comparison[user.item()].append(
                    (pred[0].item(), true.item())
                )
        tok = time.time()

    k = 50
    threshold = 3
    user_based_f1 = dict()
    for user_id, user_ratings in user_ratings_comparison.items():
        f1 = calculate_precision_recall(user_ratings, k, threshold)
        user_based_f1[user_id] = f1
    average_f1 = sum(f1 for f1 in user_based_f1.values()) / len(user_based_f1)

    print(f"f1 @ {k}: {average_f1:.4f} | Time: {tok-tik:.3f}s")
    return average_f1
