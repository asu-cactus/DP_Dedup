import pdb
from pathlib import Path

import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

from final_constitution_quantile import model0, model1
from utils.parse_args import parse_args
from utils.blocker import get_blocks
from utils import load_models_info
from text_task_utils.evaluate import evaluate


def compute_measure(blocks: np.ndarray, measure: str):
    if measure == "3rd_quartile":
        return np.quantile(blocks, 0.75, axis=1)
    elif measure == "l2_norm":
        return np.linalg.norm(blocks, axis=1)
    else:
        raise NotImplementedError


def plot_constitution_heatmap():
    # Load model storage
    model_args, data_args, training_args = parse_args()
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
    models_info = load_models_info(model_args)
    model_paths = [info["model_path"] for info in models_info]
    models_storage = get_blocks(
        model_paths=model_paths, npz_filename="model_storage_nonorm"
    )

    blocks = models_storage["blocks"]

    measures_before = compute_measure(blocks, measure="3rd_quartile")
    measures0_after = measures_before[model0].reshape(-1, 1)
    measures1_after = measures_before[model1].reshape(-1, 1)

    measures_before = measures_before.reshape(2, -1).transpose()
    measures = np.concatenate(
        (measures_before, measures0_after, measures1_after), axis=1
    )

    df = pd.DataFrame(
        measures,
        columns=[
            "model0_original",
            "model1_original",
            "model0_deduped",
            "model1_deduped",
        ],
    )

    fig, ax = plt.subplots(figsize=(15, 5))
    sns.heatmap(df.T, cmap="YlGnBu", yticklabels=df.columns, ax=ax)
    fig.savefig("heatmap.png", dpi=300, bbox_inches="tight", pad_inches=0.1)


def test_evaluate():
    model_args, data_args, training_args = parse_args()
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
    models_info = load_models_info(model_args)
    model_paths = [info["model_path"] for info in models_info]
    models_storage = get_blocks(
        model_paths=model_paths, npz_filename="model_storage_4_11"
    )
    data_args.task_name = models_info[0]["task_name"]
    model_args.model_name_or_path = models_info[0]["model_path"]
    acc = evaluate(
        models_storage,
        0,
        model0,
        data_args,
        model_args,
        training_args,
    )
    print(f"Accuracy: {acc}")

    data_args.task_name = models_info[1]["task_name"]
    model_args.model_name_or_path = models_info[1]["model_path"]
    acc = evaluate(
        models_storage,
        1,
        model1,
        data_args,
        model_args,
        training_args,
    )
    print(f"Accuracy: {acc}")


def check():
    count = 0

    for i, block in enumerate(model1):
        if block == i + 833:
            count += 1

    print(count)


if __name__ == "__main__":
    # plot_constitution_heatmap()
    test_evaluate()
    # check()
