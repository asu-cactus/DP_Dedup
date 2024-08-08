import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_DISABLED"] = "true"
from pathlib import Path
import time

import timm
from opacus.validators import ModuleValidator
import torch

from text_task_utils.evaluate import evaluate
from vision_task_utils.evaluate import evaluate
from utils.parse_args import parse_args
from utils.common import load_models_info
from utils.blocker import block_model_1d


def text_task_evaluate():
    model_args, data_args, training_args = parse_args()
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
    models_info = load_models_info(model_args)
    for model_info in models_info:
        tic = time.time()
        data_args.task_name = model_info["task_name"]
        print(f"Model: {model_info['model_path']}")
        model_args.model_name_or_path = model_info["model_path"]
        acc = evaluate(data_args, model_args, training_args, model_info)
        print(f"Accuracy: {acc}")
        print(f"Time taken: {time.time() - tic}")


def vision_task_parameters():


    model = timm.create_model("resnet152.tv2_in1k", pretrained=True, num_classes=40)
    model = ModuleValidator.fix(model)
    model.load_state_dict(
        torch.load("../models/in1k_CelebA_eps0.4.pt", map_location="cpu")
    )

    for name, param in model.named_parameters():
        print(name, param.size())

    total_number_of_parameters = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_number_of_parameters}")
    model_storage = block_model_1d(model)
    untouch_weights = model_storage["untouch_weights"]
    untouch_weights_counts = 0
    for weight in untouch_weights.values():
        untouch_weights_counts += weight.size
    print(f"Number of untouch weights: {untouch_weights_counts}")


def vision_task_evaluation():
    model_args, data_args, training_args = parse_args()
    models_info = load_models_info(model_args)
    # model_paths = [info["model_path"] for info in models_info]

    for model_info in models_info:
        tic = time.time()
        acc = evaluate(
            data_args,
            model_args,
            training_args,
            model_info,
        )
        print(f"Accuracy: {acc}")
        print(f"Time taken: {time.time() - tic}")


if __name__ == "__main__":

    text_task_evaluate()
    # vision_task_evaluation()
    # vision_task_parameters()
