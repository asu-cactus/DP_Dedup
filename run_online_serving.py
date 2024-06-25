import torch
import timm
from opacus.validators import ModuleValidator
import numpy as np

import math
import json
import argparse
from time import time
import pdb

from vision_task_utils.dataset import load_vision_dataset


def load_models_info(args):
    if args.n_models == 5:
        model_info_path = "models/vision_vit.json"
    elif args.n_models == 20:
        model_info_path = "models/vision_vit_20models.json"
    else:
        raise ValueError("Invalid number of models")
    with open(model_info_path, "r") as f:
        models_info = json.load(f)
    models_info = list(models_info.values())
    # for info in models_info:
    #     print(info)
    return models_info


def reconstruct_weight(model, model_storage, model_id):
    # pdb.set_trace()
    model_constitution = model_storage["model_constitution"][model_id]
    blocks = model_storage["blocks"]
    untouched_weights = model_storage["untouch_weights"][model_id]

    start_idx = 0
    block_size = blocks.shape[1]

    for name, params in model.named_parameters():
        params.requires_grad_(False)
        if params.squeeze().dim() == 1 or params.numel() < block_size:
            params.copy_(torch.from_numpy(untouched_weights[name]))
            continue
        # Reconstruct weights
        numel = params.numel()
        nblocks_for_params = math.ceil(numel / block_size)
        end_idx = start_idx + nblocks_for_params
        constitution_range = model_constitution[start_idx:end_idx]

        try:
            new_weight = blocks[constitution_range].flatten()[:numel]
        except:
            pdb.set_trace()
        # Set parameter to new weight
        params.copy_(torch.from_numpy(new_weight.reshape(params.shape)))

        start_idx = end_idx


def inference(args, model_ids):
    inference_time = 0.0
    model_loading_time = 0.0
    device = torch.device("cuda:0" if args.gpu else "cpu")

    assert args.batch_size % args.mini_bs == 0
    n_iter = args.batch_size // args.mini_bs

    # Load model from disk
    models_info = load_models_info(args)
    model = timm.create_model(args.model_name, num_classes=args.num_classes)
    model = ModuleValidator.fix(model)

    # Load model_storage
    if args.load_from == "memory":
        if args.n_models == 5:
            storage_path = "../models/vision_vit_5models_storage.npz"
        else:
            storage_path = "../models/vision_vit_20models_storage.npz"
        model_storage = np.load(storage_path, allow_pickle=True)

    # Load dataset
    testset = load_vision_dataset(args)

    for model_id in model_ids:
        model_loading_start = time()
        model.to("cpu")
        if args.load_from == "disk":
            model_path = models_info[model_id]["model_path"]
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
        else:
            reconstruct_weight(model, model_storage, model_id)
        model.to(device)
        model_loading_end = time()
        model_loading_time += model_loading_end - model_loading_start

        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=args.mini_bs,
            shuffle=False,
            num_workers=args.num_workers,
        )

        for i, (inputs, target) in enumerate(testloader):
            if i == n_iter:
                break
            inference_start = time()
            with torch.no_grad():
                model(inputs.to(device))
            inference_end = time()
            inference_time += inference_end - inference_start

    print(f"Model loading time: {model_loading_time:.4f}")
    print(f"Inference time: {inference_time:.4f}")
    print(f"Total time: {model_loading_time + inference_time:.4f}")


def workload_generate(args):
    repeats = 10
    if args.workload == "random":
        rng = np.random.default_rng(seed=42)
        n_runs = repeats * args.n_models
        model_ids = rng.integers(args.n_models, size=n_runs)
    else:
        model_ids = np.tile(np.arange(args.n_models), repeats)
    print(f"Workload: {model_ids}")
    return model_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare inference time")
    parser.add_argument(
        "-L",
        "--load_from",
        required=True,
        type=str,
        choices=["memory", "disk"],
        help="Whether to load model from memory or from disk",
    )
    parser.add_argument(
        "-W",
        "--workload",
        required=True,
        type=str,
        choices=["random", "roundrobin"],
        help="Workload type",
    )
    parser.add_argument("--n_models", default=5, type=int, help="Number of models")
    parser.add_argument(
        "--mini_bs",
        type=int,
        default=16,
        help="Mini batch size for inference",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for dataloader",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for inference",
    )

    args = parser.parse_args()
    args.dataset_name = "CIFAR100"
    args.model_name = "vit_large_patch16_224"
    args.num_classes = 100
    print(args)

    model_ids = workload_generate(args)
    inference(args, model_ids)
