import torch
import timm
from opacus.validators import ModuleValidator
import numpy as np
from transformers import default_data_collator

import math
from random import choice
import json
import argparse
from time import time
import pdb

from vision_task_utils.dataset import load_vision_dataset
from text_task_utils.evaluate import evaluate

torch.manual_seed(42)


def load_models_info(args):
    if args.dataset_name == "CIFAR100":
        model_info_path = "models/vision_vit_20models.json"
    elif args.dataset_name == "CelebA":
        model_info_path = "models/vision_resnet_20models.json"
    elif args.dataset_name == "qnli":
        model_info_path = "models/text_10models.json"
    else:
        raise ValueError("Unknown dataset or task name")

    with open(model_info_path, "r") as f:
        models_info = json.load(f)
    models_info = list(models_info.values())
    return models_info


def load_model_storage(args):
    if args.dataset_name == "CIFAR100":
        storage_path = f"../models/vision_vit_10models_storage.npz"
    elif args.dataset_name == "CelebA":
        storage_path = f"../models/vision_resnet_20models_storage.npz"
    elif args.dataset_name == "qnli":
        storage_path = f"../models/text_10models_storage.npz"
    else:
        raise ValueError("Unknown dataset or task name")
    model_storage = np.load(storage_path, allow_pickle=True)
    return model_storage


def load_model_and_testset(args, model_info=None):
    if args.dataset_name in ["CIFAR100", "CelebA"]:
        model = timm.create_model(args.model_name, num_classes=args.num_classes)
        model.to("cpu")
        model = ModuleValidator.fix(model)
        testset = load_vision_dataset(args)
    else:
        args.model_name_or_path = model_info["model_path"]
        model, testset, _ = evaluate(args, args, args, return_verbose=True)
    return model, testset


def reconstruct_weight(model, blocks, model_constitution, untouched_weights):
    start_idx = 0
    block_size = blocks.shape[1]

    for name, params in model.named_parameters():
        if params.squeeze().dim() == 1 or params.numel() < block_size:
            params.copy_(torch.from_numpy(untouched_weights[name]))
            continue
        # Reconstruct weights
        numel = params.numel()
        nblocks_for_params = math.ceil(numel / block_size)
        end_idx = start_idx + nblocks_for_params
        constitution_range = model_constitution[start_idx:end_idx]
        new_weight = blocks[constitution_range].flatten()[:numel]
        # Set parameter to new weight
        params.copy_(torch.from_numpy(new_weight.reshape(params.shape)))

        start_idx = end_idx


def inference(args, model_ids):
    inference_time = 0.0
    load_or_construct_time = 0.0
    # device = torch.device("cuda:0" if args.gpu else "cpu")

    assert args.batch_size % args.mini_bs == 0
    n_iter = args.batch_size // args.mini_bs

    # Load model from disk
    models_info = load_models_info(args)
    model, testset = load_model_and_testset(args, models_info[0])

    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    # Load model_storage
    if args.load_from == "memory":
        model_storage = load_model_storage(args)
        model_constitution = model_storage["model_constitution"]
        blocks = model_storage["blocks"]
        untouched_weights = model_storage["untouch_weights"]

    for model_id in model_ids:
        if args.load_from == "disk":
            if args.dataset_name in ["CIFAR100", "CelebA"]:
                model_loading_start = time()
                model_path = models_info[model_id]["model_path"]
                model.load_state_dict(torch.load(model_path, map_location="cpu"))
                load_or_construct_time += time() - model_loading_start
            else:
                args.model_name_or_path = models_info[model_id]["model_path"]
                model, testset, loading_time = evaluate(
                    args, args, args, return_verbose=True
                )
                load_or_construct_time += loading_time
        else:
            model_construct_start = time()
            reconstruct_weight(
                model, blocks, model_constitution[model_id], untouched_weights[model_id]
            )
            load_or_construct_time += time() - model_construct_start
        # model.to(device)

        collate_fn = default_data_collator if args.dataset_name == "qnli" else None
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=args.mini_bs,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )
        for i, item in enumerate(testloader):
            if i == n_iter:
                break
            inference_start = time()
            if args.dataset_name == "qnli":
                # pdb.set_trace()
                # input_ids = torch.tensor(item.input_ids, dtype=torch.long)
                # attention_mask = torch.tensor(item.attention_mask, dtype=torch.long)
                # mask_pos = torch.tensor(item.mask_pos, dtype=torch.long)
                item.pop("labels")
                with torch.no_grad():
                    model(**item)
            else:
                images = item[0]
                with torch.no_grad():
                    model(images)
            inference_time += time() - inference_start

    print(f"Model load or construct time: {load_or_construct_time:.4f}")
    print(f"Inference time: {inference_time:.4f}")
    print(f"Total time: {load_or_construct_time + inference_time:.4f}")


def workload_generate(args):
    n_queries = args.n_queries
    n_models = args.n_models
    assert n_queries % n_models == 0
    if args.workload == "random":
        rng = np.random.default_rng(seed=42)
        model_ids = rng.integers(n_models, size=n_queries)
    else:
        model_ids = np.tile(np.arange(n_models), n_queries // n_models)
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
    parser.add_argument(
        "-D",
        "--dataset_name",
        required=True,
        type=str,
        choices=["CIFAR100", "CelebA", "qnli"],
        help="Dataset name",
    )
    parser.add_argument(
        "--n_queries",
        type=int,
        default=100,
        help="Number of queries",
    )

    parser.add_argument(
        "--mini_bs",
        type=int,
        default=1,
        help="Mini batch size for inference",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers for dataloader",
    )
    # parser.add_argument(
    #     "--gpu",
    #     action="store_true",
    #     help="Use GPU for inference",
    # )

    args = parser.parse_args()
    if args.dataset_name == "CIFAR100":
        args.n_models = 10
        args.num_classes = 100
        args.model_name = "vit_base_patch16_224"
    elif args.dataset_name == "CelebA":
        args.n_models = 20
        args.num_classes = 40
        args.model_name = "resnet152.tv2_in1k"
    elif args.dataset_name == "qnli":
        args.n_models = 10
        args.task_name = args.dataset_name
        args.few_shot_type = "prompt-demo"
        args.prompt = False
        args.template_path = None
        args.data_root_dir = (
            "../fast-differential-privacy/examples/text_classification/data/original/"
        )
        args.prompt_path = None
        args.mapping_path = None
        args.auto_demo = True
        args.gpt3_in_context_head = False
        args.gpt3_in_context_tail = False
        args.template_list = None
        args.config_name = None
        args.cache_dir = None
        args.tokenizer_name = None
        args.num_sample = 1
        args.max_seq_length = 256
        args.overwrite_cache = False
        args.demo_filter = False
        args.inference_time_demo = False
        args.double_demo = False
        args.first_sent_limit = None
        args.other_sent_limit = None
        args.truncate_head = None

    else:
        raise ValueError("Unknown dataset or task name")
    print(args)

    model_ids = workload_generate(args)
    inference(args, model_ids)
