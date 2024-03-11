import numpy as np
import torch.nn.functional as F
import torch

from text_task_utils.models import RobertaForPromptFinetuning

import math
import os
import pdb

BLOCKSIZE = 196608  # 768 * 256 for base models and 1024 * 192 for large models


def block_model_1d(
    model,  # PyTorch model
):
    """
    Partition the model weights into blocks with given block size.
    TODO: try flatten the weights to avoid padded wasted space.
    """

    # DP-trained models have large variance that depend on the amount of noise added.
    # We normalize weights to have unit variance and will demonormalize during reconstruction.
    scale_factors = []

    # Start blocking
    bias_dict = {}
    blocks = []
    # nblocks = []
    n_weights = 0

    for i, params in enumerate(model.parameters()):
        n_weights += 1
        params.requires_grad = False
        # No deduplicating 1-d vectors (mostly bias)
        if params.dim() == 1:
            bias_dict[i] = params.numpy()
            continue

        # Normalize weights
        std = params.var().sqrt().item()
        params = params / std
        print(f"Layer {i} std: {std}")
        scale_factors.append(std)

        # Block 2-d matrix
        params_flatten = params.flatten()
        remainder = params_flatten.shape[0] % BLOCKSIZE
        if remainder != 0:
            params_flatten = F.pad(
                params_flatten, (0, BLOCKSIZE - remainder), value=0.0
            )
        block = params_flatten.reshape(-1, BLOCKSIZE)
        blocks.append(block.numpy())
        # nblocks.append(block.shape[0])

    model_storage = {
        "scale_factors": np.array(scale_factors, dtype=np.float32),
        "blocks": np.concatenate(blocks, axis=0),
        # "nblocks": nblocks,  # list of int
        "bias_dict": bias_dict,  # list of numpy array
        "n_weights": n_weights,  # int
    }

    # blocks = torch.cat(blocks, axis=0)
    return model_storage


def get_blocks(
    blocks_dir: str = "block_storage",
    model_paths: list[str] = None,
    npz_filename: str = "model_storage",
):
    """
    Get blocks from a given model or a list of models.
    """
    # model_paths = ["roberta-base", "roberta-base"]
    blocks_path = f"{blocks_dir}/{npz_filename}.npz"
    if not os.path.exists(blocks_path) and model_paths is not None:
        blocks = []
        biases = {}
        # block_names = []
        scale_factors = []
        model_range = [0]

        # Load blocks from blocks_path
        for ith_model, model_path in enumerate(model_paths):
            model = RobertaForPromptFinetuning.from_pretrained(model_path)

            model_storage = block_model_1d(model)
            # model_storage["nblocks"].reverse()

            # Merge blocks
            blocks.append(model_storage["blocks"])
            model_range.append(model_range[-1] + blocks[-1].shape[0])

            # Merge scale_factors
            # model_storage["scale_factors"].reverse()
            scale_factors.append(model_storage["scale_factors"])

            #
            # Merge model storage for all models
            for jth_weight in range(model_storage["n_weights"]):
                # Merge biases
                if jth_weight in model_storage["bias_dict"]:
                    biases[f"{ith_model}_{jth_weight}"] = model_storage["bias_dict"][
                        jth_weight
                    ]
                    continue

                # # Merge block_names
                # nblocks = model_storage["nblocks"].pop()
                # block_names.extend(
                #     [f"{ith_model}_{jth_weight}_{k}" for k in range(nblocks)]
                # )

                # # Merge scale_factors
                # scale_factors[f"{ith_model}_{jth_weight}"] = model_storage[
                #     "scale_factors"
                # ].pop()

        # Get rid of the first dummy block
        blocks = np.concatenate(blocks, axis=0)

        # Save blocks, biases, block_names, scale_factors to blocks_path
        np.savez(
            blocks_path,
            blocks=blocks,  #  numpy array, shape: (n_all_blocks, BLOCKSIZE)
            # block_names=np.array(block_names),  #  numpy array of type str
            model_range=np.array(model_range),  #  numpy array of type int
            biases=biases,  # dict: {f"{ith_model}_{jth_weight}": bias}
            scale_factors=np.array(scale_factors, dtype=object),  # list of numpy arrays
            model_paths=np.array(model_paths),  #  numpy array of type str
        )

    # Load models from blocks_path
    model_storage = np.load(blocks_path, allow_pickle=True)

    return model_storage


def reconstruct_weight(model_storage, model, model_id, model_constitution: list[int]):
    """Reconstruct a model weights from a given model storage object."""
    start_idx = 0
    non_bias_idx = 0
    model_start_point = model_storage["model_range"][model_id]

    for _, params in enumerate(model.parameters()):
        params.requires_grad = False
        if params.dim() == 1:
            # For now, assume the models are loaded from original storage
            # So no need to reconstruct bias from model_storage
            # TODO: load from model_storage
            continue
        # Reconstruct weights
        numel = params.numel()
        nblocks_for_params = math.ceil(numel / BLOCKSIZE)
        end_idx = start_idx + nblocks_for_params
        constitution_range = model_constitution[start_idx:end_idx]
        # Avoid reconstructing new weights if the model constitution is the same as the original one
        # `model_start_point` is the index of the first block of the model
        # `start_idx` is the index of the first block of the current layer
        start = model_start_point + start_idx
        if not all(
            (block_id == start + i for i, block_id in enumerate(constitution_range))
        ):
            new_weight = model_storage["blocks"][constitution_range].flatten()[:numel]
            # Scale weights
            new_weight = (
                new_weight * model_storage["scale_factors"][model_id][non_bias_idx]
            )
            # Set parameter to new weight
            params.copy_(torch.from_numpy(new_weight.reshape(params.shape)))

        start_idx = end_idx
        non_bias_idx += 1


if __name__ == "__main__":
    model_storage = get_blocks()
    pdb.set_trace()
