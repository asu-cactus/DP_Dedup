import math
import pdb

import numpy as np
import torch.nn.functional as F
import torch

from text_task_utils.models import RobertaForPromptFinetuning


# Block size for Roberta models
# BLOCKSIZE = 196608  # 768 * 256 for base models and 1024 * 192 for large models
# BLOCKSIZE = 589824  # 768 * 768 for base models and 1024 * 576 for large models

# Block size for vision models
# BLOCKSIZE = 1048576  # 1024 * 1024 for VIT large models
# BLOCKSIZE = 262144  # 512 * 512 for ResNet152 models

# # Block size for recommendation models
# BLOCKSIZE = 131072  # 256 * 512 for movielens models
# BLOCKSIZE = 1180000  # 236 * 5000 for rcv1 models


def block_model_1d(block_size, model_or_paramdict, skip_embeds=False):
    """
    Partition the model weights into blocks with given block size.
    """

    # # DP-trained models have large variance that depend on the amount of noise added.
    # # We normalize weights to have unit variance and will demonormalize during reconstruction.
    # scale_factors = []
    # Start blocking
    untouch_weights = {}
    blocks = []

    # layer_ids, layer_names = [], []

    if isinstance(model_or_paramdict, dict):
        iterator = model_or_paramdict.items()
    else:
        iterator = model_or_paramdict.named_parameters()
    for i, (name, params) in enumerate(iterator):
        if skip_embeds and "embeddings" in name:
            continue
        params = params.detach().cpu()

        # No deduplicating 1-d vectors (mostly bias)
        if params.squeeze().dim() == 1 or params.numel() < block_size:
            untouch_weights[name] = params.numpy()
            continue

        # # Normalize weights
        # std = params.var().sqrt().item()
        # params = params / std
        # # print(f"Layer {i} std: {std}")
        # scale_factors.append(std)

        # Block 2-d matrix
        params_flatten = params.flatten()
        remainder = params_flatten.shape[0] % block_size
        if remainder != 0:
            params_flatten = F.pad(
                params_flatten, (0, block_size - remainder), value=0.0
            )
        block = params_flatten.reshape(-1, block_size)
        blocks.append(block.numpy())

    #     layer_ids.extend([i + 1] * block.shape[0])
    #     layer_names.extend([name] * block.shape[0])

    # df = pd.DataFrame(
    #     {
    #         "block_ids": np.arange(1, len(layer_ids) + 1),
    #         "layer_id": layer_ids,
    #         "layer_name": layer_names,
    #     }
    # )
    # df.to_csv("largeblock_block_vs_layer.csv", index=False)
    # exit()

    # # This part is just for experiment. Will not be used in the final version.
    # # For each block, get the layer that it belongs to, and save its start and end block index
    # search_range = np.empty((833, 4), dtype=int)
    # start_index = 0
    # for block in blocks:
    #     layer_nblocks = block.shape[0]
    #     search_range[start_index : start_index + layer_nblocks, 0] = start_index
    #     search_range[start_index : start_index + layer_nblocks, 1] = (
    #         start_index + layer_nblocks
    #     )
    #     search_range[start_index : start_index + layer_nblocks, 2] = start_index + 833
    #     search_range[start_index : start_index + layer_nblocks, 3] = (
    #         start_index + layer_nblocks + 833
    #     )
    #     start_index += layer_nblocks

    model_storage = {
        # "scale_factors": np.array(scale_factors, dtype=np.float32),
        "blocks": np.concatenate(blocks, axis=0),
        "untouch_weights": untouch_weights,  # list of numpy array
        # "search_range": search_range,  # numpy array of shape (n_all_blocks, 2)
    }

    print(f"Number of blocks: {model_storage['blocks'].shape[0]}")
    return model_storage


def get_blocks(
    # blocks_dir: str = "block_storage",
    model_paths: list[str] = None,
    # npz_filename: str = "model_storage",
):
    """
    Get blocks from a given model or a list of models.
    """
    # # Create blocks_dir if not exists
    # Path(blocks_dir).mkdir(parents=True, exist_ok=True)
    # blocks_path = f"{blocks_dir}/{npz_filename}.npz"

    # if not os.path.exists(blocks_path) and model_paths is not None:

    # search_range = None

    blocks = []
    untouch_weights = {}
    # scale_factors = []
    model_range = [0]

    # Load blocks from blocks_path
    for ith_model, model_path in enumerate(model_paths):
        model = RobertaForPromptFinetuning.from_pretrained(model_path)
        model_storage = block_model_1d(model)

        # Merge blocks
        blocks.append(model_storage["blocks"])
        model_range.append(model_range[-1] + model_storage["blocks"].shape[0])

        # # Merge scale_factors
        # scale_factors.append(model_storage["scale_factors"])

        # Merge bias for all models
        for jth_weight, weight in model_storage["untouch_weights"].items():
            untouch_weights[f"{ith_model}_{jth_weight}"] = weight

        # if ith_model == 1:
        #     search_range = model_storage["search_range"]

    # Merge blocks into a single numpy array
    blocks = np.concatenate(blocks, axis=0)

    return {
        "blocks": blocks,  #  numpy array, shape: (n_all_blocks, block_size)
        "model_range": np.array(model_range),  #  numpy array of type int
        "untouch_weights": untouch_weights,  # dict: {f"{ith_model}_{jth_weight}": bias}
    }
    # # Save blocks, biases, scale_factors to blocks_path
    # np.savez(
    #     blocks_path,
    #     blocks=blocks,  #  numpy array, shape: (n_all_blocks, block_size)
    #     model_range=np.array(model_range),  #  numpy array of type int
    #     biases=biases,  # dict: {f"{ith_model}_{jth_weight}": bias}
    #     # search_range=search_range,  # numpy array of shape (n_all_blocks, 2)
    #     # scale_factors=np.array(scale_factors, dtype=object),  # list of numpy arrays
    #     # model_paths=np.array(model_paths),  #  numpy array of type str
    # )

    # # Load models from blocks_path
    # model_storage = np.load(blocks_path, allow_pickle=True)

    # return model_storage


def reconstruct_weight(model_storage, model, model_id, model_constitution: list[int]):
    """Reconstruct a model weights from a given model storage object."""
    model_start_point = model_storage["model_range"][model_id]

    reconstruct_weight_helper(
        model,
        model_storage["blocks"],
        model_start_point,
        model_constitution,
        untouched_weights=model_storage["untouch_weights"][model_id],
    )


def reconstruct_weight_helper(
    model,
    blocks,
    model_start_point,
    model_constitution,
    untouched_weights=None,
):
    start_idx = 0
    non_bias_idx = 0
    block_size = blocks.shape[1]

    for name, params in model.named_parameters():
        params.requires_grad_(False)
        if params.squeeze().dim() == 1 or params.numel() < block_size:
            if untouched_weights is not None:
                params.copy_(torch.from_numpy(untouched_weights[name]))
            continue
        # Reconstruct weights
        numel = params.numel()
        nblocks_for_params = math.ceil(numel / block_size)
        end_idx = start_idx + nblocks_for_params
        constitution_range = model_constitution[start_idx:end_idx]
        # Avoid reconstructing new weights if the model constitution is the same as the original one
        # `model_start_point` is the index of the first block of the model
        # `start_idx` is the index of the first block of the current layer
        start = model_start_point + start_idx
        if not all(
            (block_id == start + i for i, block_id in enumerate(constitution_range))
        ):
            new_weight = blocks[constitution_range].flatten()[:numel]
            # # Scale weights
            # new_weight = (
            #     new_weight * model_storage["scale_factors"][model_id][non_bias_idx]
            # )
            # Set parameter to new weight
            params.copy_(torch.from_numpy(new_weight.reshape(params.shape)))

        start_idx = end_idx
        non_bias_idx += 1


if __name__ == "__main__":
    model_storage = get_blocks()
    pdb.set_trace()
