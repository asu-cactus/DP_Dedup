import pdb

from torch import nn
import torch.nn.utils.prune as prune
from scipy.sparse import csr_matrix


def prune_model(model):
    parameters_to_prune = []
    for name, m in model.named_modules():
        if not isinstance(m, nn.Sequential) and (
            "conv" in name or "bn" in name or "downsample" in name
        ):
            parameters_to_prune.append((m, "weight"))

    # pdb.set_trace()
    prune.global_unstructured(
        tuple(parameters_to_prune),
        pruning_method=prune.L1Unstructured,
        amount=0.9,
    )

    for name, m in model.named_modules():
        if not isinstance(m, nn.Sequential) and (
            "conv" in name or "bn" in name or "downsample" in name
        ):
            prune.remove(m, "weight")
    return model


def sparsify_model_storage(model_storage):
    # csr_matrix is the best among (csr_matrix, csc_matrix, coo_matrix)
    model_storage["blocks"] = csr_matrix(model_storage["blocks"])

    # Sparsify the untouched weights
    for name, params in model_storage["untouched_weights"].items():
        if "conv" in name or "bn" in name or "downsample" in name:
            model_storage["untouched_weights"][name] = csr_matrix(params.reshape(1, -1))
