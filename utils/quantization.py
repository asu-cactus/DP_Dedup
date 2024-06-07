import numpy as np

import torch
import pdb


def quant_layer(params, quant_stat=None):
    # quant
    # weight = layer.get_weights()
    params.requires_grad_(False)
    if quant_stat is None:
        # Make sure if weights are integers
        min_val = torch.min(params)
        max_val = torch.max(params)
        quant_stat = (min_val, max_val)
        step = (max_val - min_val) / 256
        new_params = torch.div(params - min_val, step, rounding_mode="floor")
        params.copy_(new_params)
    else:
        step = (quant_stat[1] - quant_stat[0]) / 256
        new_params = torch.div(params - quant_stat[0], step, rounding_mode="floor")
        new_params = torch.clamp(new_params, 0, 255)
        params.copy_(new_params)
    return quant_stat


def quant_model(model, quant_data=None):
    if quant_data is None:
        quant_data = []
        for params in model.parameters():
            quant_stat = quant_layer(params)
            quant_data.append(quant_stat)
    else:
        for params, quant_stat in zip(model.parameters(), quant_data):
            quant_layer(params, quant_stat)
    return model, quant_data


def dequant_layer(params, quant_stat):
    params.requires_grad_(False)
    # weight = torch.empty_like(params)
    # for idx, (min_val, max_val) in enumerate(quant_stat):
    #     step = (max_val - min_val) / 256
    #     weight[idx] = (params[idx] * step) + min_val
    step = (quant_stat[1] - quant_stat[0]) / 256
    new_params = (params * step) + quant_stat[0]
    params.copy_(new_params)


def dequant_model(model, quant_data):
    for params, quant_stat in zip(model.parameters(), quant_data):
        dequant_layer(params, quant_stat)
    return model


def quant_and_dequant_model(model, quant_data=None):
    q_model, quant_data = quant_model(model, quant_data)
    deq_model = dequant_model(q_model, quant_data)
    return deq_model, quant_data, q_model
