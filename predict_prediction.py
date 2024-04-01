from collections import defaultdict
import pdb
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

from sensitivity_measure import get_model_and_dateset, fisher_sensitity
from utils.blocker import block_model_1d, reconstruct_weight_helper
from utils.parse_args import parse_args
from utils import load_models_info


def run():
    models_info = load_models_info()
    deduplicate(models_info[0])


def deduplicate(model_info):

    model_args, data_args, training_args = parse_args()
    data_args.task_name = model_info["task_name"]
    model_args.model_name_or_path = model_info["model_path"]
    # Get the model and dataset
    model, dataset = get_model_and_dateset(
        data_args,
        model_args,
        training_args,
    )

    dedup_indices = set()
    dedup_dict = defaultdict(list)
    predict_prediction(model, dataset, dedup_indices, dedup_dict, sample_size=16)


def predict_prediction(
    model,
    dataset,
    dedup_indices,
    dedup_dict,
    batch_size=16,
    sample_size=None,
    param_delta_threshold=0.01,
    prob_delta_threshold=0.01,
):
    model_blocks = block_model_1d(model)["blocks"]
    if Path("ordered_indices.npy").exists():
        ordered_indices = np.load("ordered_indices.npy")
    else:
        # measure = model_blocks
        measure = fisher_sensitity(model, dataset, skip_embeds=False)
        block_sensitivity = np.linalg.norm(measure, axis=1)
        ordered_indices = np.argsort(block_sensitivity)
        np.save("ordered_indices.npy", ordered_indices)

    model.eval()
    model.cuda()

    if sample_size is None:
        sample_size = len(dataset)

    for start_id in range(0, sample_size, batch_size):
        end_idx = min(start_id + batch_size, sample_size)

        input_ids = torch.tensor(
            [[dataset[i].input_ids for i in range(start_id, end_idx)]], dtype=torch.long
        )
        input_ids = input_ids.squeeze().cuda()

        attention_mask = torch.tensor(
            [[dataset[i].attention_mask for i in range(start_id, end_idx)]],
            dtype=torch.long,
        )
        attention_mask = attention_mask.squeeze().cuda()

        mask_pos = torch.tensor(
            [[dataset[i].mask_pos for i in range(start_id, end_idx)]], dtype=torch.long
        )
        mask_pos = mask_pos.squeeze().cuda()

        labels = torch.tensor(
            [dataset[i].label for i in range(start_id, end_idx)], dtype=torch.long
        )
        labels = labels.unsqueeze(1).cuda()

        # Get logits and probs
        logits = model(input_ids, attention_mask, mask_pos)[0]
        probs = F.softmax(logits, dim=1).gather(1, labels).squeeze()
        print(f"{logits=}")
        print(f"{labels=}")
        print(f"{probs=}")

        # Compute gradients per sample
        grad_blockss = []
        for prob in probs:
            model.zero_grad()
            torch.autograd.backward(prob, retain_graph=True)

            grads = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    grads[name] = param.grad.detach()

            grad_blocks = block_model_1d(grads)["blocks"]
            grad_blockss.append(grad_blocks)
        probs = probs.detach().cpu().numpy()
        original_probs = probs.copy()

        # Start deduplication
        # Iterate blocks to be replaced
        for i in ordered_indices:
            block_2b_replaced = model_blocks[i]
            diff = np.linalg.norm(model_blocks - block_2b_replaced, axis=1)
            # ind = diff.argsort()[:3]  # TODO: More efficient implementation
            ind = np.argpartition(diff, 10)[:10]
            ind = ind[np.argsort(diff[ind])]

            probs_copy = probs.copy()
            # Iterate blocks to replace
            for j in ind:  # j is the index of the block to replace
                if j == i or j in dedup_indices:
                    continue

                param_delta = model_blocks[i] - model_blocks[j]
                max_param_delta = np.max(np.abs(param_delta))

                # Break if all gradients are zero
                grad_all_zero = True
                for k, grad_blocks in enumerate(grad_blockss):
                    if not (grad_blocks[i] == 0.0).all():
                        grad_all_zero = False
                        break

                if not grad_all_zero:
                    # if max_param_delta > 0.01:
                    #     print(f"{i=} {max_param_delta=}")
                    #     break

                    # delta_too_big = False
                    mul_too_big = False
                    probs_copy2 = probs_copy.copy()
                    for k, grad_blocks in enumerate(grad_blockss):
                        # TODO: optimize the following line because index for grad_blocks is i.
                        # prob_delta = np.dot(param_delta, grad_blocks[i])
                        # probs_copy[k] += prob_delta

                        # Want to check np.max(param_delta * grad_blocks[i]) to see if it is too big
                        mul = param_delta * grad_blocks[i]
                        max_mul = np.max(mul)
                        if max_mul > 0.00001:
                            print(f"{max_mul=}")
                            mul_too_big = True
                            break
                        probs_copy2[k] += np.sum(mul)

                        # if np.abs(prob_delta) > 0.00001:
                        #     print(f"{prob_delta=}")
                        #     delta_too_big = True
                        #     break
                        # probs_copy2[k] += prob_delta

                    # if delta_too_big:
                    #     break
                    if mul_too_big:
                        break
                    probs_copy = probs_copy2

                if ((original_probs < 0.5) | (probs_copy > 0.5)).all():
                    dedup_indices.add(i)
                    dedup_dict[j].append(i)
                    if i in dedup_dict:
                        dedup_dict[j].extend(dedup_dict[i])
                        del dedup_dict[i]
                    probs = probs_copy
                break

            # Change only one block that the gradient is not zero
            if not grad_all_zero:
                break

        print(f"Number of deduplications: {len(dedup_indices)}")

        # Get the model constitution
        model_constitution = list(range(len(model_blocks)))
        for block_to_replace, blocks_2b_replaced in dedup_dict.items():
            for block_2b_replaced in blocks_2b_replaced:
                model_constitution[block_2b_replaced] = block_to_replace

        reconstruct_weight_helper(model, model_blocks, 0, model_constitution)

        # Get logits and probs
        logits = model(input_ids, attention_mask, mask_pos)[0]
        new_probs = F.softmax(logits, dim=1).gather(1, labels).squeeze()
        print(f"{new_probs=}")
        print(f"{probs=}")
        print(f"{original_probs=}")
        pdb.set_trace()


run()
