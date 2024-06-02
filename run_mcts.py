import os
import pdb

os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TQDM_DISABLE"] = "1"

import numpy as np

from utils.parse_args import parse_args
from utils.blocker import block_model_1d
from utils.common import (
    load_models_info,
    load_model,
    merge_model_storage,
    compute_compression_ratio,
    set_model_args,
)
from mcts.fix_second_v2.mcts import MCTS


def order_blocks(model_args, training_args, model_info, model_constitution):
    if model_args.task_type == "text":
        from utils.text_model_sensitivity import get_block_sensitivity as sensitivity_fn
    elif "vision" in model_args.task_type:
        from utils.vision_model_sensitivity import (
            get_block_sensitivity as sensitivity_fn,
        )
    elif model_args.task_type == "recommendation":
        from utils.recommender_sensitivity import (
            get_block_sensitivity as sensitivity_fn,
        )
    else:
        raise ValueError(f"Invalid task name: {model_args.task_type}")

    return_n_embed_blocks = training_args.sensitivity_measure == "wanda"
    measures, n_embed_blocks = sensitivity_fn(
        model_info,
        training_args.sensitivity_measure,
        skip_embeds=False,
        return_n_embed_blocks=return_n_embed_blocks,
    )
    if training_args.orderby == "l1_norm":
        block_sens = np.sum(np.abs(measures), axis=1)
    elif training_args.orderby == "l2_norm":
        block_sens = np.linalg.norm(measures, axis=1)
    elif training_args.orderby == "l_inf_norm":
        block_sens = np.max(np.abs(measures), axis=1)
    elif training_args.orderby == "3rd_quantile":
        block_sens = np.quantile(measures, 0.75, axis=1)
    else:
        raise ValueError(f"Invalid orderby: {training_args.orderby}")

    # Because Wanda only applies to linear layers, we deduplicate them at last
    ordered_indices = np.argsort(block_sens)
    if training_args.sensitivity_measure == "wanda":
        embed_seq = model_constitution[:n_embed_blocks]
        other_seq = model_constitution[n_embed_blocks:]
        other_seq = other_seq[ordered_indices]
        model_constitution = np.concatenate((embed_seq, other_seq))
    else:
        model_constitution = model_constitution[ordered_indices]
    return model_constitution


def create_action_space(model_args, training_args, model_info, models_storage):
    model_start = models_storage["model_range"][1]
    model_end = models_storage["model_range"][2]
    model_constitution = np.arange(model_start, model_end)
    assert len(model_constitution) == models_storage["blocks"].shape[0] // 2

    model_constitution = order_blocks(
        model_args, training_args, model_info, model_constitution
    )

    # Group model_constitution into groups of size group_size
    group_size = training_args.every_n
    action_space = {}
    for i in range(0, len(model_constitution), group_size):
        action_space[i] = model_constitution[i : i + group_size]
    return action_space


def run():
    model_args, data_args, training_args = parse_args()
    models_info = load_models_info(model_args.task_type)
    task_type = model_args.task_type
    output_dir = training_args.output_dir

    # if task_type == "text":
    #     from text_task_utils.evaluate import evaluate as eval_fn
    # elif task_type.startswith("vision"):
    #     from vision_task_utils.evaluate import evaluate as eval_fn
    #     from vision_task_utils.train import train as train_fn
    # elif task_type == "recommendation":
    #     from recommendation_task_utils.evaluate import evaluate as eval_fn
    #     from recommendation_task_utils.train import train as train_fn
    # else:
    #     raise ValueError(f"Unknown task type: {task_type}")

    base_model, eval_fn, train_fn, _ = load_model(models_info[0], model_args)
    base_model_storage = block_model_1d(model_args.block_size, base_model)
    # n_base_blocks = base_model_storage["blocks"].shape[0]
    set_model_args(model_args, model, base_model_storage)

    total_new_blocks = 0
    # blockss_from_base = set()
    for model_info in models_info[1:]:
        print(f"Model info: {model_info}")
        model = load_model(model_info, model_args)[0]
        curr_model_storage = block_model_1d(model_args.block_size, model)
        models_storage = merge_model_storage(base_model_storage, curr_model_storage)
        action_space = create_action_space(
            model_args, training_args, model_info, models_storage
        )
        mcts = MCTS(
            model_args,
            data_args,
            training_args,
            model_info,
            models_storage,
            eval_fn,
            train_fn,
            action_space,
        )

        max_v = 0
        original_num_blocks = curr_model_storage["blocks"].shape[0]

        for i in range(1, training_args.n_episodes + 1):
            print(f"Start episode {i}")
            init_state = mcts.initial_episode()
            v = mcts.search(state=init_state)
            n_dedup_blocks = round(v * original_num_blocks)
            print(f"Episode {i} number of dedup blocks: {n_dedup_blocks}\n")

            if v > max_v:
                max_v = v
                with open(f"{output_dir}/best_value_{task_type}.txt", "a") as f:
                    f.write(f"Episode {i}: {n_dedup_blocks}\n")
            if v == 1:  # all blocks can be deduplicated
                break

        n_new_blocks = original_num_blocks - round(max_v * original_num_blocks)
        print(f"{model_info['model_path']} Number of new blocks: {n_new_blocks}\n")
        total_new_blocks += n_new_blocks
    print(f"\n{total_new_blocks=}")
    cr = compute_compression_ratio(
        total_new_blocks,
        model_args.block_size,
        model_args.untouched_weights,
        model_args.n_original_weights,
    )
    print(f"Compression ratio: {cr}")


if __name__ == "__main__":
    run()
