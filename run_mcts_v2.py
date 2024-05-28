import pdb

import numpy as np

from utils.parse_args import parse_args
from utils.blocker import block_model_1d
from utils import load_models_info
from utils.common import load_model, merge_model_storage, separate_blocks
from mcts.fix_second.mcts import MCTS


def run():
    model_args, data_args, training_args = parse_args()
    models_info = load_models_info(model_args.task_type)
    # model_paths = [info["model_path"] for info in models_info]

    base_model = load_model(models_info[0], model_args)[0]
    base_model_storage = block_model_1d(base_model)
    n_base_blocks = base_model_storage["blocks"].shape[0]

    total_new_blocks = 0
    blockss_from_base = set()
    for model_info in models_info[1:]:
        print(f"Model info: {model_info}")
        model, eval_fn, train_fn, sensitivity_fn = load_model(model_info, model_args)
        curr_model_storage = block_model_1d(model)
        model_storage = merge_model_storage(base_model_storage, curr_model_storage)

        mcts = MCTS(
            model_args,
            data_args,
            training_args,
            model,
            model_info,
            model_storage,
            eval_fn,
            train_fn,
            sensitivity_fn,
        )

        max_v = 0
        original_num_blocks = curr_model_storage["blocks"].shape[0]

        start_episode = training_args.resume_episode if training_args.resume else 0
        for i in range(start_episode + 1, start_episode + training_args.n_episodes + 1):
            init_state = mcts.initial_episode()

            v, steps2fail = mcts.search(
                state=init_state,
                steps_before_eval=training_args.eval_every - 1,
            )
            # print(f"steps2fail: {steps2fail}")
            if steps2fail == training_args.eval_every - 1:
                n_dedup_blocks = 0
            else:
                n_dedup_blocks = round(v * original_num_blocks)
            print(f"Episode {i} number of dedup blocks: {n_dedup_blocks}\n")

            if v > max_v:
                max_v = v
                with open(f"{training_args.output_dir}/best_value.txt", "a") as f:
                    f.write(f"Episode {i}: {n_dedup_blocks}\n")

        n_new_blocks = original_num_blocks - round(max_v * original_num_blocks)
        print(f"{model_info['model_path']} Number of new blocks: {n_new_blocks}")
        total_new_blocks += n_new_blocks
    print(f"\n{total_new_blocks=}")
