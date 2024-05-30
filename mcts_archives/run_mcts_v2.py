import os

os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TQDM_DISABLE"] = "1"

import pdb

from utils.parse_args import parse_args
from utils.blocker import block_model_1d
from utils import load_models_info
from utils.common import load_model, merge_model_storage, separate_blocks
from mcts.fix_second.mcts import MCTS


def run():
    model_args, data_args, training_args = parse_args()
    models_info = load_models_info(model_args.task_type)
    # model_paths = [info["model_path"] for info in models_info]

    if model_args.task_type == "text":
        from text_task_utils.evaluate import evaluate as eval_fn

    elif model_args.task_type.startswith("vision"):
        from vision_task_utils.evaluate import evaluate as eval_fn
        from vision_task_utils.train import train as train_fn
    elif model_args.task_type == "recommendation":
        from recommendation_task_utils.evaluate import evaluate as eval_fn
        from recommendation_task_utils.train import train as train_fn
    else:
        raise ValueError(f"Unknown task type: {model_args.task_type}")

    base_model = load_model(models_info[0], model_args)[0]
    base_model_storage = block_model_1d(base_model)
    # n_base_blocks = base_model_storage["blocks"].shape[0]

    total_new_blocks = 0
    # blockss_from_base = set()
    for model_info in models_info[1:]:
        print(f"Model info: {model_info}")
        model = load_model(model_info, model_args)[0]
        curr_model_storage = block_model_1d(model)
        models_storage = merge_model_storage(base_model_storage, curr_model_storage)

        mcts = MCTS(
            model_args,
            data_args,
            training_args,
            # model,
            model_info,
            models_storage,
            eval_fn,
            train_fn,
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
                if v == -1:
                    v = 1
                n_dedup_blocks = round(v * original_num_blocks)
            print(f"Episode {i} number of dedup blocks: {n_dedup_blocks}\n")

            if v > max_v:
                max_v = v
                with open(f"{training_args.output_dir}/best_value.txt", "a") as f:
                    f.write(f"Episode {i}: {n_dedup_blocks}\n")
            if v == 1:  # all blocks can be deduplicated
                break

        n_new_blocks = original_num_blocks - round(max_v * original_num_blocks)
        print(f"{model_info['model_path']} Number of new blocks: {n_new_blocks}\n")
        total_new_blocks += n_new_blocks
    print(f"\n{total_new_blocks=}")


if __name__ == "__main__":
    run()
