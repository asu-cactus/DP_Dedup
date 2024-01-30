import os
import json
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
random.seed(10)

from parse_args import parse_args
from mcts import MCTS
from blocker import get_blocks


def load_models_info(model_args) -> list[dict]:
    with open("models/model_info.json", "r") as f:
        models_info = json.load(f)

    model_ids = model_args.model_ids.split(",")
    models_info = [models_info[idx] for idx in model_ids]
    return models_info
    # acc_drop_thresholds = [info["acc_drop_threshold"] for info in models_info]
    # original_accs = [info["original_acc"] for info in models_info]
    # budgets = [info["budget"] for info in models_info]
    # model_paths = [info["model_path"] for info in models_info]
    # return (acc_drop_thresholds, original_accs, budgets, model_paths)


def main():
    model_args, data_args, training_args = parse_args()
    # acc_drop_thresholds, original_accs, budgets, model_paths = load_model_info()
    models_info = load_models_info(model_args)
    model_paths = [info["model_path"] for info in models_info]
    models_storage = get_blocks(model_paths=model_paths)
    mcts = MCTS(
        model_args,
        data_args,
        training_args,
        models_info,
        models_storage,
    )

    max_v = -float("inf")
    original_num_blocks = models_storage["model_range"][-1]
    print(f"Original total number of blocks: {original_num_blocks}")

    start_episode = training_args.resume_episode if training_args.resume else 0
    for i in range(start_episode + 1, start_episode + training_args.n_episodes + 1):
        v = mcts.search(mcts.init_state)
        n_distinct_blocks = -v * original_num_blocks
        print(f"Episode {i} return value: {n_distinct_blocks}\n")
        if v > max_v:
            max_v = v
            with open(f"{training_args.output_dir}/best_value.txt", "a") as f:
                f.write(f"Episode {i}: {n_distinct_blocks}\n")
        if i % training_args.save_every == 0:
            save_i = i
            delete_i = i - training_args.save_every * training_args.keep_n
            mcts.save_state(save_i, delete_i)


if __name__ == "__main__":
    main()
