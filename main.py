import os
import sys
import random
from pathlib import Path

# maximum number of blocks that can be deduplicated
sys.setrecursionlimit(5000)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_DISABLED"] = "true"
random.seed(10)


from utils.parse_args import parse_args
from utils.blocker import get_blocks
from utils import load_models_info


def main():
    model_args, data_args, training_args = parse_args()
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
    models_info = load_models_info(model_args)
    model_paths = [info["model_path"] for info in models_info]
    models_storage = get_blocks(model_paths=model_paths)
    if training_args.mcts_mode == "uct_mcts":
        from mcts.mcts import MCTS
    else:
        from mcts.heuristic_mc_rave import MCTS
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
        v = mcts.search(mcts.init_state, False)
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
    # from mcts.heuristics import get_heuristics_dict

    # model_args, data_args, training_args = parse_args()
    # Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
    # models_info = load_models_info(model_args)
    # model_paths = [info["model_path"] for info in models_info]
    # models_storage = get_blocks(model_paths=model_paths)
    # get_heuristics_dict(
    #     model_args,
    #     data_args,
    #     training_args,
    #     models_info,
    #     models_storage,
    # )

    # # Load pickle fiel heuristics_dict.pkl
    # import pickle
    # import pdb

    # with open("heuristics_dict.pkl", "rb") as f:
    #     heuristics_dict = pickle.load(f)
    # pdb.set_trace()
