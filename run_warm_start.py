import os
import sys
import random
from pathlib import Path

from final_constitution_quantile import model0, model1
from mcts.warm_start_naive.mcts import MCTS

# maximum number of blocks that can be deduplicated
sys.setrecursionlimit(5000)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WANDB_DISABLED"] = "true"
random.seed(10)
print(f"{os.environ['CUDA_VISIBLE_DEVICES']=}")

from utils.parse_args import parse_args
from utils.blocker import get_blocks
from utils import load_models_info


def main():
    model_args, data_args, training_args = parse_args()
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
    models_info = load_models_info(model_args)
    model_paths = [info["model_path"] for info in models_info]
    models_storage = get_blocks(model_paths=model_paths)

    max_v = 0
    model_range = models_storage["model_range"]

    for model_id, model_constitute in enumerate([model0, model1]):
        original_num_blocks = model_range[model_id + 1] - model_range[model_id]
        mcts = MCTS(
            model_id,
            model_args,
            data_args,
            training_args,
            models_info,
            models_storage,
        )
        for i in range(1, training_args.n_episodes + 1):
            init_state = mcts.initial_episode(model_constitute)
            v, model_constitution = mcts.search(init_state, False)
            n_dedup_blocks = round(v * original_num_blocks)

            print(
                f"Model{model_id} Episode {i} number of dedup blocks: {n_dedup_blocks}\n"
            )

            if v > max_v:
                max_v = v
                with open(f"{training_args.output_dir}/best_value.txt", "a") as f:
                    f.write(f"Model{model_id} Episode {i}: {n_dedup_blocks}\n")
                    f.write(f"Model constitution: {model_constitution}\n")
            if i % training_args.save_every == 0:
                mcts.save_state(i)


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
