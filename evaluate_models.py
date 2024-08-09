import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_DISABLED"] = "true"
from pathlib import Path
import time

from text_task_utils.evaluate import evaluate
from vision_task_utils.evaluate import evaluate
from utils.parse_args import parse_args
from utils.common import load_models_info


def text_task_evaluate():
    model_args, data_args, training_args = parse_args()
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
    models_info = load_models_info(model_args)
    for model_info in models_info:
        tic = time.time()
        data_args.task_name = model_info["task_name"]
        print(f"Model: {model_info['model_path']}")
        model_args.model_name_or_path = model_info["model_path"]
        acc = evaluate(data_args, model_args, training_args, model_info)
        print(f"Accuracy: {acc}")
        print(f"Time taken: {time.time() - tic}")


def vision_task_evaluation():
    model_args, data_args, training_args = parse_args()
    models_info = load_models_info(model_args)
    # model_paths = [info["model_path"] for info in models_info]

    for model_info in models_info:
        tic = time.time()
        acc = evaluate(
            data_args,
            model_args,
            training_args,
            model_info,
        )
        print(f"Accuracy: {acc}")
        print(f"Time taken: {time.time() - tic}")


if __name__ == "__main__":

    text_task_evaluate()
    # vision_task_evaluation()
    # vision_task_parameters()
