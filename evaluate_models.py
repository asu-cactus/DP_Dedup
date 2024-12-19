import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WANDB_DISABLED"] = "true"
from pathlib import Path
import time

from text_task_utils.evaluate import evaluate

# from vision_task_utils.evaluate import evaluate
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


def compute_sensitivity():
    from utils.text_model_sensitivity import get_block_sensitivity as text_sensitivity
    from utils.vision_model_sensitivity import (
        get_block_sensitivity as vision_sensitivity,
    )
    import numpy as np

    model_args, data_args, training_args = parse_args()
    models_info = load_models_info(model_args)
    model_info1 = models_info[0]

    if model_info1["task_name"] in ("qnli", "mnli", "sst-2"):
        sensitivity1, _ = text_sensitivity(model_info1, "gradient")
    else:
        sensitivity1, _ = vision_sensitivity(model_info1, "gradient")

    # Concatenate the sensitivity
    all_params = []
    for name, params in sensitivity1.items():
        params = params.detach().cpu().flatten().numpy()
        all_params.append(params)

    all_params = np.concatenate(all_params)

    # Save the sensitivity
    print(f"shape of all_params: {all_params.shape}")
    np.save("cifar100_sensitivity.npy", all_params)


def compute_disparity():
    from utils.text_model_sensitivity import get_block_sensitivity as text_sensitivity
    from utils.vision_model_sensitivity import (
        get_block_sensitivity as vision_sensitivity,
    )
    import numpy as np

    model_args, data_args, training_args = parse_args()
    models_info = load_models_info(model_args)

    model_info1 = models_info[0]
    if model_info1["task_name"] in ("qnli", "mnli", "sst-2"):
        sens_blocks1, model_blocks1 = text_sensitivity(model_info1, "gradient")
    else:
        sens_blocks1, model_blocks1 = vision_sensitivity(model_info1, "gradient")

    model_info2 = models_info[1]
    if model_info2["task_name"] in ("qnli", "mnli", "sst-2"):
        sens_blocks2, model_blocks2 = text_sensitivity(model_info2, "gradient")
    else:
        sens_blocks2, model_blocks2 = vision_sensitivity(model_info2, "gradient")

    scores = np.empty((len(model_blocks1), len(model_blocks2)))

    for i, (mblock1, sblock1) in enumerate(zip(model_blocks1, sens_blocks1)):
        for j, (mblock2, sblock2) in enumerate(zip(model_blocks2, sens_blocks2)):
            scores[i, j] = np.dot((mblock1 - mblock2) ** 2, np.abs(sblock1 - sblock2))

    # Save the scores
    print(scores)
    np.save("qnli_sst2_scores.npy", scores)


if __name__ == "__main__":

    # text_task_evaluate()
    # vision_task_evaluation()
    # vision_task_parameters()

    # compute_sensitivity()
    compute_disparity()
