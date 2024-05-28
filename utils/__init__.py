import json


def load_models_info(task_type) -> list[dict]:
    """
    Load model information from model_info.json.
    If model_ids is not specified, load all models. Otherwise, load the specified models.
    """
    if task_type == "text":
        model_info_path = "models/model_info_text.json"
    elif task_type == "vision_vit":
        model_info_path = "models/model_info_vision_vit.json"
    elif task_type == "vision_resnet":
        model_info_path = "models/model_info_vision_resnet.json"
    elif task_type == "recommendation":
        model_info_path = "models/model_info_recommendation.json"
    else:
        raise ValueError(f"Invalid task type: {task_type}")

    with open(model_info_path, "r") as f:
        models_info = json.load(f)
    models_info = list(models_info.values())
    for info in models_info:
        print(info)
    return models_info
