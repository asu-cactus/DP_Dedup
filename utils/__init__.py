import json


def load_models_info(task_type) -> list[dict]:
    """
    Load model information from model_info.json.
    If model_ids is not specified, load all models. Otherwise, load the specified models.
    """
    model_info_path = (
        "models/model_info_text.json"
        if task_type == "text"
        else "models/model_info_vision.json"
    )
    with open(model_info_path, "r") as f:
        models_info = json.load(f)
    models_info = list(models_info.values())
    for info in models_info:
        print(info)
    return models_info
