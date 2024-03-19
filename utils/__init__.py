import json


def load_models_info() -> list[dict]:
    """
    Load model information from model_info.json.
    If model_ids is not specified, load all models. Otherwise, load the specified models.
    """
    with open("models/model_info.json", "r") as f:
        models_info = json.load(f)
    models_info = list(models_info.values())
    for info in models_info:
        print(info)
    return models_info
