import json


def load_models_info(model_args) -> list[dict]:
    """
    Load model information from model_info.json.
    If model_ids is not specified, load all models. Otherwise, load the specified models.
    """
    # TODO: make sure the models_info is ordered by budget, for now, it is done by manually order them in the json file
    with open("models/model_info.json", "r") as f:
        models_info = json.load(f)
    if not model_args.model_ids:
        models_info = list(models_info.values())
    else:
        model_ids = model_args.model_ids.split(",")
        models_info = [models_info[idx] for idx in model_ids]
    return models_info
