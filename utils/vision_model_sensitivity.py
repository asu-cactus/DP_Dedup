import pdb
import torch
import torch.nn.functional as F

from utils.blocker import block_model_1d
from utils.parse_args import parse_args
from utils.common import load_model
from vision_task_utils.dataset import load_vision_dataset


def get_block_sensitivity(
    model_info, measure, skip_embeds=False, return_n_embed_blocks=False
):
    model_args, data_args, training_args = parse_args()
    data_args.task_name = model_info["task_name"]
    model_args.model_name_or_path = model_info["model_path"]
    # Get the model and dataset
    # dataset = load_vision_dataset("FGVCAircraft")
    dataset = load_vision_dataset(data_args.dataset_name)
    model = load_model(model_info, model_args)[0]

    block_size = model_args.block_size
    if measure == "magnitude":
        blocks = magnitute_sensitivity(model, block_size)
    elif measure == "fisher":
        if data_args.dataset_name == "CelebA":
            raise ValueError("Fisher sensitivity is not implemented for CelebA.")
        blocks = fisher_sensitity(model, dataset, block_size)
    elif measure == "gradient":
        blocks = gradient_sensitity(model, dataset, data_args.dataset_name, block_size)
    else:
        raise ValueError(f"Unknown sensitivity measure: {measure}")

    return blocks, None


def magnitute_sensitivity(model, block_size):
    model_storage = block_model_1d(block_size, model)
    blocks = model_storage["blocks"]
    return blocks


def fisher_sensitity(model, dataset, block_size, batch_size=16):
    model.eval()
    model.cuda()

    sample_size = len(dataset)

    testloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    for batch_idx, (inputs, labels) in enumerate(testloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        logits = model(inputs)
        probs = F.softmax(logits, dim=1)
        logprobs = F.log_softmax(logits, dim=1)

        probs = probs.gather(1, labels).squeeze()
        logprobs = logprobs.gather(1, labels).squeeze()

        # Initialize Fisher information matrix
        fim = {}
        no_grad_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                fim[name] = torch.zeros_like(param, device="cpu")
            else:
                no_grad_params[name] = param
        assert len(no_grad_params) == 0, "There are some parameters without grad."

        # Compute Fisher information
        for logprob, prob in zip(logprobs, probs):
            model.zero_grad()
            torch.autograd.backward(logprob, retain_graph=True)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    fim[name] += (param.grad.square() * prob).detach().cpu()

    fim = {k: grad2 / sample_size for k, grad2 in fim.items()}

    # Block the Fisher information corresponding to each parameter
    blocks = block_model_1d(block_size, fim)["blocks"]

    return blocks


def gradient_sensitity(
    model,
    dataset,
    dataset_name,
    block_size,
    batch_size=16,
    sample_size=None,
):
    model.eval()
    model.cuda()

    if sample_size is None:
        sample_size = len(dataset)
    else:
        sample_size = min(sample_size, len(dataset))
        dataset = dataset[:sample_size]

    testloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    if dataset_name == "CelebA":
        criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    else:
        criterion = torch.nn.CrossEntropyLoss()
    # accum_iter = sample_size / batch_size

    for inputs, targets in testloader:
        try:
            inputs, targets = inputs.cuda(), targets.cuda()
        except:
            pdb.set_trace()
        outputs = model(inputs)

        if dataset_name == "CelebA":
            loss = criterion(outputs, targets.float()).sum(dim=1).mean()
        else:
            loss = criterion(outputs, targets)
        # loss = loss / accum_iter
        loss.backward()

    grads = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            grads[name] = param.grad

    # Block the Gradients corresponding to each parameter
    blocks = block_model_1d(block_size, grads)["blocks"]

    return blocks
