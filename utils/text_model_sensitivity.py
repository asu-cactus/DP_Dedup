import pdb
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
)
from transformers import GlueDataset

from utils.blocker import block_model_1d
from utils.parse_args import (
    parse_args,
    ModelArguments,
    DynamicDataTrainingArguments,
    DynamicTrainingArguments,
)
from text_task_utils.dataset import FewShotDataset
from text_task_utils.models import (
    BertForPromptFinetuning,
    RobertaForPromptFinetuning,
    AlbertForPromptFinetuning,
    DistilBertForPromptFinetuning,
    resize_token_type_embeddings,
)
from text_task_utils.processors import (
    num_labels_mapping,
    output_modes_mapping,
    compute_metrics_mapping,
    bound_mapping,
)
from text_task_utils import common

import json


def load_final_constitution(exp_name: str = "l1_exp"):
    with open(f"final_constitution/{exp_name}.json") as f:
        constitution = json.load(f)
    return constitution


def count_input_ids2(dataset):
    from collections import defaultdict

    counter = defaultdict(int)

    for features in dataset:
        input_ids = set(features.input_ids)
        input_ids = set([input_ids // 768 for input_ids in input_ids])

        for input_id in input_ids:
            counter[input_id] += 1

    counter = dict(sorted(counter.items()))
    print(f"counters: \n{counter}")
    pdb.set_trace()


def count_input_ids(dataset, count_sentence=True):
    from collections import defaultdict

    counter = defaultdict(int)

    for features in dataset:
        if count_sentence:
            input_ids = set(features.input_ids)
        else:
            input_ids = features.input_ids
        for input_id in input_ids:
            counter[input_id] += 1

    counter = dict(counter)
    print(f"max key is {max(counter, key=int)}")
    print(f"min key is {min(counter, key=int)}")

    counter2 = defaultdict(int)
    for i in range(66):
        for j in range(768):
            input_id = i * 768 + j
            counter2[i] += counter.get(input_id, 0)
    counter2 = dict(counter2)
    print(f"counter2: \n{counter2}")
    pdb.set_trace()


def get_model_and_dateset(
    data_args: DynamicDataTrainingArguments,
    model_args: ModelArguments,
    training_args: DynamicTrainingArguments,
):
    # Add some additional arguments to make it work
    # original_task_name = data_args.task_name
    # if data_args.task_name == "mnli":
    #     data_args.task_name = "qnli"
    # elif data_args.task_name == "qnli":
    #     data_args.task_name = "sst-2"

    data_args.data_dir = (
        f"{data_args.data_root_dir}/{common.task_name2suffix_name[data_args.task_name]}"
    )

    training_args.local_rank = -1

    data_args.template = {
        "sst-2": "*cls**sent_0*_It_was*mask*.*sep+*",
        "mnli": "*cls**sent-_0*?*mask*,*+sentl_1**sep+*",
        "qnli": "*cls**sent-_0*?*mask*,*+sentl_1**sep+*",
        "qqp": "*cls**sent-_0**mask*,*+sentl_1**sep+*",
    }[data_args.task_name]

    # Work with some arguments
    if "prompt" in model_args.few_shot_type:
        data_args.prompt = True

    # TODO: Hacky mapping creation. Refactor this in the future.
    #  Currently gets replace if mapping_id and mapping_path is set.
    if data_args.task_name == "sst-2":
        data_args.mapping = "{'0':'terrible','1':'great'}"
    elif data_args.task_name == "mnli":
        data_args.mapping = (
            "{'contradiction': 'no', 'entailment': 'yes', 'neutral': 'maybe'}"
        )
    elif data_args.task_name == "qnli":
        data_args.mapping = "{'not_entailment': 'no', 'entailment': 'yes'}"
    elif data_args.task_name == "qqp":
        data_args.mapping = (
            "{'1': 'yes', '0': 'no'}"  # 1 -- equivalent, 0 -- not equivalent.
        )
    else:
        raise ValueError(f"Unknown task: {data_args.task_name}")

    # Load prompt/template/mapping file
    if data_args.prompt:
        if data_args.prompt_path is not None:
            assert data_args.prompt_id is not None
            prompt_list = []
            with open(data_args.prompt_path) as f:
                for line in f:
                    line = line.strip()
                    template, mapping = line.split("\t")
                    prompt_list.append((template, mapping))

            data_args.template, data_args.mapping = prompt_list[data_args.prompt_id]

        else:
            if data_args.template_path is not None:
                with open(data_args.template_path) as f:
                    data_args.template_list = []
                    for line in f:
                        line = line.strip()
                        if len(line) > 0:
                            data_args.template_list.append(line)

                # Load top-n templates
                if data_args.top_n_template is not None:
                    data_args.template_list = data_args.template_list[
                        : data_args.top_n_template
                    ]

                # ... or load i-th template
                if data_args.template_id is not None:
                    data_args.template = data_args.template_list[data_args.template_id]
                    data_args.template_list = None

            if data_args.mapping_path is not None:
                assert (
                    data_args.mapping_id is not None
                )  # Only can use one label word mapping
                with open(data_args.mapping_path) as f:
                    mapping_list = []
                    for line in f:
                        line = line.strip()
                        mapping_list.append(line)

                data_args.mapping = mapping_list[data_args.mapping_id]

    num_labels = num_labels_mapping[data_args.task_name]

    # Automatically generate template for using demonstrations
    if data_args.auto_demo and model_args.few_shot_type == "prompt-demo":
        # GPT-3's in-context learning
        if data_args.gpt3_in_context_head or data_args.gpt3_in_context_tail:
            assert data_args.template_list is None

            old_template = data_args.template
            new_template = old_template + ""
            old_template = old_template.replace("*cls*", "")
            # Single sentence or sentence pair?
            sent_num = 1
            if "_1" in old_template:
                sent_num = 2
            for instance_id in range(data_args.gpt3_in_context_num):
                sub_template = old_template + ""
                # Replace sent_id
                for sent_id in range(sent_num):
                    sub_template = sub_template.replace(
                        "_{}*".format(sent_id),
                        "_{}*".format(sent_num + sent_num * instance_id + sent_id),
                    )
                # Replace mask
                sub_template = sub_template.replace(
                    "*mask*", "*labelx_{}*".format(instance_id)
                )
                if data_args.gpt3_in_context_tail:
                    new_template = new_template + sub_template  # Put context at the end
                else:
                    new_template = (
                        sub_template + new_template
                    )  # Put context at the beginning
            # logger.info("| {} => {}".format(data_args.template, new_template))
            data_args.template = new_template
        else:
            # logger.info("Automatically convert the template to using demonstrations.")
            if data_args.template_list is not None:
                for i in range(len(data_args.template_list)):
                    old_template = data_args.template_list[i]
                    new_template = old_template + ""
                    old_template = old_template.replace("*cls*", "")
                    # Single sentence or sentence pair?
                    sent_num = 1
                    if "_1" in old_template:
                        sent_num = 2
                    for label_id in range(num_labels):
                        sub_template = old_template + ""
                        # Replace sent id
                        for sent_id in range(sent_num):
                            sub_template = sub_template.replace(
                                "_{}*".format(sent_id),
                                "_{}*".format(sent_num + sent_num * label_id + sent_id),
                            )
                        # Replace mask
                        sub_template = sub_template.replace(
                            "*mask*", "*label_{}*".format(label_id)
                        )
                        new_template = new_template + sub_template
                    # logger.info(
                    #     "| {} => {}".format(data_args.template_list[i], new_template)
                    # )
                    data_args.template_list[i] = new_template
            else:
                old_template = data_args.template
                new_template = old_template + ""
                old_template = old_template.replace("*cls*", "")
                # Single sentence or sentence pair?
                sent_num = 1
                if "_1" in old_template:
                    sent_num = 2
                for label_id in range(num_labels):
                    sub_template = old_template + ""
                    # Replace sent id
                    for sent_id in range(sent_num):
                        sub_template = sub_template.replace(
                            "_{}".format(sent_id),
                            "_{}".format(sent_num + sent_num * label_id + sent_id),
                        )
                    # Replace mask
                    sub_template = sub_template.replace(
                        "*mask*", "*label_{}*".format(label_id)
                    )
                    new_template = new_template + sub_template
                data_args.template = new_template

    # Create config
    config = AutoConfig.from_pretrained(
        (
            model_args.config_name
            if model_args.config_name
            else model_args.model_name_or_path
        ),
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )

    if "prompt" in model_args.few_shot_type:
        if config.model_type == "roberta":
            model_fn = RobertaForPromptFinetuning
        elif config.model_type == "bert":
            model_fn = BertForPromptFinetuning
        elif config.model_type == "albert":
            model_fn = AlbertForPromptFinetuning
        elif config.model_type == "distilbert":
            model_fn = DistilBertForPromptFinetuning
        else:
            raise NotImplementedError
    elif model_args.few_shot_type == "finetune":
        model_fn = AutoModelForSequenceClassification
    else:
        raise NotImplementedError
    special_tokens = []

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        (
            model_args.tokenizer_name
            if model_args.tokenizer_name
            else model_args.model_name_or_path
        ),
        additional_special_tokens=special_tokens,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    # print(f" | tokenizer: {tokenizer}, size: {len(tokenizer)} \n\n\n")

    # Get our special datasets.
    if model_args.few_shot_type == "finetune":
        assert data_args.num_sample == 1
        eval_dataset = GlueDataset(data_args, tokenizer, mode="dev")

        if eval_dataset is not None:
            eval_dataset.num_sample = 1
    else:
        use_demo = "demo" in model_args.few_shot_type
        eval_dataset = FewShotDataset(
            data_args, tokenizer=tokenizer, mode="dev", use_demo=use_demo
        )
    # count_input_ids2(eval_dataset)
    print(f" *** eval dataset sizes: {len(eval_dataset)}")

    model = model_fn.from_pretrained(
        model_args.model_name_or_path,
        from_tf=False,
        config=config,
        cache_dir=model_args.cache_dir,
    )
    # print(f" *** model type: {type(model)}")

    # For BERT, increase the size of the segment (token type) embeddings
    if config.model_type == "bert":
        model.resize_token_embeddings(len(tokenizer))
        resize_token_type_embeddings(
            model, new_num_types=10, random_segment=model_args.random_segment
        )

    # Pass dataset and argument information to the model
    if data_args.prompt:
        model.label_word_list = torch.tensor(eval_dataset.label_word_list).long().cuda()
        # print(f" | Classification label_word_list: {model.label_word_list}")
        # print(
        #     f"   converted words: {tokenizer.convert_ids_to_tokens(model.label_word_list)}"
        # )
    if output_modes_mapping[data_args.task_name] == "regression":
        # lower / upper bounds
        model.lb, model.ub = bound_mapping[data_args.task_name]
        print(f" | Regression lb: {model.lb}, ub: {model.ub}")
    model.model_args = model_args
    model.data_args = data_args
    model.tokenizer = tokenizer

    # data_args.task_name = original_task_name
    return model, eval_dataset


def get_sens_insens_blocks(constitution, start_idx):
    sensitive_blocks = []
    insensitive_blocks = []
    for i, block in enumerate(constitution):
        if block == i + start_idx:
            sensitive_blocks.append(i)
        else:
            insensitive_blocks.append(i)
    return sensitive_blocks, insensitive_blocks


def make_boxplot(blocks, constitution, model_id, fig_name):
    import matplotlib.pyplot as plt

    assert blocks.shape[0] == len(constitution)
    start_idx = 0 if model_id == 0 else 832
    start_idx += 832 - len(constitution)
    sensitive_blocks, insensitive_blocks = get_sens_insens_blocks(
        constitution, start_idx
    )
    print(f"Number of sensitive blocks: {len(sensitive_blocks)}")
    print(f"Number of insensitive blocks: {len(insensitive_blocks)}")

    # Compute l1 norm for each block, blocks shape: (n_all_blocks, BLOCKSIZE)
    l1_norms = np.linalg.norm(blocks, ord=1, axis=1)
    sens_l1 = l1_norms[sensitive_blocks]
    insens_l1 = l1_norms[insensitive_blocks]

    # Compute l2 norm for each block
    l2_norms = np.linalg.norm(blocks, ord=2, axis=1)
    sens_l2 = l2_norms[sensitive_blocks]
    insens_l2 = l2_norms[insensitive_blocks]

    # Compute l-inf norm for each block
    l_inf_norms = np.linalg.norm(blocks, ord=np.inf, axis=1)
    sen_inf = l_inf_norms[sensitive_blocks]
    insen_inf = l_inf_norms[insensitive_blocks]

    # Compute 3rd quartile for each block
    sens_3rd_quartile = np.percentile(blocks, 75, axis=1)
    sen_3rd = sens_3rd_quartile[sensitive_blocks]
    insen_3rd = sens_3rd_quartile[insensitive_blocks]

    # Plot boxplot
    fig, ax = plt.subplots(1, 4, figsize=(16, 10))
    ax[0].boxplot([sens_l1, insens_l1], labels=["Sensitive", "Insensitive"], widths=0.8)
    ax[0].set_title("L1 Norm")
    ax[1].boxplot([sens_l2, insens_l2], labels=["Sensitive", "Insensitive"], widths=0.8)
    ax[1].set_title("L2 Norm")
    ax[2].boxplot([sen_inf, insen_inf], labels=["Sensitive", "Insensitive"], widths=0.8)
    ax[2].set_title("L-inf Norm")
    ax[3].boxplot([sen_3rd, insen_3rd], labels=["Sensitive", "Insensitive"], widths=0.8)
    ax[3].set_title("3rd Quartile")
    plt.savefig(fig_name, bbox_inches="tight")


def find_layers(module, layers=[nn.Linear], name=""):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


# Define WrappedGPT class
class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples


def get_sensitivity(
    measure: str,
    exp_name: str,
    model_id: int,
    skip_embeds: bool = True,
):
    l1_exp = load_final_constitution()
    constitution = l1_exp[exp_name][str(model_id)]
    exps = exp_name.split("-")
    exp = exps[model_id]
    taskname, eps = exp.split("_")

    blocks, _ = get_block_sensitivity(taskname, measure, eps, skip_embeds)

    # First three layers aren't linear layers thus not used in Wanda method
    constitution = constitution[-blocks.shape[0] :]

    # Make a box plot
    self_other = "self" if model_id == 0 else "other"
    fig_name = f"plots/sensitivity/{measure}_{eps}_{self_other}.png"
    make_boxplot(blocks, constitution, model_id, fig_name)


def get_block_sensitivity(
    model_info,
    measure,
    skip_embeds=True,
    return_n_embed_blocks=False,
    return_model=True,
):
    model_args, data_args, training_args = parse_args()
    data_args.task_name = model_info["task_name"]
    model_args.model_name_or_path = model_info["model_path"]
    # Get the model and dataset
    model, dataset = get_model_and_dateset(
        data_args,
        model_args,
        training_args,
    )

    block_size = model_args.block_size
    if measure == "magnitude":
        blocks = magnitute_sensitivity(model, block_size, skip_embeds=skip_embeds)
    elif measure == "fisher":
        blocks = fisher_sensitity(model, dataset, block_size, skip_embeds=skip_embeds)
    elif measure == "wanda":
        blocks = wanda_sensitivity(model, dataset, block_size)
    elif measure == "gradient":
        blocks = gradient_sensitity(model, dataset, block_size)
    else:
        raise ValueError(f"Unknown sensitivity measure: {measure}")

    if return_n_embed_blocks:
        model.cpu()
        embed_params = {}
        for name, params in model.named_parameters():
            if "embeddings" in name:
                embed_params[name] = params
        embed_blocks = block_model_1d(block_size, embed_params)["blocks"]
        n_embed_blocks = embed_blocks.shape[0]
        return blocks, n_embed_blocks
    if return_model:
        return blocks, block_model_1d(block_size, model)["blocks"]
    return blocks, None


def magnitute_sensitivity(model, block_size, skip_embeds=True):
    model_storage = block_model_1d(block_size, model, skip_embeds)
    blocks = model_storage["blocks"]
    return blocks


def fisher_sensitity(
    model, dataset, block_size, batch_size=16, skip_embeds=True, sample_size=None
):
    model.eval()
    model.cuda()

    if sample_size is None:
        sample_size = len(dataset)

    for start_id in range(0, sample_size, batch_size):
        end_idx = min(start_id + batch_size, sample_size)

        input_ids = torch.tensor(
            [[dataset[i].input_ids for i in range(start_id, end_idx)]], dtype=torch.long
        )
        input_ids = input_ids.squeeze().cuda()

        attention_mask = torch.tensor(
            [[dataset[i].attention_mask for i in range(start_id, end_idx)]],
            dtype=torch.long,
        )
        attention_mask = attention_mask.squeeze().cuda()

        mask_pos = torch.tensor(
            [[dataset[i].mask_pos for i in range(start_id, end_idx)]], dtype=torch.long
        )
        mask_pos = mask_pos.squeeze().cuda()

        labels = torch.tensor(
            [dataset[i].label for i in range(start_id, end_idx)], dtype=torch.long
        )
        labels = labels.unsqueeze(1).cuda()

        # Get logits
        logits = model(input_ids, attention_mask, mask_pos)[0]
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
    blocks = block_model_1d(block_size, fim, skip_embeds)["blocks"]

    return blocks


@torch.no_grad()
def wanda_sensitivity(model, dataset, block_size):
    layer_names = []
    for name, param in model.named_parameters():
        layer_names.append(name)

    model.eval()
    model.cuda()

    # Instead of using the whole dataset, we use a subset of it
    batch_size = 128
    # batch_size = len(dataset)
    input_ids = torch.tensor(
        [[dataset[i].input_ids for i in range(batch_size)]], dtype=torch.long
    )
    input_ids = input_ids.squeeze().cuda()

    attention_mask = torch.tensor(
        [[dataset[i].attention_mask for i in range(batch_size)]], dtype=torch.long
    )
    attention_mask = attention_mask.squeeze().cuda()

    mask_pos = torch.tensor(
        [[dataset[i].mask_pos for i in range(batch_size)]], dtype=torch.long
    )
    mask_pos = mask_pos.squeeze().cuda()

    # Wanda method
    subset = find_layers(model)
    wrapped_layers = {}
    for name in subset:
        wrapped_layers[name] = WrappedGPT(subset[name])

    def add_batch(name):
        def tmp(_, inp, out):
            wrapped_layers[name].add_batch(inp[0].data, out.data)

        return tmp

    handles = []
    for name in wrapped_layers:
        handles.append(subset[name].register_forward_hook(add_batch(name)))
    model(input_ids, attention_mask, mask_pos)
    for h in handles:
        h.remove()

    wanda_importance = {}
    for name in subset:
        W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
            wrapped_layers[name].scaler_row.reshape((1, -1))
        )
        wanda_importance[name] = W_metric.cpu()

    blocks = block_model_1d(block_size, wanda_importance)["blocks"]
    torch.cuda.empty_cache()
    return blocks


def gradient_sensitity(
    model,
    dataset,
    block_size,
    batch_size=16,
    skip_embeds=False,
    sample_size=None,
    correct_only=True,
    do_block=True,
):
    model.eval()
    model.cuda()

    if sample_size is None:
        sample_size = len(dataset)
    else:
        sample_size = min(sample_size, len(dataset))
        dataset = dataset[:sample_size]

    criterion = nn.CrossEntropyLoss()
    accum_iter = np.ceil(sample_size / batch_size)

    for start_id in range(0, sample_size, batch_size):
        end_idx = min(start_id + batch_size, sample_size)

        input_ids = torch.tensor(
            [[dataset[i].input_ids for i in range(start_id, end_idx)]], dtype=torch.long
        )
        input_ids = input_ids.squeeze().cuda()

        attention_mask = torch.tensor(
            [[dataset[i].attention_mask for i in range(start_id, end_idx)]],
            dtype=torch.long,
        )
        attention_mask = attention_mask.squeeze().cuda()

        mask_pos = torch.tensor(
            [[dataset[i].mask_pos for i in range(start_id, end_idx)]], dtype=torch.long
        )
        mask_pos = mask_pos.squeeze().cuda()

        labels = torch.tensor(
            [dataset[i].label for i in range(start_id, end_idx)], dtype=torch.long
        )
        # labels = labels.unsqueeze(1).cuda()
        labels = labels.cuda()

        # Get logits
        logits = model(input_ids, attention_mask, mask_pos)[0]

        if correct_only:
            preds = torch.argmax(logits, dim=1)
            mask = preds == labels
            labels = labels[mask]
            logits = logits[mask]

        loss = criterion(logits, labels)
        loss = loss / accum_iter
        loss.backward()

    grads = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            grads[name] = param.grad

    if not do_block:
        return grads

    # Block the Fisher information corresponding to each parameter
    blocks = block_model_1d(block_size, grads, skip_embeds)["blocks"]

    return blocks


if __name__ == "__main__":
    # skip_embeds = False
    # for measure in ["magnitude", "fisher", "wanda"]:
    #     # for measure in ["fisher"]:
    #     for exp in ["sst_4-sst_11", "sst_7-sst_8", "sst_11-sst_4"]:
    #         for model_id in [0, 1]:
    #             get_sensitivity(measure, exp, model_id, skip_embeds)

    exps = [("sst", 8), ("sst", 11), ("sst", 12)]
    all_zero_counts = []
    total_blocks = []
    for taskname, eps in exps:
        blocks, _ = get_block_sensitivity(taskname, "gradient", eps, False, False)

        all_zeros = np.all(blocks == 0, axis=1)
        all_zero_counts.append(np.sum(all_zeros))
        total_blocks.append(len(all_zeros))
        all_zero_blocks = np.nonzero(all_zeros)[0]
        print(f"Blocks that graidents are all zeros: {all_zero_blocks}")

        # all_zero_blocks = []
        # for i, block in enumerate(blocks):
        #     if np.all(block == 0):
        #         all_zero_blocks.append(i + 1)
        # print(f"Blocks that graidents are all zeros: {all_zero_blocks}")
        # all_zero_counts.append(len(all_zero_blocks))
        # total_blocks.append(len(blocks))

    for n_all_zeros, nblocks in zip(all_zero_counts, total_blocks):
        print(f"Number of all-zero blocks: {n_all_zeros} out of {nblocks}")
