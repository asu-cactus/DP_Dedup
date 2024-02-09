"""Finetuning the library models for sequence classification on GLUE."""

import logging
import os
from typing import Callable, Dict

import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
)
from transformers import GlueDataset

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

from utils.blocker import reconstruct_weight
from utils.parse_args import (
    ModelArguments,
    DynamicDataTrainingArguments,
    DynamicTrainingArguments,
)
from text_task_utils import common

# from text_task_utils.trainer import Trainer
from transformers import Trainer


# logger = logging.getLogger(__name__)


def evaluate(
    model_storage,
    model_id: int,
    model_constitution: list[int],
    data_args: DynamicDataTrainingArguments,
    model_args: ModelArguments,
    training_args: DynamicTrainingArguments,
):
    # Add some additional arguments to make it work
    task_name = data_args.task_name
    data_args.data_dir = (
        f"{data_args.data_root_dir}/{common.task_name2suffix_name[task_name]}"
    )

    training_args.local_rank = -1

    data_args.template = {
        "sst-2": "*cls**sent_0*_It_was*mask*.*sep+*",
        "mnli": "*cls**sent-_0*?*mask*,*+sentl_1**sep+*",
        "qnli": "*cls**sent-_0*?*mask*,*+sentl_1**sep+*",
        "qqp": "*cls**sent-_0**mask*,*+sentl_1**sep+*",
    }[task_name]

    # Work with some arguments
    if "prompt" in model_args.few_shot_type:
        data_args.prompt = True

    # if training_args.no_train:
    #     training_args.do_train = False
    # if training_args.no_predict:
    #     training_args.do_predict = False

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

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
            # logger.info(
            #     "Specify load the %d-th prompt: %s | %s"
            #     % (data_args.prompt_id, data_args.template, data_args.mapping)
            # )
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
                # logger.info(
                #     "Load top-%d templates from %s"
                #     % (len(data_args.template_list), data_args.template_path)
                # )

                # ... or load i-th template
                if data_args.template_id is not None:
                    data_args.template = data_args.template_list[data_args.template_id]
                    data_args.template_list = None
                    # logger.info(
                    #     "Specify load the %d-th template: %s"
                    #     % (data_args.template_id, data_args.template)
                    # )

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
                # logger.info(
                #     "Specify using the %d-th mapping: %s"
                #     % (data_args.mapping_id, data_args.mapping)
                # )

    # # Check save path
    # logger.info("Training/evaluation parameters %s", training_args)

    try:
        num_labels = num_labels_mapping[data_args.task_name]
        # output_mode = output_modes_mapping[data_args.task_name]
        # logger.info(
        #     "Task name: {}, number of labels: {}, output mode: {}".format(
        #         data_args.task_name, num_labels, output_mode
        #     )
        # )
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Automatically generate template for using demonstrations
    if data_args.auto_demo and model_args.few_shot_type == "prompt-demo":
        # GPT-3's in-context learning
        if data_args.gpt3_in_context_head or data_args.gpt3_in_context_tail:
            # logger.info(
            #     "Automatically convert the template to GPT-3's in-context learning."
            # )
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
                # logger.info("| {} => {}".format(data_args.template, new_template))
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

    print(f" *** eval dataset sizes: {len(eval_dataset)}")

    model = model_fn.from_pretrained(
        model_args.model_name_or_path,
        from_tf=False,
        config=config,
        cache_dir=model_args.cache_dir,
    )
    print(f" *** model type: {type(model)}")

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

    # Reconstruct the parameters using the model constitution
    reconstruct_weight(model_storage, model, model_id, model_constitution)

    # Build metric
    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            # Note: the eval dataloader is sequential, so the examples are in order.
            # We average the logits over each sample for using demonstrations.
            predictions = p.predictions
            num_logits = predictions.shape[-1]
            # print(f"predictions shape: {predictions.shape}")
            logits = predictions.reshape([eval_dataset.num_sample, -1, num_logits])
            logits = logits.mean(axis=0)

            if num_logits == 1:
                preds = np.squeeze(logits)
            else:
                preds = np.argmax(logits, axis=1)

            # Just for sanity, assert label ids are the same.
            label_ids = p.label_ids.reshape([eval_dataset.num_sample, -1])
            label_ids_avg = label_ids.mean(axis=0)
            label_ids_avg = label_ids_avg.astype(p.label_ids.dtype)
            assert (label_ids_avg - label_ids[0]).mean() < 1e-2
            label_ids = label_ids[0]

            return compute_metrics_mapping[task_name](task_name, preds, label_ids)

        return compute_metrics_fn

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(data_args.task_name),
    )
    metrics = trainer.evaluate()
    print("*" * 20)
    return metrics["eval_acc"]
