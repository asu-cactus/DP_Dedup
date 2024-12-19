from dataclasses import dataclass, field

from typing import Optional


from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import HfArgumentParser

from text_task_utils.common import true_tags
from text_task_utils.compiled_args import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    task_type: str = field(
        metadata={
            "help": "Task type. Check utils.common.py for the list of task types."
        },
    )
    block_size: int = field(
        default=589824,
        metadata={"help": "Block size for the model"},
    )
    prune: bool = field(
        default=False,
        metadata={"help": "Whether to prune the model"},
    )
    quantize: bool = field(
        default=False,
        metadata={"help": "Whether to quantize the model"},
    )
    heter: bool = field(
        default=False,
        metadata={"help": "Whether to use heterogeneous model deduplication"},
    )
    big_batch: bool = field(
        default=False,
        metadata={"help": "Whether to use a big group (greater than 5)"},
    )
    dummy_base_model: int = field(
        default=-1,
        metadata={
            "help": "Index of the dummy base model, -1 meaning not using dummy base model"
        },
    )
    inter_data_mode: str = field(
        default="None",
        metadata={
            "help": "Inter-data mode. Choice: None, cifar100_celeba, celeba_cifar100, qnli_sst2, sst2_qnli"
        },
    )
    save_combined_storage: bool = field(
        default=False,
        metadata={"help": "Whether to save the combined storage"},
    )
    n_base_models: int = field(
        default=1,
        metadata={"help": "Number of base models"},
    )
    in_group_n_base: bool = field(
        default=False,
        metadata={"help": "Special in-group n base model experiment"},
    )
    base_model_selection: bool = field(
        default=False,
        metadata={"help": "Whether to select base models"},
    )
    # For text task
    model_name_or_path: Optional[str] = field(
        default="",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )
    # Few-shot type
    #   - finetune: standard fine-tuning
    #   - prompt: prompt-based fine-tuning
    #   - prompt-demo: prompt-based fine-tuning with demonstrations
    few_shot_type: str = field(
        default="prompt-demo",
        metadata={
            "help": "Few-shot learning model type. Choice: finetune, prompt, prompt-demo"
        },
    )

    # Only for BERT-type model
    random_segment: bool = field(
        default=False,
        metadata={
            "help": "Whether to reinitialize the token type embeddings (only for BERT)."
        },
    )

    static_embedding: str = field(default="no")
    static_lm_head: str = field(default="no")
    attention_only: str = field(default="no")
    bias_only: str = field(default="no")

    randomly_initialize: str = field(
        default="no",
        metadata={
            "help": "Randomly initialize the model; useful only for ablation studies."
        },
    )

    def __post_init__(self):
        self.static_embedding = self.static_embedding.lower() in true_tags  # noqa
        self.static_lm_head = self.static_lm_head.lower() in true_tags  # noqa
        self.attention_only = self.attention_only.lower() in true_tags  # noqa
        self.bias_only = self.bias_only.lower() in true_tags  # noqa
        self.randomly_initialize = self.randomly_initialize.lower() in true_tags  # noqa


@dataclass
class DynamicDataTrainingArguments(DataTrainingArguments):
    """
    Arguments for dynamic training.
    """

    # For text task
    data_root_dir: Optional[str] = field(
        default="../fast-differential-privacy/examples/text_classification/data/original/",
        metadata={"help": "The input data root dir."},
    )
    task_name: Optional[str] = field(
        default="",
        metadata={"help": "The name of the task that the model is trained on"},
    )
    data_dir: Optional[str] = field(
        default="",
        metadata={
            "help": "The input data dir. Not used in the script, just make the script works"
        },
    )

    # Original settings
    num_k: Optional[int] = field(
        default=1, metadata={"help": "Number of training instances per class"}
    )

    num_sample: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of samples (for inference) in fine-tuning with demonstrations"
        },
    )

    num_demo: Optional[int] = field(
        default=1, metadata={"help": "Number of demonstrations from each class"}
    )

    auto_demo: bool = field(
        default=True,
        metadata={"help": "Automatically generate template for using demonstrations"},
    )

    # For prompting
    template: str = field(default=None, metadata={"help": "Template"})

    mapping: str = field(default=None, metadata={"help": "Label word mapping"})

    template_path: str = field(
        default=None,
        metadata={
            "help": "Path to a txt file that stores all the templates, one per line. Do not set this when prompt_path "
            "is used"
        },
    )

    mapping_path: str = field(
        default=None,
        metadata={
            "help": "Path to a txt file that stores all the label word mappings, one per line. Do not set this when "
            "prompt_path is used"
        },
    )

    prompt_path: str = field(
        default=None,
        metadata={
            "help": "Path to a txt file that stores all the prompts (templates and mappings), one per line"
        },
    )

    template_id: int = field(
        default=None, metadata={"help": "Template id if using template_path"}
    )

    mapping_id: int = field(
        default=None, metadata={"help": "Mapping id if using template_path"}
    )

    prompt_id: int = field(
        default=None, metadata={"help": "Prompt id if using prompt_path"}
    )

    top_n_template: int = field(
        default=None, metadata={"help": "Use top-n template in the template path"}
    )

    # For logging
    tag: str = field(
        default="",
        metadata={"help": "Set the tag and find the result easier in the log."},
    )

    # For filtering when using demonstrations
    demo_filter: bool = field(
        default=False, metadata={"help": "Only use similar instances in demonstrations"}
    )

    demo_filter_rate: float = field(
        default=0.5,
        metadata={"help": "Only use top-x\% similar instances in demonstrations"},
    )

    demo_filter_model: str = field(
        default=None,
        metadata={
            "help": "Model name for demonstration filter embeddings. Will load embeddings based on the model name."
        },
    )

    debug_mode: bool = field(default=False, metadata={"help": "Debug mode"})

    # For max length
    max_seq_len: int = field(
        default=256, metadata={"help": "Maximum sequence length for the model"}
    )

    double_demo: bool = field(
        default=False, metadata={"help": "Use double length for using demonstrations"}
    )

    first_sent_limit: int = field(
        default=None,
        metadata={"help": "Limit the length of the first sentence (i.e., sent_0)"},
    )

    other_sent_limit: int = field(
        default=None,
        metadata={
            "help": "Limit the length of sentences other than the first sentence"
        },
    )

    use_full_length: bool = field(
        default=None, metadata={"help": "Use the full length (512)"}
    )

    # GPT-3's in-context learning
    gpt3_in_context_head: bool = field(
        default=False,
        metadata={"help": "GPT-3's in-context learning (context at the beginning)"},
    )

    gpt3_in_context_tail: bool = field(
        default=False,
        metadata={"help": "GPT-3's in-context learning (context at the end)"},
    )

    gpt3_in_context_num: int = field(
        default=32, metadata={"help": "Number of context examples"}
    )

    truncate_head: bool = field(
        default=False,
        metadata={
            "help": "When exceeding the maximum length, truncate the head instead of the tail."
        },
    )

    # Do not set up the following fields. They are set up automatically.
    prompt: bool = field(
        default=False, metadata={"help": "Whether to use prompt-based fine-tuning"}
    )
    template_list: tuple = field(
        default=None,
        metadata={
            "help": "(DO NOT List of templates (only initialized after the program starts."
        },
    )

    inference_time_demo: bool = field(
        default=False,
        metadata={
            "help": "Do not use demonstrations during inference time; "
            "the original paper attaches to each test example a few training examples as demo -- "
            "apparently this breaks privacy. We turn this off by default here."
        },
    )


@dataclass
class DynamicTrainingArguments(TrainingArguments):
    disable_tqdm: bool = field(
        default=True,
        metadata={"help": "Disable tqdm"},
    )
    per_device_eval_batch_size: int = field(
        default=16,
        metadata={"help": "Evaluation batch size per device"},
    )
    output_dir: str = field(
        default="outputs",
        metadata={"help": "Output directory"},
    )

    # For batch deduplication (used in every_n.py) and MCTS
    every_n: int = field(
        default=10,
        metadata={"help": "Run deduplication every n blocks"},
    )
    # For binary search and successive halving (used in binary_search.py and recursive_search_variant.py)
    min_dedup_len: int = field(
        default=10,
        metadata={"help": "Minimum number of blocks to deduplicate"},
    )

    # For heuristics
    orderby: str = field(
        default="l2_norm",
        metadata={
            "help": "Order block_2b_replaced by one of [3rd_quantile, l2_norm, l1_norm, l_inf_norm]"
        },
    )
    sensitivity_measure: str = field(
        default="gradient",
        metadata={
            "help": "Sensitivity measure, choice: [magnitude, fisher, wanda, gradient]"
        },
    )

    # For fairnes
    enforce_fairness: bool = field(
        default=True,
        metadata={"help": "Enforce fairness during deduplication"},
    )

    # SVT arguments
    extra_val_eps: Optional[float] = field(
        default=-1, metadata={"help": "Epsilon for SVT"}
    )
    max_fails: Optional[int] = field(
        default=3, metadata={"help": "Maximum number of fails for SVT"}
    )

    # For MCTS
    n_episodes: int = field(
        default=20,
        metadata={"help": "Number of episodes for MCTS"},
    )
    cprod: float = field(
        default=0.1,
        metadata={"help": "C-prod for UCT"},
    )
    # save_every: int = field(
    #     default=1000,
    #     metadata={"help": "Save MCTS results every n episodes"},
    # )

    # # For Heuristic MC-RAVE
    # mcts_mode: str = field(
    #     default="dyn_prune_mcts",
    #     metadata={
    #         "help": "MCTS mode, choice: [mc_rave, heuristic_mc_rave, uct_mcts, dyn_prune_mcts, dyn_prune_uct_rave]"
    #     },
    # )

    # # MCTS parameters
    # fanout: int = field(
    #     default=100,
    #     metadata={"help": "Fanout of the first sub-action for MCTS"},
    # )
    # eval_every: int = field(
    #     default=3,
    #     metadata={"help": "Evaluate every n steps"},
    # )
    # top_k: int = field(
    #     default=5,
    #     metadata={"help": "Top-k actions to consider in heuristic test"},
    # )
    # top_k_actual: int = field(
    #     default=1,
    #     metadata={"help": "Top-k actions to consider in actual test"},
    # )
    # equivalence_param: int = field(
    #     default=10000,
    #     metadata={"help": "Equivalence parameter k for Hand-select schedule"},
    # )

    # # For MCTS resume search
    # resume: bool = field(
    #     default=False,
    #     metadata={"help": "Whether to resume Es dict for MCTS"},
    # )
    # resume_episode: int = field(
    #     default=0,
    #     metadata={"help": "Resume from episode n"},
    # )
    # keep_n: int = field(
    #     default=5,
    #     metadata={"help": "Keep n saved MCTS results"},
    # )

    # # For ensemble
    # array_id: int = field(
    #     default=-1,
    #     metadata={
    #         "help": "Array ID (contains seed and hyper-paramter search) to idenfity the model"
    #     },
    # )

    # model_id: int = field(
    #     default=-1,
    #     metadata={
    #         "help": "Model ID (contains template information) to identify the model"
    #     },
    # )

    # save_logit: bool = field(
    #     default=False,
    #     metadata={
    #         "help": "Save test file logit with name $TASK-$MODEL_ID-$ARRAY_ID.npy"
    #     },
    # )

    # save_logit_dir: str = field(
    #     default=None, metadata={"help": "Where to save the prediction result"}
    # )

    # # Regularization
    # fix_layers: int = field(
    #     default=0, metadata={"help": "Fix bottom-n layers when optimizing"}
    # )

    # # Training
    # save_at_last: bool = field(
    #     default=False,
    #     metadata={
    #         "help": "Instead of saving the best (dev performance) checkpoint, save the last checkpoint"
    #     },
    # )

    # # Turn off train/test
    # no_train: bool = field(default=True, metadata={"help": "No training"})
    # no_predict: bool = field(default=True, metadata={"help": "No test"})

    # evaluate_after_training: bool = field(
    #     default=True, metadata={"help": "Always run evaluation after training ends."}
    # )

    def __post_init__(self):
        super(DynamicTrainingArguments, self).__post_init__()


def parse_args():
    parser = HfArgumentParser(
        (
            ModelArguments,
            DynamicDataTrainingArguments,
            DynamicTrainingArguments,
        )
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    task_type = model_args.task_type
    if task_type.startswith("text_") or task_type in ("cifar100_qnli", "cifar100_sst2"):
        model_args.block_size = 589824
        # model_args.block_size = 49152
        if model_args.task_type.endswith("mnli"):
            training_args.enforce_fairness = False
        data_args.dataset_name = "CIFAR100"
        # model_args.untouched_weights = 569433
        # model_args.n_original_weights = 163300953
    elif model_args.task_type == "vision_vit":
        model_args.model = "vit_large_patch16_224"
        data_args.dataset_name = (
            "CIFAR100"
            if not model_args.heter
            and not model_args.inter_data_mode == "cifar100_celeba"
            else "CelebA"
        )
        training_args.bs = 500
        training_args.mini_bs = 50
        model_args.block_size = 1048576 if not model_args.heter else 262144
        # model_args.untouched_weights = 1414244
        # model_args.n_original_weights = 303404132
    elif model_args.task_type == "vision_resnet":
        model_args.model = "resnet152.tv2_in1k"
        data_args.dataset_name = "CelebA"
        training_args.bs = 500
        training_args.mini_bs = 50
        model_args.block_size = 262144
        # model_args.untouched_weights = 2913384
        # model_args.n_original_weights = 58225768
    elif model_args.task_type == "recommendation":
        training_args.bs = 512
        training_args.mini_bs = 64
        model_args.block_size = 1180000
        # model_args.untouched_weights = 30005
        # model_args.n_original_weights = 236210005

    print(f"model_args:\n{model_args}")
    print(f"data_args:\n{data_args}")
    print(f"training_args:\n{training_args}")
    return model_args, data_args, training_args
