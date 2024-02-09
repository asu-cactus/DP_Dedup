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

    model_ids: Optional[str] = field(
        default="",
        metadata={
            "help": "Model ID to identify the model. Should be numbers separated by comma, e.g., 0,2,3"
        },
    )
    model_name_or_path: Optional[str] = field(
        default="../fast-differential-privacy/examples/outputs/base-sst-2-eps8/",
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

    data_root_dir: Optional[str] = field(
        default="../fast-differential-privacy/examples/text_classification/data/original/",
        metadata={"help": "The input data root dir."},
    )
    task_name: Optional[str] = field(
        default="sst-2",
        metadata={"help": "The name of the task that the model is trained on"},
    )

    data_dir: Optional[str] = field(
        default="",
        metadata={
            "help": "The input data dir. Not used in the script, just make the script works"
        },
    )

    num_k: Optional[int] = field(
        default=16, metadata={"help": "Number of training instances per class"}
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
    n_episodes: int = field(
        default=10000,
        metadata={"help": "Number of episodes for MCTS"},
    )
    save_every: int = field(
        default=100,
        metadata={"help": "Save MCTS results every n episodes"},
    )
    keep_n: int = field(
        default=5,
        metadata={"help": "Keep n saved MCTS results"},
    )
    # For Heuristic MC-RAVE
    mcts_mode: str = field(
        default="heuristic_mc_rave",
        metadata={"help": "MCTS mode, choice: heuristic_mc_rave, uct_mcts"},
    )
    top_k: int = field(
        default=5,
        metadata={"help": "Top-k actions to consider in heuristic test"},
    )
    equivalence_param: int = field(
        default=1000,
        metadata={"help": "Equivalence parameter k for Hand-select schedule"},
    )
    cprod: float = field(
        default=0.1,
        metadata={"help": "C-prod for UCT for Heuristic MC-RAVE"},
    )
    # For resume search
    resume: bool = field(
        default=False,
        metadata={"help": "Whether to resume runing MCTS"},
    )
    resume_episode: int = field(
        default=0,
        metadata={"help": "Resume from episode n"},
    )

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
    print(f"model_args:\n{model_args}")
    print(f"data_args:\n{data_args}")
    print(f"training_args:\n{training_args}")
    return model_args, data_args, training_args
