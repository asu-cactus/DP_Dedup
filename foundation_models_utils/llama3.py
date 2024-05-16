import os

os.environ["TQDM_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from pathlib import Path
import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import ORPOConfig, ORPOTrainer, setup_chat_format


def load_model(base_model="meta-llama/Meta-Llama-3-8B"):
    path = Path(base_model)
    if torch.cuda.get_device_capability()[0] >= 8:
        attn_implementation = "flash_attention_2"
        torch_dtype = torch.bfloat16
    else:
        attn_implementation = "eager"
        torch_dtype = torch.float16

    # QLoRA config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer
    access_token = os.environ["HUGGINGFACE_ACCESS_TOKEN"]
    tokenizer = AutoTokenizer.from_pretrained(path, token=access_token)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        path,
        token=access_token,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation=attn_implementation,
    )

    model, tokenizer = setup_chat_format(model, tokenizer)
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def load_data(tokenizer):
    dataset_name = "mlabonne/orpo-dpo-mix-40k"
    dataset = load_dataset(dataset_name, split="all")
    dataset = dataset.shuffle(seed=42).select(range(1000))

    def format_chat_template(row):
        row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
        row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
        return row

    dataset = dataset.map(
        format_chat_template,
        num_proc=os.cpu_count(),
    )
    dataset = dataset.train_test_split(test_size=0.01)
    return dataset


def train_model(
    model,
    tokenizer,
    dataset,
    new_model="../../models/OrpoLlama-3-8B",
):
    # LoRA config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "up_proj",
            "down_proj",
            "gate_proj",
            "k_proj",
            "q_proj",
            "v_proj",
            "o_proj",
        ],
    )

    orpo_args = ORPOConfig(
        learning_rate=8e-6,
        beta=0.1,
        lr_scheduler_type="linear",
        max_length=1024,
        max_prompt_length=512,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        num_train_epochs=1,
        evaluation_strategy="steps",
        eval_steps=0.2,
        logging_steps=1,
        warmup_steps=10,
        report_to="wandb",
        output_dir="./results/",
    )

    trainer = ORPOTrainer(
        model=model,
        args=orpo_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(new_model)


def run():
    wandb.login()
    model, tokenizer = load_model()
    dataset = load_data(tokenizer)
    train_model(model, tokenizer, dataset)


if __name__ == "__main__":
    run()
