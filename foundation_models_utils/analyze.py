def get_lora_size():
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
    adapter = PeftModel.from_pretrained(
        model, "../../models/OrpoLlama-3-8B", is_trainable=True
    )
    print(adapter.print_trainable_parameters())


def test_similar_blocks():
    with open("../outputs/vit_cifar100_binary_search_min20.out", "r") as f:
        for line in f:
            if line.startswith("../models"):
                segments = line.split()
                if segments[-2] != "->":
                    continue
                block1 = int(segments[-3])
                block2 = int(segments[-1])
                diff = block1 - block2
                if diff != 288:
                    print(f"Block1: {block1}, Block2: {block2}, Diff: {diff}")
