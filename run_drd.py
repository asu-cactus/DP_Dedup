import os

os.environ["WANDB_DISABLED"] = "true"
os.environ["TQDM_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# from batch_dedup.drd import run

# from batch_dedup.drd_base_model_selection import run
from batch_dedup.drd_latency_breakdown import run


print("Running drd")

run()
