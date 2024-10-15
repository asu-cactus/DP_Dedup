import os

os.environ["WANDB_DISABLED"] = "true"
os.environ["TQDM_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# from batch_dedup.dred import run
from batch_dedup.dred_base_model_selection import run

print("Running dred")

run()
