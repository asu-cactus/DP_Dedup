import os

os.environ["WANDB_DISABLED"] = "true"
os.environ["TQDM_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from batch_dedup.greedy_n import run
print("Running greedy_n")

run()
