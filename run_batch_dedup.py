import os

os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TQDM_DISABLE"] = "1"
from batch_dedup.every_n import run

# from batch_dedup.binary_search import run

# from batch_dedup.recursive_search_variant import run

run()
