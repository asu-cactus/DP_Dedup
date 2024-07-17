import os

os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TQDM_DISABLE"] = "1"
# from batch_dedup.every_n import run
# print("Running every_n")

# from batch_dedup.binary_search import run
# print("Running binary_search")

from batch_dedup.recursive_search_variant import run

print("Running recursive_search_variant")

run()
