import os

os.environ["WANDB_DISABLED"] = "true"

from batch_dedup.every_n import run

run()
