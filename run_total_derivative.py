import os

os.environ["WANDB_DISABLED"] = "true"

from total_derivative.total_derivative3 import run

run()
