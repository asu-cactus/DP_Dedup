import os

os.environ["WANDB_DISABLED"] = "true"

from baselines.baseline4 import run

print(run.__doc__)
run()
