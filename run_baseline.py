import os

os.environ["WANDB_DISABLED"] = "true"

from baselines.baseline3 import run

print(run.__doc__)
run()
