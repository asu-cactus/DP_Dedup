import os

os.environ["WANDB_DISABLED"] = "true"

from baselines.baseline5 import run

print(run.__doc__)
run()
