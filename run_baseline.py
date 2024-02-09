import os

os.environ["WANDB_DISABLED"] = "true"

from baselines.baseline2 import run

print(run.__doc__)
run()
