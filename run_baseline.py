import os

os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TQDM_DISABLE"] = "1"

# from baselines.baseline5 import run
from baselines.self_deduplication import run

print(run.__doc__)
run()
