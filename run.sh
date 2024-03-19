#!/bin/bash

# python main.py --output_dir outputs_rave

# python main.py --output_dir outputs_mcts --mcts_mode uct_mcts

nohup python run_mcts.py --output_dir outputs_rave_1sub --top_k_actual 1 &> outputs_rave_1sub.out &