#!/bin/bash

# python main.py --output_dir outputs_rave

# python main.py --output_dir outputs_mcts --mcts_mode uct_mcts

# nohup python run_mcts.py --output_dir outputs_rave_1sub --top_k_actual 1 &> outputs_rave_1sub.out &

nohup python run_online_serving.py -L memory -W roundrobin -D qnli --data_root_dir data &> outputs/serving_roberta/roberta_memory_roundrobin.out &
nohup python run_online_serving.py -L memory -W random -D qnli --data_root_dir data &> outputs/serving_roberta/roberta_memory_random.out &
nohup python run_online_serving.py -L disk -W roundrobin -D qnli --data_root_dir data &> outputs/serving_roberta/roberta_disk_roundrobin.out &
nohup python run_online_serving.py -L disk -W random -D qnli --data_root_dir data &> outputs/serving_roberta/roberta_disk_random.out &