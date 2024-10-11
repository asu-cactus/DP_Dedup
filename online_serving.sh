#!/bin/bash
python run_online_serving.py -L disk -W roundrobin -D CIFAR100 --n_queries 100
python run_online_serving.py -L disk -W random -D CIFAR100 --n_queries 100
