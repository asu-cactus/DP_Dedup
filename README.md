### This is a repository for our research paper **Privacy and Accuracy-Aware Model Deduplication for Emerging Model Infrastructures**. 

- [System Requirements](#system-requirements)
- [Software Installation](#software-installation)
  - [FastDP](#fastdp)
  - [DP\_Dedup](#dp_dedup)
- [Datasets](#datasets)
  - [Text datasets](#text-datasets)
  - [CIFAR100](#cifar100)
  - [CelebA](#celeba)
- [Run Experiments](#run-experiments)
  - [Train DP models](#train-dp-models)
  - [Dynamic algorithms](#dynamic-algorithms)
  - [GreedyN algorithms](#greedyn-algorithms)
  - [MCTS algorithms](#mcts-algorithms)
  - [Online serving](#online-serving)
  - [Pruning and quantization](#pruning-and-quantization)
  - [Evaluate original models](#evaluate-original-models)
  - [All commands for reproduction](#all-commands-for-reproduction)
## System Requirements
The training of the ViT-large model with a big batch size 50 needs a GPU with memory size 40 GB. 
Inferencing with a batch size 16 will need about 11 GB GPU memory for ViT-large, 6 GB GPU memory for Roberta-base and 5 GB GPU memory for ResNet152.

The online serving experiment use an AWS c5a.xlarge instance (7.63 GB main memory, 4vCPU) with a batch size 1.

## Software Installation
We used the fastDP Python package to train our DP models(available from https://github.com/awslabs/fast-differential-privacy). The deduplication and inference code also use the same dependencies.

### FastDP
Here are the recommended steps to set up the environment:

1. Use conda to create a virtual environment with Python 3.10 to be compatible with the Pytorch release. Here is an example command to create the environment named fastdp: `conda create -n fastdp python=3.10`
2. Install pytorch 1.11 because versions above pytorch 1.12 can slow down the training. This is stated on the home page of its official repository. Here is an example comand to install pytorch 1.11: `conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch`
3. Clone the official repository and install it: `python -m pip install -e .`.
4. Change opacus version to `opacus==1.4.0` in requirements.txt (otherwise you may encounter a `torch.func not found` error.)
5. Install the requirements with no dependencies: `pip install -r requirements.txt --no-dependencies`.
6. Install timm: `pip install -U timm`.
7. Install transformers: `pip install -U sentence-transformers`.
8. Install sentence-transformers: `pip install -U sentence-transformers`.
9. Install accelerate: `pip install -U accelerate`
10. Install pandas: `pip install -U pandas`
11. Install ml-swissknife: `pip install -U ml-swissknife`.

### DP_Dedup
Clone this repository and activate the above `fastdp` environment. 

## Datasets
The steps to download the dataset are showed below. These data are used in both training and model deduplication.

For training DP models, please refer to the instructions in `fastDP` to set up the datasets. For model deduplication, create a folder named `data` in the root directory of this repo to hold datasets.

### Text datasets
Download the data by `wget https://nlp.cs.princeton.edu/projects/lm-bff/datasets.tar` and extract the files to the `data` folder: `tar xvf datasets.tar`. We will only use the QNLI, MNLI, and SST2 datasets.

### CIFAR100
The dataset can be downloaded from https://www.cs.toronto.edu/~kriz/cifar.html
Unzip the downloaded file into the `data` folder.

### CelebA
Here is the home page of the CelebA dataset: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

After click the first link under "Downloads", you will be redirected to a Google Drive, where you can download the data. Then download the followings:
/home/local/ASUAD/hguan6/DP_dedup/fastdp_examples/image_classification1. The file `img_align_celeba.zip` in the `Img` folder. 
2. All text files in `Eval` and `Anno`. 
3. Create a folder under `data` named `celeba` and unzip the `img_align_celeba.zip` file into `celeba` and also move all text files to `celeba`.

## Run Experiments
To run the experiments, the first thing you need is to train DP models. Next, you can run the model deduplication algorithms: GreedyN, Dynamic algorithms (DRD and DRED), and MCTS. Lastly, you can run the model online serving on a resource constraint environment, such as an AWS c5a.xlarge instance.

### Train DP models
We used the example code in `fastDP` to train DP models: Roberta-base models on QNLI, MNLI, and SST-2; ViT models on CIFAR100 and CelebA, ResNet152 models on CelebA.

Here's an example of a bash script to the QNLI dataset
```
#!/bin/bash
python -m text_classification.run_wrapper \
    --model_name_or_path roberta-base \
    --physical_batch_size 16 \
    --output_dir text_classification/models/base-qnli-eps0.2 \
    --task_name qnli \
    --target_epsilon 0.2 \
```

To enable training five partitions of MNLI seperately, we modified the `dataset.py` file in `examples/text_classification/src/` and `run_classification.py` in `examples/text_classification/`. You can replace these files in `fastdp_examples/text_classification/` of this repository. You also need to change the `part_id` argument in `run_classification.py` to specify a partition. The choices for `part_id` are {0,1,2,3,4}.

To train the vision models, run the example code by specifying the model and dataset. We set `mini_bs` to 50 and the `epochs` to 5 and used the default values for other arguments.

After training DP models, make a folder named `models` that is in the same folder with our repository `DP_Dedup`.
Then move the trained models to this `models` folder, which will be loaded by deduplication algorithm.

### Dynamic algorithms
To run our dynamic algorithms, simply run `python run_drd.py` or `python run_dred.py` with a task_type. For example,
``` 
python run_drd.py --task_type text_qnli
```
To enable SVT, set argument `extra_val_eps` to a value greater than or equal to 0. You can also change the cut-off value `c` by setting argument `max_fails` to a positive integer. For example,
```
python run_drd.py --task_type text_qnli --extra_val_eps 0.0 --max_fails 3
```
You can save the final deduplicated models by enable `--save_combined_storage True`.

### GreedyN algorithms
To run GreedyN algorithms, run `python run_greedyn.py` with argument `every_n`, which means evaluation is performed after deduplicating every `n` blocks. Other arguments are the same as the dynamic algorithms. For example,
```
python run_greedyn.py --task_type vision_vit --every_n 30
```

### MCTS algorithms
To run MCTS algorithms, run `python run_mcts.py` with argument `every_n`. For example,
```
python run_mcts.py --task_type vision_vit --every_n 20
```

### Online serving
To run online serving, you can run `python run_online_serving.py` with three arguments. For `--load_from`, you can choose from `memory` and `disk` . For `--workload`, you can choose from `random` and `roundrobin`. For `--dataset_name`, you can choose from `CIFAR100`, `CelebA`, and `qnli`.

### Pruning and quantization
You can run pruning and quantization by setting `--prune True` or `--quantize True`. Currently, we only experimented them on ResNet152.

### Evaluate original models
To evaluate the original models, you can comment or uncomment the task in `evaluate_models.py` and run `python evaluate_models.py` with the same arguments as described in the dynamic algorithms section.

### All commands for reproduction
Here are all commands for reproducing the results in our paper.