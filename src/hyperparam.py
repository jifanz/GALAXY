import argparse
import torch
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("seed", type=int)
parser.add_argument("data", type=str)
parser.add_argument("wandb_name", type=str, default="")
parser.add_argument("--smi_func", type=str, default="fl2mi")
args = parser.parse_args()
seed = args.seed
wandb_name = args.wandb_name
torch.manual_seed(seed)
np.random.seed(seed + 98765)
data = args.data
pretrain = True

if data[:15] == "svhn_unbalanced":
    batch_size = 100
    num_iters = 50
    n_class = int(data.split("_")[-1])
    init_batch_size = (batch_size // n_class) * n_class
elif data[:16] == "cifar_unbalanced":
    batch_size = 100
    num_iters = 50
    n_class = int(data.split("_")[-1])
    init_batch_size = (batch_size // n_class) * n_class
elif data[:19] == "cifar100_unbalanced":
    batch_size = 1000
    num_iters = 40
    n_class = int(data.split("_")[-1])
    init_batch_size = (batch_size // n_class) * n_class
elif data == "medmnist":
    batch_size = 100
    num_iters = 50
    n_class = 2
    init_batch_size = (batch_size // n_class) * n_class
elif data[:19] == "imagenet_unbalanced":
    batch_size = 1000
    num_iters = 10
    n_class = int(data.split("_")[-1])
    init_batch_size = (batch_size // n_class) * n_class
    pretrain = False
elif data == "svhn":
    init_batch_size = 100
    batch_size = 10
    num_iters = 500
    n_class = 10
elif data == "cifar":
    init_batch_size = 250
    batch_size = 100
    num_iters = 50
    n_class = 10
elif data == "cifar100":
    init_batch_size = 100
    batch_size = 100
    num_iters = 50
    n_class = 100

