import numpy as np
import subprocess

python_dir = "python"
wandb_name = "jifan"

# for data in ["cifar_unbalanced_2", "cifar_unbalanced_3", "cifar100_unbalanced_2", "cifar100_unbalanced_3",
#              "cifar100_unbalanced_10", "svhn_unbalanced_2", "svhn_unbalanced_3"]:
for data in ["medmnist"]:  # "cifar100_unbalanced_10",
    num_processes = 4
    processes = []
    for seed in np.linspace(1234, 9999999, num=num_processes, dtype=int)[:1]:
        # processes.append(subprocess.Popen([python_dir, "multiclass/passive.py", str(seed), data, wandb_name]))
        # processes.append(subprocess.Popen([python_dir, "multiclass/multi_linear_deeps2.py", str(seed), data, wandb_name]))
        # processes.append(subprocess.Popen([python_dir, "multiclass/confidence_sampling.py", str(seed), data, wandb_name]))
        # processes.append(subprocess.Popen([python_dir, "multiclass/most_likely_positive.py", str(seed), data, wandb_name]))
        processes.append(subprocess.Popen([python_dir, "cluster_margin/cluster_margin.py", str(seed), data, wandb_name]))
        # processes.append(subprocess.Popen([python_dir, "multiclass/base.py", str(seed), data, wandb_name]))
        # processes.append(subprocess.Popen([python_dir, "multiclass/knn_s2.py", str(seed), data, wandb_name]))

    for p in processes:
        p.wait()
