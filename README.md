#GALAXY: Graph-based Active Learning at the Extreme

###Package Requirements
Install the following dependencies:

```PyTorch```

```Weights and Biases (wandb, register for user)```

### Exporting Environment Variable
To run python scripts in the directories below, export the following variable name when a new terminal is started:
```
export PYTHONPATH="${PYTHONPATH}:<path to>/multi-graph_active_learning
```

###File Structure
* `src/mp_launcher.py`  Multi-process launcher for GALAXY, Random, Confidence Sampling, Most Likely Positive, BASE, Cluster_Margin and vanilla S^2 on kNN graph. Example commands to launch individual experiment included. To launch all experiments:
```
python mp_launcher.py
```
* `src/bait`   Implementation adopted from https://openreview.net/forum?id=DHnThtAyoPj. Run BADGE and BAIT. Example command:
```
  python bait/run.py --model resnet --nQuery 100 --data CIFAR100 --alg badge --nStart 100 --unbalance 10 --wandb_name <entity name>
```
* `src/distill` A modified version of DISTILL adopted from https://github.com/decile-team/distil.
* `src/similar` Our script for running SIMILAR. Example:
```
python similar/similar.py --seed 123 --data svhn_unbalanced_3 --wandb_name <entity name>
```
* `src/cluster_margin` Our implementation of cluster margin.
* `src/multiclass` Our implementation of GALAXY and various baselines.
