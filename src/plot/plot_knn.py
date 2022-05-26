import pandas as pd
import wandb
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.rcParams.update({'font.size': 15})
matplotlib.rcParams.update({'figure.autolayout': True})

ax = plt.axes()
ax_color_cycle = ax._get_lines.prop_cycler


def moving_avg(arr, window=10):
    arr = np.concatenate([np.ones(window - 1) * arr[0], arr])
    return np.convolve(arr, np.ones(window, dtype=float) / window, mode="valid")


api = wandb.Api()

run_name = "jifan/cifar100_unbalanced_10"
n_class = int(run_name.split("_")[-1])
dataset = run_name[6:].split("_")[0]
runs = api.runs(run_name)
acc_balance_fn = lambda maj_acc, min_acc: (maj_acc + (n_class - 1) * min_acc) / n_class

alg_list = [r"$S^2$ (1-Nearest-Neighbor)", r"$S^2$ (ResNet18)", "GALAXY (Ours)"]

colors = ["tab:orange", "tab:orange", "tab:blue"]
line_styles = ["--", "-", "-"]

runs_dict = {alg: [[], []] for alg in alg_list}
for run in runs:
    if "Active" in run.name:
        num_queries = []
        balanced_accuracies = []
        for i, row in run.history().iterrows():
            num_queries.append(row["Num Queries"])
            balanced_accuracies.append(acc_balance_fn(row["Majority Class Accuracy"], row["Minority Class Accuracy"]))
        num_queries, balanced_accuracies = np.array(num_queries), np.array(balanced_accuracies)
        runs_dict["GALAXY (Ours)"][0].append(num_queries[num_queries <= 5000])
        runs_dict["GALAXY (Ours)"][1].append(balanced_accuracies[num_queries <= 5000])
    elif "kNN" in run.name:
        num_queries = []
        graph_balanced_accuracies = []
        balanced_accuracies = []
        for i, row in run.history().iterrows():
            num_queries.append(row["Num Queries"])
            graph_balanced_accuracies.append(
                acc_balance_fn(row["Graph Majority Class Accuracy"], row["Graph Minority Class Accuracy"]))
            balanced_accuracies.append(acc_balance_fn(row["Majority Class Accuracy"], row["Minority Class Accuracy"]))
        num_queries, graph_balanced_accuracies, balanced_accuracies = np.array(num_queries), np.array(
            graph_balanced_accuracies), np.array(balanced_accuracies)
        runs_dict[alg_list[1]][0].append(num_queries[num_queries <= 5000])
        runs_dict[alg_list[1]][1].append(balanced_accuracies[num_queries <= 5000])
        runs_dict[alg_list[0]][0].append(num_queries[num_queries <= 5000])
        runs_dict[alg_list[0]][1].append(graph_balanced_accuracies[num_queries <= 5000])

for alg, style, color in zip(alg_list, line_styles, colors):
    num_queries, balanced_accuracies = runs_dict[alg]
    if len(num_queries) == 0:
        next(ax_color_cycle)
        continue
    num_queries, balanced_accuracies = np.array(num_queries), np.array(balanced_accuracies)
    num_queries = num_queries[0]
    balanced_accuracies_mean = np.mean(balanced_accuracies, axis=0)
    balanced_accuracies_ste = np.std(balanced_accuracies, ddof=1, axis=0) / np.sqrt(balanced_accuracies.shape[0])
    balanced_accuracies_mean, balanced_accuracies_ste = moving_avg(balanced_accuracies_mean), moving_avg(balanced_accuracies_ste)
    plt.plot(num_queries, balanced_accuracies_mean, label=alg, linewidth=3, color=color, linestyle=style)
    plt.fill_between(num_queries, balanced_accuracies_mean - balanced_accuracies_ste,
                     balanced_accuracies_mean + balanced_accuracies_ste, alpha=.3, color=color)

plt.xlabel("Number of Labels")
plt.ylabel("Balanced Accuracy")
plt.legend()
plt.savefig(run_name.split("/")[1] + "_comparison.pdf")
plt.show()
