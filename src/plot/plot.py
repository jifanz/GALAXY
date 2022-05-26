import pandas as pd
import wandb
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.rcParams.update({'font.size': 15})
matplotlib.rcParams.update({'figure.autolayout': True})
# plt.rcParams["figure.autolayout"] = True
ax = plt.axes()
ax_color_cycle = ax._get_lines.prop_cycler


def moving_avg(arr, window=10):
    arr = np.concatenate([np.ones(window - 1) * arr[0], arr])
    return np.convolve(arr, np.ones(window, dtype=float) / window, mode="valid")


api = wandb.Api()

for run_name in ["jifan/svhn_unbalanced_2", "jifan/svhn_unbalanced_3", "jifan/cifar_unbalanced_2",
                 "jifan/cifar_unbalanced_3", "jifan/cifar100_unbalanced_2", "jifan/cifar100_unbalanced_3",
                 "jifan/cifar100_unbalanced_10"]:
    n_class = int(run_name.split("_")[-1])
    dataset = run_name[6:].split("_")[0]
    runs = api.runs(run_name)
    acc_balance_fn = lambda maj_acc, min_acc: (maj_acc + (n_class - 1) * min_acc) / n_class

    alg_list = ["Random", "SIMILAR (FLQMI)", "BAIT", "BADGE", "Most Likely Positive", "Cluster Margin", "BASE",
                "Confidence Sampling", "GALAXY (Ours)"]

    run_alg_names = ["Passive", "fl2mi", "bait", "badge", "Most", "Margin", "BASE", "Conf", "Active"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:green", "tab:pink", "tab:orange", "tab:gray", "tab:pink", "tab:blue"]
    line_styles = ["--", "--", "--", "-", "--", "-", "-", "-", "-"]

    runs_dict = {alg: [[], [], []] for alg in alg_list}
    for run in runs:
        num_queries = []
        num_minority_queries = []
        balanced_accuracies = []
        for i, row in run.history().iterrows():
            num_queries.append(row["Num Queries"])
            num_minority_queries.append(row["Num Queries"] - row["Per Class Max Queries"])
            balanced_accuracies.append(acc_balance_fn(row["Majority Class Accuracy"], row["Minority Class Accuracy"]))
        num_queries, num_minority_queries, balanced_accuracies = np.array(num_queries), np.array(
            num_minority_queries), np.array(balanced_accuracies)
        for alg, run_alg in zip(alg_list, run_alg_names):
            if run_alg in run.name:
                runs_dict[alg][0].append(num_queries[num_queries <= 5000])
                runs_dict[alg][1].append(num_minority_queries[num_queries <= 5000])
                runs_dict[alg][2].append(balanced_accuracies[num_queries <= 5000])

    for alg, style, color in zip(alg_list, line_styles, colors):
        num_queries, num_minority_queries, balanced_accuracies = runs_dict[alg]
        if len(num_queries) == 0:
            next(ax_color_cycle)
            continue
        num_queries, num_minority_queries, balanced_accuracies = np.array(num_queries), np.array(
            num_minority_queries), np.array(balanced_accuracies)
        num_queries = num_queries[0]
        num_minority_queries_mean = np.mean(num_minority_queries, axis=0)
        num_minority_queries_ste = np.std(num_minority_queries, ddof=1, axis=0) / np.sqrt(num_minority_queries.shape[0])
        balanced_accuracies_mean = np.mean(balanced_accuracies, axis=0)
        balanced_accuracies_ste = np.std(balanced_accuracies, ddof=1, axis=0) / np.sqrt(balanced_accuracies.shape[0])
        num_minority_queries_mean, num_minority_queries_ste, balanced_accuracies_mean, balanced_accuracies_ste = \
            moving_avg(num_minority_queries_mean), moving_avg(num_minority_queries_ste), moving_avg(
                balanced_accuracies_mean), \
            moving_avg(balanced_accuracies_ste)
        plt.plot(num_queries, balanced_accuracies_mean, label=alg, linewidth=3, color=color, linestyle=style)
        plt.fill_between(num_queries, balanced_accuracies_mean - balanced_accuracies_ste,
                         balanced_accuracies_mean + balanced_accuracies_ste, alpha=.3, color=color)
        if alg == "Random":
            y_lim = balanced_accuracies_mean[0]

    plt.xlabel("Number of Labels")
    plt.ylabel("Balanced Accuracy")
    # plt.legend()
    plt.ylim([y_lim, None])
    plt.savefig(run_name.split("/")[1] + "_accuracy.pdf")
    if run_name == "jifan/svhn_unbalanced_2":
        plt.legend()
        plt.savefig(run_name.split("/")[1] + "_accuracy_legend.pdf")
    plt.show()

    for alg, style, color in zip(alg_list, line_styles, colors):
        num_queries, num_minority_queries, balanced_accuracies = runs_dict[alg]
        if len(num_queries) == 0:
            next(ax_color_cycle)
            continue
        num_queries, num_minority_queries, balanced_accuracies = np.array(num_queries), np.array(
            num_minority_queries), np.array(balanced_accuracies)
        num_queries, num_minority_queries, balanced_accuracies = num_queries[:, num_queries[0] <= 5000], \
                                                                 num_minority_queries[:, num_queries[0] <= 5000], \
                                                                 balanced_accuracies[:, num_queries[0] <= 5000]
        num_queries = num_queries[0]
        num_minority_queries_mean = np.mean(num_minority_queries, axis=0)
        num_minority_queries_ste = np.std(num_minority_queries, ddof=1, axis=0) / np.sqrt(num_minority_queries.shape[0])
        balanced_accuracies_mean = np.mean(balanced_accuracies, axis=0)
        balanced_accuracies_ste = np.std(balanced_accuracies, ddof=1, axis=0) / np.sqrt(balanced_accuracies.shape[0])
        num_minority_queries_mean, num_minority_queries_ste, balanced_accuracies_mean, balanced_accuracies_ste = \
            moving_avg(num_minority_queries_mean), moving_avg(num_minority_queries_ste), moving_avg(
                balanced_accuracies_mean), \
            moving_avg(balanced_accuracies_ste)
        plt.plot(num_queries, num_minority_queries_mean, label=alg, linewidth=3, color=color, linestyle=style)
        plt.fill_between(num_queries, num_minority_queries_mean - num_minority_queries_ste,
                         num_minority_queries_mean + num_minority_queries_ste, alpha=.3, color=color)

    plt.xlabel("Number of Labels")
    plt.ylabel("Number of In-distribution Labels")
    plt.legend()
    plt.savefig(run_name.split("/")[1] + "_labels.pdf")
    plt.show()
