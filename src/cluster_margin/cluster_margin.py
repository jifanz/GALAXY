from src.graph import *
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from src.model import Resnet18Classification
from src.s2algorithm import s2query
from src.trainer import OnlineTrainer, PassiveResnet18Trainer, PassiveMLPTrainer
import matplotlib.pyplot as plt
import wandb
from src.hyperparam import *
from src.dataset import get_data

if __name__ == "__main__":
    cluster_batch_size = int(1.25 * batch_size)
    cluster_idxs = np.load(open("./cluster_margin/%s_resnet18_clusters" % (data.split("_")[0]), "rb"))

    wandb.init(project="%s" % data, entity=wandb_name,
               name="Cluster Margin, Pretrained, Weighted, Batch Size %d, ResNet" % batch_size)
    num_trials = 1
    wandb.config = {
        "init_batch_size": init_batch_size,
        "batch_size": batch_size,
        "num_iters": num_iters,
        "num_trials": num_trials
    }
    imgs, labels = get_data()
    rnd = torch.Generator()
    rnd.manual_seed(321)
    init_batch = torch.randperm(labels.size(0), generator=rnd)[:init_batch_size]
    queried = list(init_batch.cpu().numpy())
    queried_set = set(queried)
    trainer = PassiveResnet18Trainer(batch_size=init_batch_size)
    trainer.train(imgs[init_batch], labels[init_batch], torch.arange(init_batch_size), n_class=n_class)
    with torch.no_grad():
        pred = trainer.pred(imgs)
    model_acc = torch.mean((torch.argmax(pred, dim=-1) == labels).float()).data.cpu().numpy()
    majority_acc = torch.mean(
        (torch.argmax(pred, dim=-1) == labels).float()[labels == (n_class - 1)]).data.cpu().numpy()
    minority_acc = torch.mean(
        (torch.argmax(pred, dim=-1) == labels).float()[labels != (n_class - 1)]).data.cpu().numpy()
    num_preds = torch.sum((labels[init_batch].unsqueeze(-1) == torch.arange(n_class)).float(), dim=0)
    wandb.log({"Model Accuracy": model_acc,
               "Num Queries": init_batch_size,
               "Per Class Min Queries": torch.min(num_preds).cpu().numpy(),
               "Per Class Max Queries": torch.max(num_preds).cpu().numpy(),
               "Majority Class Accuracy": majority_acc,
               "Minority Class Accuracy": minority_acc})

    for i in range(num_iters - 1):
        num_queries = init_batch_size + (i + 1) * batch_size
        sorted_pred = torch.sort(pred, dim=-1, descending=True)[0]
        margins = sorted_pred[:, 0] - sorted_pred[:, 1]
        uncertain_idxs = np.argsort(margins.cpu().numpy())
        clusters = [[] for _ in range(np.max(cluster_idxs) + 1)]
        num_batch_points = 0
        for idx in uncertain_idxs:
            if idx not in queried_set:
                clusters[cluster_idxs[idx]].append(idx)
                num_batch_points += 1
            if num_batch_points == cluster_batch_size:
                break
        cluster_sizes = np.array([len(c) for c in clusters])
        cluster_sorted_idxs = np.argsort(cluster_sizes)
        c_idx = 0
        while len(queried) != num_queries:
            cluster_idx = cluster_sorted_idxs[c_idx]
            if len(clusters[cluster_idx]) != 0:
                idx = clusters[cluster_idx].pop(0)
                queried.append(idx)
                queried_set.add(idx)
            c_idx = (c_idx + 1) % len(clusters)
        model_acc = []
        batch = torch.from_numpy(np.array(list(queried))).long()
        trainer.train(imgs[batch], labels[batch],
                      new_idxs=torch.arange(num_queries - batch_size, num_queries), n_class=n_class)
        with torch.no_grad():
            pred = trainer.pred(imgs)
        model_acc.append(torch.mean((torch.argmax(pred, dim=-1) == labels).float()).data.cpu().numpy())
        majority_acc = torch.mean(
            (torch.argmax(pred, dim=-1) == labels).float()[labels == (n_class - 1)]).data.cpu().numpy()
        minority_acc = torch.mean(
            (torch.argmax(pred, dim=-1) == labels).float()[labels != (n_class - 1)]).data.cpu().numpy()
        num_preds = torch.sum((labels[batch].unsqueeze(-1) == torch.arange(n_class)).float(), dim=0)
        wandb.log({"Model Accuracy": np.mean(np.array(model_acc)),
                   "Num Queries": num_queries,
                   "Per Class Min Queries": torch.min(num_preds).cpu().numpy(),
                   "Per Class Max Queries": torch.max(num_preds).cpu().numpy(),
                   "Majority Class Accuracy": majority_acc,
                   "Minority Class Accuracy": minority_acc})

    torch.save(trainer.model, open("cluster_margin_%s_%d_%d.model" % (data, batch_size, seed), "wb"))

    wandb.finish()
