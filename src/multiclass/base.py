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
    wandb.init(project="%s" % data, entity=wandb_name,
               name="BASE, Pretrained, Weighted, Batch Size %d, ResNet" % batch_size)
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
        pred, min_margins, per_class_margins = trainer.pred(imgs, ret_margins=True)
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
        for c in range(n_class):
            cur_class_query_count = int(batch_size / n_class) + int(c < batch_size % n_class)
            if cur_class_query_count == 0:
                continue
            cur_class_distance = torch.where(torch.argmax(pred, dim=-1) == c, min_margins,
                                             per_class_margins[:, c].squeeze())

            cur_labeled_idxs = list(torch.argsort(cur_class_distance, descending=False).numpy().astype(int))
            count = 0
            for idx in cur_labeled_idxs:
                if idx not in queried_set:
                    queried.append(idx)
                    queried_set.add(idx)
                    count += 1
                if count == cur_class_query_count:
                    break
        assert len(queried) == num_queries
        model_acc = []
        batch = torch.from_numpy(np.array(list(queried))).long()
        trainer.train(imgs[batch], labels[batch],
                      new_idxs=torch.arange(num_queries - batch_size, num_queries), n_class=n_class)
        with torch.no_grad():
            pred, min_margins, per_class_margins = trainer.pred(imgs, ret_margins=True)
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

    torch.save(trainer.model, open("mlp_%s_%d_%d.model" % (data, batch_size, seed), "wb"))

    wandb.finish()
