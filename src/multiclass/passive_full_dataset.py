from src.graph import *
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from src.model import train_resnet18
from src.linear_graph import create_linear_from_list
from src.s2algorithm import s2query
from src.trainer import OnlineTrainer, PassiveResnet18Trainer, PassiveMLPTrainer
import matplotlib.pyplot as plt
from src.utils import calculate_auc, calculate_mAP
from src.model import Resnet18Classification
import wandb
from src.hyperparam import *
from src.dataset import get_data


if __name__ == "__main__":
    wandb.init(project="%s" % data, entity=wandb_name,
               name="Passive Full Dataset, Pretrained, Weighted, Batch Size %d, ResNet" % batch_size)
    wandb.config = {
        "init_batch_size": init_batch_size,
        "batch_size": batch_size,
        "num_iters": num_iters,
    }
    imgs, labels = get_data()
    rnd = torch.Generator()
    rnd.manual_seed(12345)
    perm = torch.randperm(labels.size(0), generator=rnd).cuda()
    imgs, labels = imgs[perm], labels[perm]
    # trainer = PassiveMLPTrainer(batch_size=init_batch_size)
    trainer = PassiveResnet18Trainer(batch_size=init_batch_size)
    # model = Resnet18Classification(n_class).cuda()
    # trainer = OnlineTrainer(model, torch.optim.Adam(model.parameters(), lr=1e-4),
    #                         loss_fn=nn.CrossEntropyLoss(),
    #                         accuracy_fn=lambda pred, labels: torch.mean(
    #                             (torch.argmax(pred, dim=-1) == labels).float()).data.cpu().numpy())
    trainer.train(imgs, labels, torch.arange(labels.size(0)), n_class=n_class)
    with torch.no_grad():
        pred = trainer.pred(imgs)
    model_acc = torch.mean((torch.argmax(pred, dim=-1) == labels).float()).data.cpu().numpy()
    majority_acc = torch.mean(
        (torch.argmax(pred, dim=-1) == labels).float()[labels == (n_class - 1)]).data.cpu().numpy()
    minority_acc = torch.mean(
        (torch.argmax(pred, dim=-1) == labels).float()[labels != (n_class - 1)]).data.cpu().numpy()
    num_preds = torch.sum((labels.unsqueeze(-1) == torch.arange(n_class)).float(), dim=0)
    wandb.log({"Model Accuracy": model_acc,
               "Num Queries": imgs.size(0),
               "Per Class Min Queries": torch.min(num_preds).cpu().numpy(),
               "Per Class Max Queries": torch.max(num_preds).cpu().numpy(),
               "Majority Class Accuracy": majority_acc,
               "Minority Class Accuracy": minority_acc})

    wandb.finish()
