from src.graph import *
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from src.model import train_resnet18, Resnet18Classification
from src.multiclass.knn_graph import create_knn_graph
from src.s2algorithm import s2query, bisection_query
from src.trainer import OnlineTrainer, PassiveResnet18Trainer, PassiveMLPTrainer
import matplotlib.pyplot as plt
from src.utils import calculate_mAP
import wandb
from src.dataset import get_data
from src.hyperparam import *


def run_online_deep_s2(imgs, labels, trainer, init_batch_size, batch_size, num_iter, num_classes):
    assert init_batch_size % num_classes == 0
    rnd = torch.Generator()
    rnd.manual_seed(321)
    init_batch = torch.randperm(labels.size(0), generator=rnd)[:init_batch_size]

    trainer.model = Resnet18Classification(num_output=num_classes).cuda()
    with torch.no_grad():
        scores, features = trainer.pred(imgs, ret_features=True)
    graph = create_knn_graph(features, labels.cpu().numpy(), "graph_init", n_order=10)

    trainer.train(imgs[init_batch], labels[init_batch], torch.arange(init_batch_size))
    with torch.no_grad():
        scores, features = trainer.pred(imgs, ret_features=True)
    for idx in init_batch.cpu().numpy():
        graph.label(idx)
    graph.nn_pred()
    model_acc = torch.mean((torch.argmax(scores, dim=-1) == labels).float()).data.cpu().numpy()
    majority_acc = torch.mean(
        (torch.argmax(scores, dim=-1) == labels).float()[labels == (num_classes - 1)]).data.cpu().numpy()
    minority_acc = torch.mean(
        (torch.argmax(scores, dim=-1) == labels).float()[labels != (num_classes - 1)]).data.cpu().numpy()
    graph_pred = np.array(graph.graph.preds, dtype=int)
    graph_labels = np.array(graph.graph.labels, dtype=int)
    graph_majority_acc = np.mean((graph_pred == graph_labels).astype(float)[graph_labels == (num_classes - 1)])
    graph_minority_acc = np.mean((graph_pred == graph_labels).astype(float)[graph_labels != (num_classes - 1)])
    num_preds = torch.sum((labels[init_batch].unsqueeze(-1) == torch.arange(num_classes)).float(), dim=0)
    wandb.log({"Num Queries": init_batch_size,
               "Model Accuracy": model_acc,
               "Per Class Min Queries": torch.min(num_preds).cpu().numpy(),
               "Per Class Max Queries": torch.max(num_preds).cpu().numpy(),
               "Majority Class Accuracy": majority_acc,
               "Minority Class Accuracy": minority_acc,
               "Graph Majority Class Accuracy": graph_majority_acc,
               "Graph Minority Class Accuracy": graph_minority_acc})
    queried = init_batch
    num_queries = [init_batch_size]
    while num_queries[-1] < (num_iter - 1) * batch_size + init_batch_size:
        new_queries = []
        for _ in range(batch_size):
            query_idx, alter_idx = bisection_query(graph)
            if query_idx is None:
                if len(new_queries) < 100:
                    query_idx = alter_idx
                else:
                    break
            graph.label(query_idx)
            new_queries.append(query_idx)
        graph.nn_pred()
        num_queries.append(num_queries[-1] + len(new_queries))
        new_queries = torch.from_numpy(np.array(new_queries))
        queried = torch.cat([queried, new_queries], dim=0)
        trainer.train(imgs[queried], labels[queried], torch.arange(num_queries[-2], num_queries[-1]))
        with torch.no_grad():
            scores, features = trainer.pred(imgs, ret_features=True)
        model_acc = torch.mean((torch.argmax(scores, dim=-1) == labels).float()).data.cpu().numpy()
        majority_acc = torch.mean(
            (torch.argmax(scores, dim=-1) == labels).float()[labels == (num_classes - 1)]).data.cpu().numpy()
        minority_acc = torch.mean(
            (torch.argmax(scores, dim=-1) == labels).float()[labels != (num_classes - 1)]).data.cpu().numpy()
        num_preds = torch.sum((labels[queried].unsqueeze(-1) == torch.arange(num_classes)).float(), dim=0)
        graph_pred = np.array(graph.graph.preds, dtype=int)
        graph_labels = np.array(graph.graph.labels, dtype=int)
        graph_majority_acc = np.mean((graph_pred == graph_labels).astype(float)[graph_labels == (num_classes - 1)])
        graph_minority_acc = np.mean((graph_pred == graph_labels).astype(float)[graph_labels != (num_classes - 1)])
        wandb.log({"Num Queries": num_queries[-1],
                   "Model Accuracy": model_acc,
                   "Per Class Min Queries": torch.min(num_preds).cpu().numpy(),
                   "Per Class Max Queries": torch.max(num_preds).cpu().numpy(),
                   "Majority Class Accuracy": majority_acc,
                   "Minority Class Accuracy": minority_acc,
                   "Graph Majority Class Accuracy": graph_majority_acc,
                   "Graph Minority Class Accuracy": graph_minority_acc})
    print(num_queries)


if __name__ == "__main__":
    wandb_run = wandb.init(project="%s" % data, entity=wandb_name,
                           name="kNN S2 Order 10, Batch Size %d, ResNet" % batch_size,
                           reinit=True)
    wandb.config = {
        "init_batch_size": init_batch_size,
        "batch_size": batch_size,
        "num_iters": num_iters
    }
    imgs, labels = get_data()
    # trainer = PassiveMLPTrainer(batch_size=init_batch_size)
    trainer = PassiveResnet18Trainer(batch_size=init_batch_size)
    # model = Resnet18Classification(n_class).cuda()
    # trainer = OnlineTrainer(model, torch.optim.Adam(model.parameters(), lr=1e-4),
    #                         loss_fn=nn.CrossEntropyLoss(),
    #                         accuracy_fn=lambda pred, labels: torch.mean(
    #                             (torch.argmax(pred, dim=-1) == labels).float()).data.cpu().numpy(),
    #                         batch_size=None)
    run_online_deep_s2(imgs, labels, trainer, init_batch_size, batch_size, num_iters, n_class)

    wandb_run.finish()
