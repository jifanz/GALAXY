import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F

from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, Subset, ConcatDataset, DataLoader
from torchvision import transforms
import torchvision.datasets as datasets
import wandb

from src.distil.distil.active_learning_strategies import SMI, GLISTER, BADGE, EntropySampling, \
    RandomSampling  # All active learning strategies showcased in this example
from src.model import Resnet18Classification  # The model used in our image classification example
from src.distil.distil.utils.train_helper import data_train  # A utility training class provided by DISTIL
from src.distil.distil.utils.utils import \
    LabeledToUnlabeledDataset  # A utility wrapper class that removes labels from labeled PyTorch dataset objects

from src.trainer import PassiveResnet18Trainer
from src.hyperparam import *
from src.dataset import get_data


if __name__ == "__main__":
    smi_func = args.smi_func
    wandb.init(project="%s" % data, entity=wandb_name,
               name="SMI_%s, Pretrained, Weighted, Batch Size %d, ResNet" % (smi_func, batch_size))
    num_trials = 1
    wandb.config = {
        "init_batch_size": init_batch_size,
        "batch_size": batch_size,
        "num_iters": num_iters,
        "num_trials": num_trials
    }
    imgs, labels, dataset = get_data(ret_dataset=True)
    rnd = torch.Generator()
    rnd.manual_seed(321)
    init_batch = torch.randperm(len(dataset), generator=rnd)[:init_batch_size]

    # Define the number of classes in CIFAR10
    nclasses = n_class

    # We create the imbalance on classes 5,6,7,8,9.
    rare_classes = list(range(n_class - 1))

    # Create the imbalance by choosing which indices of the full training set to assign to the initial labeled seed set and the unlabeled set
    train_idx = list(init_batch.numpy())
    unlabeled_idx = list(set(range(len(dataset))) - set(train_idx))

    # Create the train and unlabeled subsets based on the index lists above. While the unlabeled set constructed here technically has labels, they
    # are only used when querying for labels. Hence, they only exist here for the sake of experimental design.
    cifar10_train = Subset(dataset, train_idx)
    cifar10_unlabeled = Subset(dataset, unlabeled_idx)

    net = Resnet18Classification(num_output=nclasses)

    # Define the training arguments to use.
    args = {'n_epoch': 500,  # Stop training after 300 epochs.
            'lr': 0.01,  # Use a learning rate of 0.01
            'batch_size': 250,  # Update the parameters using training batches of size 20
            'max_accuracy': 1.1,  # Stop training once the training accuracy has exceeded 0.99
            'min_diff_acc': -0.1,
            'window_size': 1000,
            'optimizer': 'adam',  # Use the stochastic gradient descent optimizer
            'device': "cuda" if torch.cuda.is_available() else "cpu",  # Use a GPU if one is available
            'isreset': False
            }

    # Create the training loop using our training dataset, provided model, and training arguments.
    # Train an initial model.
    dt = data_train(cifar10_train, copy.deepcopy(net), args)
    num_preds = torch.sum((labels[init_batch].unsqueeze(-1) == torch.arange(n_class)).float(), dim=0)
    trained_model = dt.train(weight=1. / torch.clip(num_preds, min=1e-6))
    trainer = PassiveResnet18Trainer()
    # trainer.model = trained_model
    trainer.train(imgs[init_batch], labels[init_batch])
    with torch.no_grad():
        pred = trainer.pred(imgs)
    model_acc = torch.mean((torch.argmax(pred, dim=-1) == labels).float()).data.cpu().numpy()
    majority_acc = torch.mean(
        (torch.argmax(pred, dim=-1) == labels).float()[labels == (n_class - 1)]).data.cpu().numpy()
    minority_acc = torch.mean(
        (torch.argmax(pred, dim=-1) == labels).float()[labels != (n_class - 1)]).data.cpu().numpy()
    wandb.log({"Model Accuracy": model_acc,
               "Num Queries": init_batch_size,
               "Per Class Min Queries": torch.min(num_preds).cpu().numpy(),
               "Per Class Max Queries": torch.max(num_preds).cpu().numpy(),
               "Majority Class Accuracy": majority_acc,
               "Minority Class Accuracy": minority_acc})

    # Go over the training dataset, getting the indices of all the rare-class examples
    rare_training_example_indices = []
    for index, (_, label) in enumerate(cifar10_train):
        if label in rare_classes:
            rare_training_example_indices.append(index)
    print(rare_training_example_indices)

    # Create a query set that contains only the rare-class examples of the training dataset
    rare_class_query_set = Subset(cifar10_train, rare_training_example_indices)

    # Define arguments for SMI
    selection_strategy_args = {'device': args['device'],  # Use the device used in training
                               'batch_size': args['batch_size'],  # Use the batch size used in training
                               'smi_function': smi_func,
                               # Use a facility location function, which captures representation information
                               'metric': 'cosine',  # Use cosine similarity when determining the likeness of two data points
                               'optimizer': 'LazyGreedy'
                               # When doing submodular maximization, use the lazy greedy optimizer
                               }

    # Create the SMI selection strategy. Note: We remove the labels from the unlabeled portion of CIFAR-10 that we created earlier.
    # In a practical application, one would not have these labels a priori.
    selection_strategy = SMI(cifar10_train, LabeledToUnlabeledDataset(cifar10_unlabeled), rare_class_query_set,
                             trained_model, nclasses, selection_strategy_args)

    # Do the selection, which will return the indices of the selected points with respect to the unlabeled dataset.
    budget = batch_size

    queried = list(init_batch.numpy())

    for _ in range(num_iters):
        selected_idx = selection_strategy.select(budget)
        queried = queried + [unlabeled_idx[i] for i in selected_idx]
        print(len(queried) - len(set(queried)))

        # Form a labeled subset of the unlabeled dataset. Again, we already have the labels,
        # so we simply take a subset. Note, however, that the selection was done without the
        # use of the labels and that we would normally not have these labels. Hence, the
        # following statement would usually require human effort to complete.
        smi_human_labeled_dataset = Subset(cifar10_unlabeled, selected_idx)

        # Create a new training dataset by concatenating what we have with the newly labeled points
        new_training_dataset = ConcatDataset([cifar10_train, smi_human_labeled_dataset])
        new_dt = data_train(new_training_dataset, copy.deepcopy(net), args)
        num_preds = torch.sum((labels[torch.from_numpy(np.array(queried))].unsqueeze(-1) == torch.arange(n_class)).float(), dim=0)
        new_trained_model = new_dt.train(weight=1. / torch.clip(num_preds, min=1e-6))

        unlabeled_idx = list(set(range(len(dataset))) - set(queried))
        cifar10_unlabeled = Subset(dataset, unlabeled_idx)
        # Go over the training dataset, getting the indices of all the rare-class examples
        rare_training_example_indices = []
        for index, (_, label) in enumerate(new_training_dataset):
            if label in rare_classes:
                rare_training_example_indices.append(index)
        rare_class_query_set = Subset(new_training_dataset, rare_training_example_indices)
        selection_strategy = SMI(new_training_dataset, LabeledToUnlabeledDataset(cifar10_unlabeled), rare_class_query_set,
                                 new_trained_model, nclasses, selection_strategy_args)

        trainer = PassiveResnet18Trainer()
        # trainer.model = new_trained_model
        trainer.train(imgs[torch.from_numpy(np.array(queried))], labels[torch.from_numpy(np.array(queried))])
        with torch.no_grad():
            pred = trainer.pred(imgs)
        model_acc = torch.mean((torch.argmax(pred, dim=-1) == labels).float()).data.cpu().numpy()
        majority_acc = torch.mean(
            (torch.argmax(pred, dim=-1) == labels).float()[labels == (n_class - 1)]).data.cpu().numpy()
        minority_acc = torch.mean(
            (torch.argmax(pred, dim=-1) == labels).float()[labels != (n_class - 1)]).data.cpu().numpy()
        wandb.log({"Model Accuracy": model_acc,
                   "Num Queries": len(queried),
                   "Per Class Min Queries": torch.min(num_preds).cpu().numpy(),
                   "Per Class Max Queries": torch.max(num_preds).cpu().numpy(),
                   "Majority Class Accuracy": majority_acc,
                   "Minority Class Accuracy": minority_acc})

    torch.save(trainer.model, open("similar%s_%s_%d_%d.model" % (smi_func, data, batch_size, seed), "wb"))

    wandb.finish()
