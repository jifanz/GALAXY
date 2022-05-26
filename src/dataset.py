from torchvision.datasets import ImageNet
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
import numpy as np
import torch
import pickle
from src.hyperparam import *
import medmnist
from medmnist import INFO, PathMNIST


def get_data(ret_dataset=False):
    if data == "cifar":
        dataset_class = datasets.cifar.CIFAR10
    elif data == "cifar100":
        dataset_class = datasets.cifar.CIFAR100
    elif data[:15] == "svhn_unbalanced":
        dataset_class = lambda root, train, download, transform: datasets.svhn.SVHN(root,
                                                                                    split="train" if train else "test",
                                                                                    download=download,
                                                                                    transform=transform)
    elif data[:16] == "cifar_unbalanced":
        dataset_class = datasets.cifar.CIFAR10
    elif data[:19] == "cifar100_unbalanced":
        dataset_class = datasets.cifar.CIFAR100
    elif data == "svhn":
        dataset_class = lambda root, train, download, transform: datasets.svhn.SVHN(root,
                                                                                    split="train" if train else "test",
                                                                                    download=download,
                                                                                    transform=transform)
    elif data == "medmnist":
        T = transforms.Compose(
            [transforms.Resize((32, 32)),
             transforms.ToTensor(),
             # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        def dataset_class(root, train, download, transform):
            return PathMNIST("train" if train else "test", transform=T, download=download, root=root)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = dataset_class("./data/", train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=40)
    imgs, labels = next(iter(loader))
    if data[:19] == "cifar100_unbalanced" or data[:16] == "cifar_unbalanced":
        dataset.targets = [min(label, n_class - 1) for label in dataset.targets]
        labels = torch.clip(labels, max=n_class - 1)
    elif data[:15] == "svhn_unbalanced":
        dataset.labels = np.clip(dataset.labels, a_min=None, a_max=n_class - 1)
        labels = torch.clip(labels, max=n_class - 1)
    elif data == "medmnist":
        dataset.labels = (dataset.labels != 7).astype(int)[:, 0]
        labels = (labels != 7).long()
        labels = labels.squeeze()
    return (imgs, labels, dataset) if ret_dataset else (imgs, labels)


if __name__ == "__main__":
    imgs, labels = get_data()
    print(labels.size())
    print(torch.sum(labels))
