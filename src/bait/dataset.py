import numpy as np
import pdb
import torch
from torchvision import datasets
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from medmnist import INFO, PathMNIST


def get_dataset(name, path, unbalance=None):
    if name == 'MNIST':
        return get_MNIST(path)
    elif name == 'FashionMNIST':
        return get_FashionMNIST(path)
    elif name == 'SVHN':
        return get_SVHN(path, unbalance=unbalance)
    elif name == 'CIFAR10':
        return get_CIFAR10(path, unbalance=unbalance)
    elif name == 'CIFAR100':
        return get_CIFAR100(path, unbalance=unbalance)
    elif name == 'medmnist':
        return get_medmnist(path, unbalance=unbalance)


def addNoise(labels, noiseLabels=[1,2], noiseCeil=0.5):
    labels = labels.numpy()
    nLabs = np.max(labels) + 1
    count = 0
    for y in labels:
        if y in noiseLabels and np.random.rand() < noiseCeil:
            newLabel = np.random.randint(nLabs)
            while newLabel == y: newLabel = np.random.randint(nLabs)
            labels[count] = newLabel
        count += 1
    return torch.Tensor(labels)

def get_MNIST(path):
    raw_tr = datasets.MNIST(path + '/MNIST', train=True, download=True)
    raw_te = datasets.MNIST(path + '/MNIST', train=False, download=True)
    X_tr = raw_tr.data
    Y_tr = raw_tr.targets
    X_te = raw_te.data
    Y_te = raw_te.targets
    return X_tr, Y_tr, X_te, Y_te

def get_FashionMNIST(path):
    raw_tr = datasets.FashionMNIST(path + '/FashionMNIST', train=True, download=True)
    raw_te = datasets.FashionMNIST(path + '/FashionMNIST', train=False, download=True)
    X_tr = raw_tr.train_data
    Y_tr = raw_tr.train_labels
    X_te = raw_te.test_data
    Y_te = raw_te.test_labels
    return X_tr, Y_tr, X_te, Y_te

def get_SVHN(path, unbalance=None):
    data_tr = datasets.SVHN(path + '/SVHN', split='train', download=True)
    data_te = datasets.SVHN(path +'/SVHN', split='test', download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(data_tr.labels)
    X_te = data_te.data
    Y_te = torch.from_numpy(data_te.labels)
    if unbalance is not None:
        Y_tr, Y_te = torch.clip(Y_tr, max=unbalance - 1), torch.clip(Y_te, max=unbalance - 1)
    return X_tr, Y_tr, X_te, Y_te

def get_CIFAR10(path, unbalance=None):
    data_tr = datasets.CIFAR10(path + '/CIFAR10', train=True, download=True)
    data_te = datasets.CIFAR10(path + '/CIFAR10', train=False, download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(np.array(data_tr.targets))
    X_te = data_te.data
    Y_te = torch.from_numpy(np.array(data_te.targets))
    if unbalance is not None:
        Y_tr, Y_te = torch.clip(Y_tr, max=unbalance - 1), torch.clip(Y_te, max=unbalance - 1)
    return X_tr, Y_tr, X_te, Y_te

def get_CIFAR100(path, unbalance=None):
    data_tr = datasets.CIFAR100(path + '/CIFAR100', train=True, download=True)
    data_te = datasets.CIFAR100(path + '/CIFAR100', train=False, download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(np.array(data_tr.targets))
    X_te = data_te.data
    Y_te = torch.from_numpy(np.array(data_te.targets))
    if unbalance is not None:
        Y_tr, Y_te = torch.clip(Y_tr, max=unbalance - 1), torch.clip(Y_te, max=unbalance - 1)
    return X_tr, Y_tr, X_te, Y_te

def get_medmnist(path, unbalance=None):
    data_tr = PathMNIST('train', root='./', download=True, transform=None)
    data_te = PathMNIST('test', root='./', download=True, transform=None)
    X_tr = data_tr.imgs
    Y_tr = torch.from_numpy(np.array(data_tr.labels)).squeeze()
    X_te = data_te.imgs
    Y_te = torch.from_numpy(np.array(data_te.labels)).squeeze()
    Y_tr, Y_te = (Y_tr != 7).long(), (Y_te != 7).long()
    print(X_tr.shape)
    print(Y_tr.size())
    return X_tr, Y_tr, X_te, Y_te


def get_handler(name):
    if name == 'MNIST':
        return DataHandler3
    elif name == 'FashionMNIST':
        return DataHandler3
    elif name == 'SVHN':
        return DataHandler2
    elif name == 'CIFAR10':
        return DataHandler3
    elif name == 'CIFAR100':
        return DataHandler3
    elif name == 'medmnist':
        return DataHandler3
    else:
        return DataHandler4

class DataHandler1(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x.numpy(), mode='L')
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class DataHandler2(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(np.transpose(x, (1, 2, 0)))
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class DataHandler3(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x)
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class DataHandler4(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        return x, y, index

    def __len__(self):
        return len(self.X)
