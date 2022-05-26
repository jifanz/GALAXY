import numpy as np
import sys
import gc
import gzip
import pickle
# import openml
import os
import argparse
from dataset import get_dataset, get_handler
from model import get_net
import vgg
import resnet
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
import torch
import time
import pdb
from scipy.stats import zscore
import wandb
from src.model import Resnet18Classification
import os

os.environ["OMP_NUM_THREADS"] = '30'
os.environ["OPENBLAS_NUM_THREADS"] = '30'
os.environ["MKL_NUM_THREADS"] = '30'

from query_strategies import RandomSampling, BadgeSampling, \
    BaselineSampling, LeastConfidence, MarginSampling, \
    EntropySampling, CoreSet, ActiveLearningByLearning, \
    LeastConfidenceDropout, MarginSamplingDropout, EntropySamplingDropout, \
    KMeansSampling, KCenterGreedy, BALDDropout, CoreSet, \
    AdversarialBIM, AdversarialDeepFool, ActiveLearningByLearning, BaitSampling

# code based on https://github.com/ej0cl6/deep-active-learning and https://github.com/JordanAsh/badge
parser = argparse.ArgumentParser()
parser.add_argument('--alg', help='acquisition algorithm', type=str, default='rand')
parser.add_argument('--did', help='openML dataset index, if any', type=int, default=0)
parser.add_argument('--lr', help='learning rate', type=float, default=1e-2)
parser.add_argument('--paramScale', help='learning rate', type=float, default=1)
parser.add_argument('--model', help='model - resnet, vgg, or mlp', type=str, default='mlp')
parser.add_argument('--path', help='data path', type=str, default='data')
parser.add_argument('--data', help='dataset (non-openML)', type=str, default='')
parser.add_argument('--nQuery', help='number of points to query in a batch', type=int, default=100)
parser.add_argument('--nStart', help='number of points to start', type=int, default=100)
parser.add_argument('--nEnd', help='total number of points to query', type=int, default=50000)
parser.add_argument('--nEmb', help='number of embedding dims (mlp)', type=int, default=128)
parser.add_argument('--rounds', help='number of embedding dims (mlp)', type=int, default=0)
parser.add_argument('--trunc', help='number of embedding dims (mlp)', type=int, default=-1)
parser.add_argument('--btype', help='acquisition algorithm', type=str, default='min')
parser.add_argument('--modes', help='openML dataset index, if any', type=int, default=1)
parser.add_argument('--aug', help='do augmentation (for cifar)', type=int, default=0)
parser.add_argument('--lamb', help='lambda', type=float, default=1)
parser.add_argument('--fishIdentity', help='for ablation, setting fisher to be identity', type=int, default=0)
parser.add_argument('--fishInit', help='initialize selection with fisher on seen data', type=int, default=1)
parser.add_argument('--backwardSteps', help='openML dataset index, if any', type=int, default=1)
parser.add_argument('--dummy', help='dummy input for indexing replicates', type=int, default=1)
parser.add_argument('--unbalance', help='number of classes in an unbalanced dataset', type=int, default=None)
parser.add_argument("--wandb_name", type=str, default="")
opts = parser.parse_args()

data_name = opts.data if opts.data != 'CIFAR10' else 'CIFAR'
wandb.init(project="%s" % (
        data_name.lower() + ("" if opts.unbalance is None or data_name == "medmnist" else ("_unbalanced_%d" % opts.unbalance))),
           entity=opts.wandb_name,
           name="%s, Pretrained, Weighted, Batch Size %d, %s" % (opts.alg, opts.nQuery, opts.model))
print(opts, flush=True)

# parameters
NUM_INIT_LB = opts.nStart
NUM_QUERY = opts.nQuery
NUM_ROUND = int((opts.nEnd - NUM_INIT_LB) / opts.nQuery)
DATA_NAME = opts.data

# non-openml data defaults
args_pool = {'MNIST':
                 {'n_epoch': 10,
                  'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                  'loader_tr_args': {'batch_size': 64, 'num_workers': 1},
                  'loader_te_args': {'batch_size': 1000, 'num_workers': 1},
                  'optimizer_args': {'lr': 0.01, 'momentum': 0.5}},
             'FashionMNIST':
                 {'n_epoch': 10,
                  'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                  'loader_tr_args': {'batch_size': 64, 'num_workers': 1},
                  'loader_te_args': {'batch_size': 1000, 'num_workers': 1},
                  'optimizer_args': {'lr': 0.01, 'momentum': 0.5}},
             'SVHN':
                 {'n_epoch': 20, 'transform': transforms.Compose(
                     [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                  'loader_tr_args': {'batch_size': 250, 'num_workers': 1},
                  'loader_te_args': {'batch_size': 1000, 'num_workers': 1},
                  'optimizer_args': {'lr': 0.01, 'momentum': 0.5}},
             'CIFAR10':
                 {'n_epoch': 3, 'transform': transforms.Compose(
                     [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                  'loader_tr_args': {'batch_size': 250, 'num_workers': 1},
                  'loader_te_args': {'batch_size': 1000, 'num_workers': 1},
                  'optimizer_args': {'lr': 0.05, 'momentum': 0.3},
                  'transformTest': transforms.Compose(
                      [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])},
             'CIFAR100':
                 {'n_epoch': 3, 'transform': transforms.Compose(
                     [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                  'loader_tr_args': {'batch_size': 250, 'num_workers': 1},
                  'loader_te_args': {'batch_size': 1000, 'num_workers': 1},
                  'optimizer_args': {'lr': 0.05, 'momentum': 0.3},
                  'transformTest': transforms.Compose(
                      [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])},
             'medmnist':
                 {'n_epoch': 3, 'transform': transforms.Compose(
                     [transforms.Resize((32, 32)),
                      transforms.ToTensor(),
                      # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                  'loader_tr_args': {'batch_size': 100, 'num_workers': 1},
                  'loader_te_args': {'batch_size': 1000, 'num_workers': 1},
                  'optimizer_args': {'lr': 0.05, 'momentum': 0.3},
                  'transformTest': transforms.Compose(
                      [transforms.Resize((32, 32)),
                       transforms.ToTensor(),
                       # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])},
             }
if opts.aug == 0:
    args_pool['CIFAR10']['transform'] = args_pool['CIFAR10']['transformTest']  # remove data augmentation
args_pool['MNIST']['transformTest'] = args_pool['MNIST']['transform']
args_pool['FashionMNIST']['transformTest'] = args_pool['FashionMNIST']['transform']
args_pool['SVHN']['transformTest'] = args_pool['SVHN']['transform']
args_pool['CIFAR100']['transform'] = args_pool['CIFAR100']['transformTest']
args_pool['medmnist']['transform'] = args_pool['medmnist']['transformTest']

if opts.did == 0: args = args_pool[DATA_NAME]
if not os.path.exists(opts.path):
    os.makedirs(opts.path)

# load openml dataset if did is supplied
if opts.did > 0:
    data = pickle.load(open('oml/data_' + str(opts.did) + '.pk', 'rb'))['data']
    X = np.asarray(data[0])
    y = np.asarray(data[1])
    y = LabelEncoder().fit(y).transform(y)
    opts.nClasses = int(max(y) + 1)
    nSamps, opts.dim = np.shape(X)
    testSplit = .1
    inds = np.random.permutation(nSamps)
    X = X[inds]
    y = y[inds]

    split = int((1. - testSplit) * nSamps)
    while True:
        inds = np.random.permutation(split)
        if len(inds) > 50000: inds = inds[:50000]
        X_tr = X[:split]
        X_tr = X_tr[inds]
        X_tr = torch.Tensor(X_tr)

        y_tr = y[:split]
        y_tr = y_tr[inds]
        Y_tr = torch.Tensor(y_tr).long()

        X_te = torch.Tensor(X[split:])
        Y_te = torch.Tensor(y[split:]).long()

        if len(np.unique(Y_tr)) == opts.nClasses: break

    args = {'transform': transforms.Compose([transforms.ToTensor()]),
            'n_epoch': 10,
            'loader_tr_args': {'batch_size': 128, 'num_workers': 1},
            'loader_te_args': {'batch_size': 1000, 'num_workers': 1},
            'optimizer_args': {'lr': 0.01, 'momentum': 0},
            'transformTest': transforms.Compose([transforms.ToTensor()])}
    handler = get_handler('other')

# load non-openml dataset
else:
    X_tr, Y_tr, X_te, Y_te = get_dataset(DATA_NAME, opts.path, unbalance=opts.unbalance)
    opts.dim = np.shape(X_tr)[1:]
    handler = get_handler(opts.data)
    opts.nClasses = int(max(Y_tr) + 1)
    print("Number of Classes:", opts.nClasses)

if opts.trunc != -1:
    inds = np.random.permutation(len(X_tr))[:opts.trunc]
    X_tr = X_tr[inds]
    Y_tr = Y_tr[inds]
    inds = torch.where(Y_tr < 10)[0]
    X_tr = X_tr[inds]
    Y_tr = Y_tr[inds]
    opts.nClasses = int(max(Y_tr) + 1)

args['lr'] = opts.lr
args['modelType'] = opts.model
args['fishIdentity'] = opts.fishIdentity
args['fishInit'] = opts.fishInit
args['lamb'] = opts.lamb
args['backwardSteps'] = opts.backwardSteps

# start experiment
n_pool = len(Y_tr)
n_test = len(Y_te)
print('number of labeled pool: {}'.format(NUM_INIT_LB), flush=True)
print('number of unlabeled pool: {}'.format(n_pool - NUM_INIT_LB), flush=True)
print('number of testing pool: {}'.format(n_test), flush=True)

# generate initial labeled pool
idxs_lb = np.zeros(n_pool, dtype=bool)
idxs_lb[np.random.permutation(n_pool)[:NUM_INIT_LB]] = True


# linear model class
class linMod(nn.Module):
    def __init__(self, dim=28):
        super(linMod, self).__init__()
        self.dim = dim
        self.lm = nn.Linear(dim, opts.nClasses)

    def forward(self, x):
        x = x.view(-1, self.dim)
        out = self.lm(x)
        return out, x

    def get_embedding_dim(self):
        return self.dim


# mlp model class
class mlpMod(nn.Module):
    def __init__(self, dim, embSize=128, useNonLin=True):
        super(mlpMod, self).__init__()
        self.embSize = embSize
        self.dim = int(np.prod(dim))
        self.lm1 = nn.Linear(self.dim, embSize)
        self.lm2 = nn.Linear(embSize, embSize)
        self.linear = nn.Linear(embSize, opts.nClasses, bias=False)
        self.useNonLin = useNonLin

    def forward(self, x):
        x = x.view(-1, self.dim)
        if self.useNonLin:
            emb = F.relu(self.lm1(x))
        else:
            emb = self.lm1(x)
        out = self.linear(emb)
        return out, emb

    def get_embedding_dim(self):
        return self.embSize


# load specified network
if opts.model == 'mlp':
    net = mlpMod(opts.dim, embSize=opts.nEmb)
elif opts.model == 'resnet':
    net = Resnet18Classification(num_output=opts.nClasses, ret_emb=True)
    resnet.ResNet18()
elif opts.model == 'vgg':
    net = vgg.VGG('VGG16')
elif opts.model == 'lin':
    dim = np.prod(list(X_tr.shape[1:]))
    net = linMod(dim=dim)
else:
    print('choose a valid model - mlp, resnet, or vgg', flush=True)
    raise ValueError

if opts.did > 0 and opts.model != 'mlp':
    print('openML datasets only work with mlp', flush=True)
    raise ValueError

if type(X_tr[0]) is not np.ndarray:
    X_tr = X_tr.numpy()

# set up the specified sampler
if opts.alg == 'rand':  # random sampling
    strategy = RandomSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'bait':  # bait sampling
    strategy = BaitSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'conf':  # confidence-based sampling
    strategy = LeastConfidence(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'marg':  # margin-based sampling
    strategy = MarginSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'badge':  # batch active learning by diverse gradient embeddings
    strategy = BadgeSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'coreset':  # coreset sampling
    strategy = CoreSet(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'entropy':  # entropy-based sampling
    strategy = EntropySampling(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'baseline':  # badge but with k-DPP sampling instead of k-means++
    strategy = BaselineSampling(X_tr, Y_tr, idxs_lb, net, handler, args)
elif opts.alg == 'albl':  # active learning by learning
    albl_list = [LeastConfidence(X_tr, Y_tr, idxs_lb, net, handler, args),
                 CoreSet(X_tr, Y_tr, idxs_lb, net, handler, args)]
    strategy = ActiveLearningByLearning(X_tr, Y_tr, idxs_lb, net, handler, args, strategy_list=albl_list, delta=0.1)
else:
    print('choose a valid acquisition function', flush=True)
    raise ValueError

# print info
if opts.did > 0: DATA_NAME = 'OML' + str(opts.did)
print(DATA_NAME, flush=True)
print(type(strategy).__name__, flush=True)

if type(X_te) == torch.Tensor: X_te = X_te.numpy()

# round 0 accuracy
strategy.train()
P = strategy.predict(X_te, Y_te)
acc = np.zeros(NUM_ROUND + 1)
acc[0] = 1.0 * (Y_te == P).sum().item() / len(Y_te)
print(str(opts.nStart) + '\ttesting accuracy {}'.format(acc[0]), flush=True)
scores = strategy.predict_prob(X_tr, Y_tr)
queried = torch.from_numpy(np.arange(n_pool)[idxs_lb])
model_acc = torch.mean((torch.argmax(scores, dim=-1) == Y_tr).float()).data.cpu().numpy()
majority_acc = torch.mean(
    (torch.argmax(scores, dim=-1) == Y_tr).float()[Y_tr == (opts.nClasses - 1)]).data.cpu().numpy()
minority_acc = torch.mean(
    (torch.argmax(scores, dim=-1) == Y_tr).float()[Y_tr != (opts.nClasses - 1)]).data.cpu().numpy()
num_preds = torch.sum((Y_tr[queried].unsqueeze(-1) == torch.arange(opts.nClasses)).float(), dim=0)
wandb.log({"Num Queries": sum(idxs_lb),
           "Model Accuracy": model_acc,
           "Per Class Min Queries": torch.min(num_preds).cpu().numpy(),
           "Per Class Max Queries": torch.max(num_preds).cpu().numpy(),
           "Majority Class Accuracy": majority_acc,
           "Minority Class Accuracy": minority_acc})
print((majority_acc + minority_acc) / 2)

for rd in range(1, NUM_ROUND + 1):
    print('Round {}'.format(rd), flush=True)
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

    # query
    output = strategy.query(NUM_QUERY)
    q_idxs = output
    idxs_lb[q_idxs] = True

    # update
    strategy.update(idxs_lb)
    strategy.train(verbose=False)

    # round accuracy
    P = strategy.predict(X_te, Y_te)
    acc[rd] = 1.0 * (Y_te == P).sum().item() / len(Y_te)
    print(str(sum(idxs_lb)) + '\t' + 'testing accuracy {}'.format(acc[rd]), flush=True)
    scores = strategy.predict_prob(X_tr, Y_tr)
    queried = torch.from_numpy(np.arange(n_pool)[idxs_lb])
    model_acc = torch.mean((torch.argmax(scores, dim=-1) == Y_tr).float()).data.cpu().numpy()
    majority_acc = torch.mean(
        (torch.argmax(scores, dim=-1) == Y_tr).float()[Y_tr == (opts.nClasses - 1)]).data.cpu().numpy()
    minority_acc = torch.mean(
        (torch.argmax(scores, dim=-1) == Y_tr).float()[Y_tr != (opts.nClasses - 1)]).data.cpu().numpy()
    num_preds = torch.sum((Y_tr[queried].unsqueeze(-1) == torch.arange(opts.nClasses)).float(), dim=0)
    wandb.log({"Num Queries": sum(idxs_lb),
               "Model Accuracy": model_acc,
               "Per Class Min Queries": torch.min(num_preds).cpu().numpy(),
               "Per Class Max Queries": torch.max(num_preds).cpu().numpy(),
               "Majority Class Accuracy": majority_acc,
               "Minority Class Accuracy": minority_acc})

    if sum(~strategy.idxs_lb) < opts.nQuery: break
    if opts.rounds > 0 and rd == (opts.rounds - 1): break

torch.save(strategy.clf, open("%s_%s_%d.model" % (opts.alg,
                                                  data_name.lower() + ("" if opts.unbalance is None else (
                                                          "_unbalanced_%d" % opts.unbalance)), NUM_QUERY), "wb"))

wandb.finish()
