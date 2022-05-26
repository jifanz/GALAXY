import numpy as np
from torch.utils.data import DataLoader
from .strategy import Strategy
import pickle
from scipy.spatial.distance import cosine
import sys
import gc
from scipy.linalg import det
from scipy.linalg import pinv as inv
from copy import copy as copy
from copy import deepcopy as deepcopy
import torch
from torch import nn
import torchfile
from torch.autograd import Variable
import resnet
import vgg
import torch.optim as optim
import pdb
from torch.nn import functional as F
import argparse
import torch.nn as nn
from collections import OrderedDict
from scipy import stats
import time
import numpy as np
import scipy.sparse as sp
from itertools import product
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.utils.extmath import row_norms, squared_norm, stable_cumsum
from sklearn.utils.sparsefuncs_fast import assign_rows_csr
from sklearn.utils.sparsefuncs import mean_variance_axis
from sklearn.utils.validation import _num_samples
from sklearn.utils import check_array
from sklearn.utils import gen_batches
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.metrics.pairwise import rbf_kernel as rbf
#from sklearn.externals.six import string_types
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances
from dataclasses import dataclass
from typing import List

import torch
from toma import toma
from tqdm.auto import tqdm
import joint_entropy
import time

# kmeans ++ initialization
def init_centers(X, K):
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] >  newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2)/ sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        while ind in indsAll: ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    return indsAll

class mlpMod(nn.Module):
    def __init__(self, dim, nClasses, embSize=128, useNonLin=True):
        super(mlpMod, self).__init__()
        self.embSize = embSize
        self.dim = int(np.prod(dim))
        self.lm1 = nn.Linear(self.dim, embSize)
        self.lm2 = nn.Linear(embSize, embSize)
        self.linear = nn.Linear(embSize, nClasses, bias=False)
        self.useNonLin = useNonLin
        self.dropout = True
        self.do1 = nn.Dropout(0.5)
    def forward(self, x):
        x = x.view(-1, self.dim)
        if self.useNonLin: emb = F.relu(self.lm1(x))
        else: emb = self.lm1(x)
        out = self.linear(self.do1(emb))
        return out, emb
    def get_embedding_dim(self):
        return self.embSize


def compute_conditional_entropy(log_probs_N_K_C: torch.Tensor) -> torch.Tensor:
    N, K, C = log_probs_N_K_C.shape
    entropies_N = torch.empty(N, dtype=torch.double)
    pbar = tqdm(total=N, desc="Conditional Entropy", leave=False)

    @toma.execute.chunked(log_probs_N_K_C, 1024)
    def compute(log_probs_n_K_C, start: int, end: int):
        nats_n_K_C = log_probs_n_K_C * torch.exp(log_probs_n_K_C)

        entropies_N[start:end].copy_(-torch.sum(nats_n_K_C, dim=(1, 2)) / K)
        pbar.update(end - start)

    pbar.close()
    return entropies_N


def compute_entropy(log_probs_N_K_C: torch.Tensor) -> torch.Tensor:
    N, K, C = log_probs_N_K_C.shape
    entropies_N = torch.empty(N, dtype=torch.double)

    @toma.execute.chunked(log_probs_N_K_C, 1024)
    def compute(log_probs_n_K_C, start: int, end: int):
        mean_log_probs_n_C = torch.logsumexp(log_probs_n_K_C, dim=1) - math.log(K)
        nats_n_C = mean_log_probs_n_C * torch.exp(mean_log_probs_n_C)
        entropies_N[start:end].copy_(-torch.sum(nats_n_C, dim=1))

    return entropies_N

class BatchBALD(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(BatchBALD, self).__init__(X, Y, idxs_lb, net, handler, args)

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        if 'mlp' in str(self.clf.__class__):
            net = mlpMod(self.clf.lm1.in_features, self.clf.linear.out_features).cuda()
        if 'vgg' in str(type(self.clf)): net = vgg.VGG('VGG16', dropout=True).cuda()
        
        self.train(net=net)
        probs = self.predict_prob_dropout_split(self.X[idxs_unlabeled], self.Y[idxs_unlabeled], 10)
        pdb.set_trace()
        log_probs_N_K_C = probs.transpose(0, 1)
        N, K, C = log_probs_N_K_C.shape
        batch_size = min(n, N)

        candidate_indices = []
        candidate_scores = []
        conditional_entropies_N = compute_conditional_entropy(log_probs_N_K_C)
        batch_joint_entropy = joint_entropy.DynamicJointEntropy(N, batch_size - 1, K, C)
        scores_N = torch.empty(N, dtype=torch.double, pin_memory=torch.cuda.is_available())
        
        start = time.time()
        for i in range(batch_size):
            if i > 0:
                latest_index = candidate_indices[-1]
                batch_joint_entropy.add_variables(log_probs_N_K_C[latest_index : latest_index + 1])

            shared_conditinal_entropies = conditional_entropies_N[candidate_indices].sum()
            batch_joint_entropy.compute_batch(log_probs_N_K_C, output_entropies_B=scores_N)

            scores_N -= conditional_entropies_N + shared_conditinal_entropies
            scores_N[candidate_indices] = -float("inf")
            candidate_score, candidate_index = scores_N.max(dim=0)
            candidate_indices.append(candidate_index.item())
            candidate_scores.append(candidate_score.item())
            end = time.time()
            print(i, end - start)
            start = end


        return idxs_unlabeled[candidate_indices]
