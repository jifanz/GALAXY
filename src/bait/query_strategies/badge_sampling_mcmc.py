import numpy as np
from torch.utils.data import DataLoader
from .strategy import Strategy
import pickle
import gc
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
from sklearn.externals.six import string_types
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances

# kmeans ++ initialization
def updateInv(currentInv, vec, inc=True):
    dim = currentInv.shape[-1]
    rank = vec.shape[-2]
    C = torch.eye(rank).cuda()
    if inc == False: C = -1 * C
    innerInv = torch.inverse(C + vec @ currentInv @ vec.transpose(-2, -1))
    newInvs = currentInv - currentInv @ vec.transpose(-2, -1) @ innerInv @ vec @ currentInv
    return newInvs



def init_centers(X, K, fisher, iterates):
    dim = X.shape[-1]
    fisher = fisher.cuda()
    currentInds = np.random.permutation(len(X))[:K]
    currentMat = torch.sum(X[currentInds].transpose(1,2) @ X[currentInds], 0) + 0.01 * torch.eye(dim)
    currentInv = torch.inverse(currentMat).cuda()
    currentTrace = torch.trace(currentInv @ fisher).item()

    nCandidates = 100
    nCandidates = min(nCandidates, len(X) - K)
    if len(X) == K: return np.asarray(currentInds)
    for i in range(10000):
         
        switchInd = np.random.choice(currentInds)
        removedInv = updateInv(currentInv, X[switchInd].cuda(), inc=False)

        possibleInds = np.asarray([True for i in range(len(X))])
        possibleInds[currentInds] = False
        candidates = np.where(possibleInds)[0][np.random.permutation(len(X) - K)][:nCandidates]

        newInvs = updateInv(removedInv, X[candidates].cuda())
        newTraces = np.abs(torch.sum(torch.sum((newInvs * fisher.t()), 1), 1).cpu().numpy())
        newTraces = np.asarray(list(newTraces) + [currentTrace])
        dist = -1 * newTraces 
        dist = dist - np.min(dist) + 1e-7
        dist = dist / np.sum(dist)
        samp = stats.rv_discrete(values=(np.arange(nCandidates + 1), dist)).rvs(size=1)[0]
        if samp != nCandidates:
            currentInds[np.where(currentInds == switchInd)[0][0]] = candidates[samp]
            currentInv = newInvs[samp]
            currentTrace = newTraces[samp]
        print(i, currentTrace, flush=True)

    return np.asarray(currentInds)

class BadgeSamplingAk(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(BadgeSamplingAk, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.badgeModes = args['badgeModes']
        self.badgeSamps = args['badgeSamps'] 
        self.badgeOpt = args['badgeOpt'] 
        self.badgeModal = args['badgeModal']

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        nEns = self.badgeModes
        nSamps = self.badgeSamps
        opt = self.badgeOpt


        print('getting fisher matrix ...', flush=True)
        xt = self.get_exp_grad_embedding(self.X, self.Y)
        if False:
            batchSize = 1000
            nClass = torch.max(self.Y).item() + 1
            fisher = torch.zeros(xt.shape[-1], xt.shape[-1])
            for i in range(int(np.ceil(len(self.X) / batchSize))):
                xt_ = xt[i * batchSize : (i + 1) * batchSize].cuda()
                op = torch.sum(torch.matmul(xt_.transpose(1,2), xt_) / (nClass * len(xt_)), 0).cpu()
                fisher = fisher + op

            print('getting H^-1 GG^T H^-1 ...', flush=True)
            epochs = self.train_val(shrink=False, valFrac=0.1, opt=opt)
            weights, iterates = self.get_dist(epochs, nEns=nEns, opt=opt)
            mask = []
            for names, p in self.clf.named_parameters():
                param = torch.zeros_like(p).flatten()
                if 'linear' in names: param = param + 1
                mask.append(param)
            mask = torch.cat(mask)
            maskInds = torch.where(mask == 1)[0]
        
            for i in range(len(weights)):
                weights[i] = weights[i][maskInds]
                iterates[i] = iterates[i][:, maskInds]

        #chosen = init_centers(xt[idxs_unlabeled], n, fisher, iterates[0])
        chosen = init_centers(xt[idxs_unlabeled], n, torch.eye(xt.shape[-1]), [])
        return idxs_unlabeled[chosen]

