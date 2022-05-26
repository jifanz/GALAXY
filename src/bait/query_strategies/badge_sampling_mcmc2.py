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
#from sklearn.externals.six import string_types
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances

# kmeans ++ initialization
def kmpp(X, K):
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


def updateInv_einsum(currentInv, vec, inc=True):
    rank = vec.shape[-2]

    U = vec.transpose(-2, -1)
    V = vec
    tmp   = torch.einsum('qab,bc,qcd->qad',
                      V, currentInv, U)
    C = torch.eye(rank)
    if inc == False: C = -1 * C
    innerInv = torch.inverse(C + tmp)

    tmp   = torch.einsum('ab,qbc,qcd,qde,ef->qaf',
                      currentInv, U, innerInv, V, currentInv)
    return currentInv - tmp

def updateInv(currentInv, vec, inc=True):
    dim = currentInv.shape[-1]
    rank = vec.shape[-2]
    C = torch.eye(rank).cuda()
    if inc == False: C = -1 * C
    innerInv = torch.inverse(C + vec @ currentInv @ vec.transpose(-2, -1))
    newInvs = currentInv - currentInv @ vec.transpose(-2, -1) @ innerInv @ vec @ currentInv
    return newInvs

def batchOuterProdDet(X, A, batchSize):
    dets = []
    rank = X.shape[-2]
    batches = int(np.ceil(len(X) / batchSize))
    for i in range(batches):
        x = X[i * batchSize : (i + 1) * batchSize].cuda()
        outerProds = (torch.matmul(torch.matmul(x, A), torch.transpose(x, 1, 2))).detach()
        newDets = (torch.det(outerProds + torch.eye(rank).cuda())).detach()
        dets.append(newDets.cpu().numpy())
        outerProds = outerProds.cpu()
        del outerProds
        torch.cuda.empty_cache()
        gc.collect()

    dets = np.abs(np.concatenate(dets))
    dets[np.isinf(dets)] = np.finfo('float32').max
    dist = dets / np.finfo('float32').max
    dist -= np.min(dist)
    dist /= np.sum(dist)
    return dist, dets

def hes_centers(X, K, invSeed=[], unlabWeight=-1):

    numEmbs = len(X)
    indsAll = []
    currentDet = 0
    dim = X[0].shape[-1]
    rank = X[0].shape[-2]

    if len(invSeed) == 0: currentInv = torch.eye(dim).cuda() * 100.
    else:
        if unlabWeight == -1: cov = (invSeed[-1].t() @ invSeed[-1]).cuda() / len(invSeed[-1])
        else: cov = (invSeed[-1].t() @ invSeed[-1]).cuda() / len(invSeed[-1]) * (1 - unlabWeight)
        currentInv = torch.inverse(cov + 0.01 * torch.eye(dim).cuda()).detach()
        torch.cuda.empty_cache()
        gc.collect()

    for i in range(K):

        if unlabWeight == -1: Xtensor = torch.Tensor(X[np.random.randint(numEmbs)]) / np.sqrt(K)
        else: Xtensor = torch.Tensor(X[np.random.randint(numEmbs)]) / np.sqrt(K) * np.sqrt(unlabWeight)
        dist, dets = batchOuterProdDet(Xtensor, currentInv, 50000)
        sampler = stats.rv_discrete(values=(np.arange(len(Xtensor)), dist))
        ind = sampler.rvs(size=1)[0]
        while ind in indsAll:
            ind = sampler.rvs(size=1)[0]
        indsAll.append(ind)

        # update inverse
        samp = Xtensor[ind].cuda()
        innerInv = torch.inverse(torch.eye(rank).cuda() + samp @ currentInv @ samp.t()).detach()
        currentInv = (currentInv - currentInv @ samp.t() @ innerInv @ samp @ currentInv).detach()

        torch.cuda.empty_cache()
        gc.collect()
        print(i, dets[ind].item(), flush=True)


    return indsAll

def fromScratch(X, K, fisher, iterates):
    dim = X.shape[-1]
    fisher = fisher.cuda()
    currentInds= []
    currentMat = 0.01 * torch.eye(dim)
    currentInv = torch.inverse(currentMat).cuda()
    currentTrace = torch.trace(currentInv @ fisher).item()

    batchSize = 512
    for j in range(K):
        newTraces = []
        for i in range(int(np.ceil(len(X)/batchSize))):
            startInd = i * batchSize
            endInd = (i + 1) * batchSize
            newInvs = updateInv(currentInv, X[startInd:endInd].cuda() / np.sqrt(K))
            newTraces.append(torch.sum(torch.sum((newInvs * fisher.t()), 1), 1).cpu().numpy())

        newTraces = np.concatenate(newTraces)
        dist = -1 * newTraces
        dist = dist - np.min(dist)
        dist = dist + 1e-10
        dist = dist / np.sum(dist)
        samp = stats.rv_discrete(values=(np.arange(len(X)), dist)).rvs(size=1)[0]
        while samp in currentInds: samp = stats.rv_discrete(values=(np.arange(len(X)), dist)).rvs(size=1)[0]
        currentInds.append(samp)
        currentInv = updateInv(currentInv, X[samp].cuda() / np.sqrt(K))
        print(j, flush=True)
    return currentInds



def init_centers(X, K, fisher, iterates, rounds, currentInds=[], sample=False):
    dim = X.shape[-1]
    fisher = fisher.cuda()
    if len(currentInds) == 0: currentInds = np.random.permutation(len(X))[:K]
    currentMat = (torch.sum(X[currentInds].transpose(1,2) @ X[currentInds], 0)) / K + 0.01 * torch.eye(dim)
    currentInv = torch.inverse(currentMat).cuda()
    currentTrace = torch.trace(currentInv @ fisher).item()

    nCandidates = 512
    nCandidates = min(nCandidates, len(X) - K)
    if len(X) == K: return np.asarray(currentInds)
    for i in range(rounds):
        switchInd = np.random.choice(currentInds)
        removedInv = updateInv(currentInv, X[switchInd].cuda() / np.sqrt(K), inc=False)
       
        possibleInds = np.asarray([True for i in range(len(X))])
        possibleInds[currentInds] = False
        candidates = np.where(possibleInds)[0][np.random.permutation(len(X) - K)][:nCandidates]

        newInvs = updateInv(removedInv, X[candidates].cuda() / np.sqrt(K))
        newTraces = torch.sum(torch.sum((newInvs * fisher.t()), 1), 1).cpu().numpy()

        currentInd = switchInd

        newTraces = np.asarray(list(newTraces) + [currentTrace])
        dist = -1 * newTraces
        dist = dist - np.min(dist)
        dist = dist + 1e-10
        dist = dist / np.sum(dist)
        samp = stats.rv_discrete(values=(np.arange(nCandidates + 1), dist)).rvs(size=1)[0]
        if not sample: samp = np.argmax(dist)
        if samp != nCandidates:
            currentInds[np.where(currentInds == switchInd)[0][0]] = candidates[samp]
            currentInv = newInvs[samp]
            currentTrace = newTraces[samp]
        del newTraces
        torch.cuda.empty_cache()
        gc.collect()

        #for j in range(nCandidates):
        #    alpha = currentTrace / newTraces[j]
            #alpha = alpha ** 2
            #alpha = np.exp(currentTrace - newTraces[j])
        #    if alpha > 1:# or np.random.rand() < alpha:
        #        currentInd = candidates[j]
        #        currentInv = newInvs[j]
        #        currentTrace = newTraces[j]
        #currentInds[np.where(currentInds == switchInd)[0][0]] = currentInd
        # print(torch.det(torch.inverse(torch.sum(X[currentInds].transpose(1,2) @ X[currentInds], 0) + 0.01 * torch.eye(dim)).cuda() @ (torch.sum(X[currentInds].transpose(1,2) @ X[currentInds], 0) + 0.01 * torch.eye(dim)).cuda()))
        print(i, currentTrace, flush=True)

    return np.asarray(currentInds)

class BadgeSamplingAk(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(BadgeSamplingAk, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.badgeModes = args['badgeModes']
        self.badgeSamps = args['badgeSamps'] 
        self.badgeOpt = args['badgeOpt'] 
        self.badgeModal = args['badgeModal']
        
        
        self.badgeSeed = args['badgeSeed']
        self.badgeRounds = args['badgeRounds']
        self.badgeSample = args['badgeSample']
        self.badgePosterior = args['badgePosterior']
        self.jason = args['jason']

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        nEns = self.badgeModes
        nSamps = self.badgeSamps
        opt = self.badgeOpt
        if self.badgePosterior == 1: nEns = 10
        else: nEns =1
        if False:
            print('getting H^-1 GG^T H^-1 ...', flush=True)
            epochs = self.train_val(valFrac=0.1, opt=opt)
            weights, iterates, optimizer = self.get_dist(epochs, nEns=nEns, opt=opt)
            mask = []
            for names, p in self.clf.named_parameters():
                param = torch.zeros_like(p).flatten()
                if 'linear' in names: param = param + 1
                mask.append(param)
            mask = torch.cat(mask)
            maskInds = torch.where(mask == 1)[0]

            # finish training network
            print('training from minimum ...', flush=True)
            self.clf = self.getNet(weights[-1]).cuda()

            self.train(reset=False, optimizer=0)

            if False:
                print('getting fisher matrix ...', flush=True)
                if self.badgePosterior == 1:
                    posterior = self.getPosterior(weights, iterates, self.X, self.Y, nSamps=25)
                    xt = self.get_exp_grad_embedding(self.X, self.Y, probs=posterior)
                else: xt = self.get_exp_grad_embedding(self.X, self.Y)
            
            for i in range(len(weights)):
                weights[i] = weights[i][maskInds]
                iterates[i] = iterates[i][:, maskInds]
            
        xt = self.get_exp_grad_embedding(self.X, self.Y)    
        if self.jason == 0: iterates = []
        fisher = torch.eye(xt.shape[-1])
        if False:
            batchSize = 512
            nClass = torch.max(self.Y).item() + 1
            fisher = torch.zeros(xt.shape[-1], xt.shape[-1])
            for i in range(int(np.ceil(len(self.X) / batchSize))):
                xt_ = xt[i * batchSize : (i + 1) * batchSize].cuda()
                op = torch.sum(torch.matmul(xt_.transpose(1,2), xt_) / len(xt), 0).cpu()
                fisher = fisher + op
                xt_ = xt_.cpu()
                del xt_

        #chosen = init_centers(xt[idxs_unlabeled], n, fisher, iterates[0])
        if True:
            if self.badgeSeed == 'hes':
                if self.jason == 2: unlabWeight = 0.5
                if self.jason == 3: unlabWeight = n / (np.sum(self.idxs_lb) + n)
                if self.jason > 1: seedInds = hes_centers([xt[idxs_unlabeled]], n, invSeed=iterates, unlabWeight=unlabWeight)
                else: seedInds = hes_centers([xt[idxs_unlabeled]], n, invSeed=iterates)
            if self.badgeSeed == 'kmeans':
                gradEmbedding = self.get_grad_embedding(self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled]).numpy()
                seedInds = kmpp(gradEmbedding, n)
            sample = False
            if self.badgeSample == 1: sample = True
            chosen = init_centers(xt[idxs_unlabeled], n, fisher, iterates, self.badgeRounds, currentInds=seedInds, sample=sample)
        else:
            chosen = init_centers(xt[idxs_unlabeled], n, fisher, iterates, self.badgeRounds)
            #chosen = fromScratch(xt[idxs_unlabeled], n, fisher, iterates)
        return idxs_unlabeled[chosen]

