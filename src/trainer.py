from src.graph import *
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from src.model import train_resnet18, Resnet18Classification, train_mlp
from src.linear_graph import create_linear_from_list
from src.s2algorithm import s2query, bisection_query
import matplotlib.pyplot as plt
from src.utils import calculate_auc, calculate_mAP
import wandb


class PassiveResnet18Trainer:
    def __init__(self, batch_size=250):
        self.batch_size = batch_size

    def train(self, imgs, labels, *args, n_class=None, **kwargs):
        self.model = train_resnet18(imgs, labels, batch_size=self.batch_size, n_class=n_class)

    def pred(self, imgs, ret_features=False, ret_margins=False):
        preds = []
        features = []
        min_margins = []
        per_class_margins = []
        for i in range(int(np.ceil(imgs.size(0) / 250.))):
            img_batch = imgs[i * 250: min((i + 1) * 250, imgs.size(0))].cuda()
            if ret_features:
                pred, f = self.model(img_batch, ret_features=True)
                preds.append(pred)
                features.append(f)
            elif ret_margins:
                pred, f = self.model(img_batch, ret_features=True)
                min_margin, per_class_margin = self.compute_margins(img_batch, pred, f)
                preds.append(pred)
                min_margins.append(min_margin)
                per_class_margins.append(per_class_margin)
            else:
                preds.append(self.model(img_batch))
        preds = torch.cat(preds, dim=0).squeeze(-1)
        if ret_features:
            features = torch.cat(features, dim=0).squeeze(-1)
            return (preds, features) if imgs.is_cuda else (preds.cpu(), features.cpu())
        elif ret_margins:
            min_margins = torch.cat(min_margins, dim=0)
            per_class_margins = torch.cat(per_class_margins, dim=0)
            return (preds, min_margins, per_class_margins) if imgs.is_cuda else (
            preds.cpu(), min_margins.cpu(), per_class_margins.cpu())
        else:
            return preds if imgs.is_cuda else preds.cpu()

    def compute_margins(self, x, logits, embedding):
        weight = self.model.linear.weight
        bias = self.model.linear.bias

        predictions = logits.max(dim=1).indices

        # weight.shape = (C, M) C=number of classes, M = embedding size
        weight_orilogit = weight[predictions, :]
        # weight_orilogit.shape = (B, M)
        weight_delta = weight_orilogit[:, None, :] - weight[None, :]
        # weight_delta.shape = (B, C, M)
        # (B, 1, M) - (1, C, M)
        bias_delta = bias[predictions, None] - bias[None, :]
        # bias_delta.shape = (B, C)
        # (B, 1) - (1, C)
        lam_numerator = 2 * ((embedding[:, None, :] * weight_delta).sum(dim=2) + bias_delta)
        # (B, 1, M) * (B, C, M)
        # lam_numerator.shape = (B, C)
        lam_denominator = (weight_delta ** 2).sum(dim=2)
        # lam_denominator.shape = (B, C)
        lam = lam_numerator / lam_denominator
        epsilon = -weight_delta * lam[:, :, None] / 2
        # epsilon.shape = (B, C, M)
        radius = torch.linalg.norm(epsilon, dim=2)
        radius = torch.where(torch.isnan(radius), torch.tensor(float('inf')).cuda(), radius)
        # radius.shape = (B, C)
        margins, min_margins_idx = radius.min(dim=1)

        return margins, radius


class PassiveMLPTrainer:
    def __init__(self, batch_size=250):
        self.batch_size = batch_size

    def train(self, imgs, labels, *args, **kwargs):
        self.model = train_mlp(imgs, labels, batch_size=self.batch_size)

    def pred(self, imgs):
        preds = []
        for i in range(int(np.ceil(imgs.size(0) / 250.))):
            img_batch = imgs[i * 250: min((i + 1) * 250, imgs.size(0))]
            preds.append(self.model(img_batch.cuda()))
        preds = torch.cat(preds, dim=0).squeeze(-1)
        return preds if imgs.is_cuda else preds.cpu()


class OnlineTrainer:
    def __init__(self, model, optimizer, loss_fn, accuracy_fn, batch_size=None, alpha=1, max_iter=100):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.accuracy_fn = accuracy_fn
        self.alpha = alpha
        self.max_iter = max_iter
        self.batch_size = batch_size

    def train(self, imgs, labels, new_idxs):
        new_imgs = imgs[new_idxs].cuda()
        new_labels = labels[new_idxs].cuda()
        wandb.log({"Num Queries": labels.size(0),
                   "Percent Positives": torch.sum(labels).cpu().numpy() / float(labels.size(0))})
        for i in range(2 * self.max_iter):
            if i % 2 == 0:
                train_imgs, train_labels = new_imgs, new_labels
                if self.batch_size is not None and self.batch_size > train_labels.size(0):
                    extra_batch = self.batch_size - train_labels.size(0)
                    idxs = torch.randperm(labels.size(0))[:extra_batch].cuda()
                    train_imgs, train_labels = torch.cat([train_imgs, imgs[idxs]], dim=0), torch.cat(
                        [train_labels, labels[idxs]], dim=0)
            else:
                idxs = torch.randperm(labels.size(0))[
                       :(new_idxs.size(0) if self.batch_size is None else self.batch_size)].cuda()
                train_imgs, train_labels = imgs[idxs].cuda(), labels[idxs].cuda()
            self.optimizer.zero_grad()
            pred = self.model(train_imgs).squeeze(-1)
            loss = self.loss_fn(pred, train_labels)
            if i % 2 == 0 and self.accuracy_fn(pred.data, train_labels) >= self.alpha:
                continue
            loss.backward()
            self.optimizer.step()

    def pred(self, imgs):
        preds = []
        for i in range(int(np.ceil(imgs.size(0) / 250.))):
            img_batch = imgs[i * 250: min((i + 1) * 250, imgs.size(0))]
            preds.append(self.model(img_batch.cuda()))
        preds = torch.cat(preds, dim=0).squeeze(-1)
        return preds if imgs.is_cuda else preds.cpu()
