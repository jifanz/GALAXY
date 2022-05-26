import numpy as np
import torch
import sklearn.metrics as metrics


def channel_f2l(img):
    if len(img.shape) == 4:
        return img.transpose(0, 2, 3, 1)
    elif len(img.shape) == 3:
        return img.transpose(1, 2, 0)


def channel_l2f(img):
    if len(img.shape) == 4:
        return img.transpose(0, 3, 1, 2)
    elif len(img.shape) == 3:
        return img.transpose(2, 0, 1)


def to_one_hot(label, num):
    vec = torch.zeros(label.size(0), num).float().cuda()
    vec[torch.arange(label.size(0)), label % num] = 1
    return vec


def calculate_auc(scores, labels):
    fpr, tpr, threshold = metrics.roc_curve(1 - labels.cpu().numpy(), 1 - torch.sigmoid(scores).cpu().numpy())
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc


def calculate_mAP(scores, labels, weighted=False):
    labels = labels.cpu().numpy()
    scores = torch.sigmoid(scores).cpu().numpy()
    min_label = np.min(labels)
    max_label = np.max(labels)
    ap_scores = [metrics.average_precision_score(labels, scores,
                                                 average="weighted" if weighted else "macro", pos_label=i) for i in
                 np.arange(min_label, max_label + 1)]
    return np.mean(np.asarray(ap_scores))
