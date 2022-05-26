import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
# from src.hyperparam import pretrain

pretrain = True

class Identity(nn.Module):
    def forward(self, x):
        return x


class Resnet18Feature(nn.Module):
    def __init__(self, resnet=None):
        super(Resnet18Feature, self).__init__()
        if resnet is None:
            self.resnet = models.resnet18(pretrained=pretrain)
        else:
            self.resnet = resnet
        self.resnet.avgpool = Identity()
        self.resnet.fc = Identity()
        # print(self.resnet)

    def forward(self, imgs):
        return self.resnet(imgs)


class Resnet18Classification(nn.Module):
    def __init__(self, num_output, resnet=None, ret_emb=False):
        super(Resnet18Classification, self).__init__()
        if resnet is None:
            self.resnet = models.resnet18(pretrained=pretrain)
        else:
            self.resnet = resnet
        self.resnet.fc = Identity()
        self.linear = nn.Linear(512, num_output)
        self.ret_emb = ret_emb
        self.num_output= num_output

    def forward(self, imgs, ret_features=False, last=False, freeze=False):
        if freeze:
            with torch.no_grad():
                features = self.resnet(imgs)
        else:
            features = self.resnet(imgs)
        if ret_features:
            return self.linear(features), features.data
        elif self.ret_emb or last:
            return self.linear(features), features.view(features.size(0), -1)
        else:
            return self.linear(features)

    def get_embedding_dim(self):
        return 512


class MLPClassification(nn.Module):
    def __init__(self, num_output, num_input=3 * 32 * 32):
        super(MLPClassification, self).__init__()
        self.linear1 = nn.Linear(num_input, 128)
        self.linear2 = nn.Linear(128, num_output)

    def forward(self, imgs):
        imgs = imgs.view(imgs.size(0), -1)
        return self.linear2(F.relu(self.linear1(imgs)))


class LogisticRegression(nn.Module):
    def __init__(self, in_features, mean=0, std=1):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(in_features, 1)
        self.mean = mean
        self.std = std

    def forward(self, features):
        return self.linear((features - self.mean) / self.std)


class ShallowNet(nn.Module):
    def __init__(self, in_features, mean=0, std=1):
        super(ShallowNet, self).__init__()
        self.linear1 = nn.Linear(in_features, 512)
        self.linear2 = nn.Linear(512, 1)
        self.mean = mean
        self.std = std

    def forward(self, features):
        x = self.linear1((features - self.mean) / self.std)
        return self.linear2(F.relu(x))


def train_logistic(train_features, train_labels, half=False, plot=False, num_epoch=300):
    model = LogisticRegression(train_features.size(-1), mean=torch.mean(train_features, dim=0),
                               std=torch.std(train_features, dim=0)).cuda()
    if half:
        model.half()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    accuracy = []
    for _ in range(num_epoch):
        pred = model(train_features).squeeze(-1)
        accuracy.append(torch.mean(((pred > 0).half() == train_labels.half()).half()).data.cpu().numpy())
        loss = F.binary_cross_entropy_with_logits(pred, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if plot:
        plt.plot(accuracy)
        plt.show()
    return model


def train_shallow(train_features, train_labels, half=False, plot=False, num_epoch=300):
    model = ShallowNet(train_features.size(-1), mean=torch.mean(train_features, dim=0),
                       std=torch.std(train_features, dim=0)).cuda()
    if half:
        model.half()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    accuracy = []
    for _ in range(num_epoch):
        pred = model(train_features).squeeze(-1)
        accuracy.append(torch.mean(((pred > 0).half() == train_labels.half()).half()).data.cpu().numpy())
        loss = F.binary_cross_entropy_with_logits(pred, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if plot:
        plt.plot(accuracy)
        plt.show()
    return model


def train_mlp(train_features, train_labels, plot=False, num_epoch=500, batch_size=250):
    n_class = torch.max(train_labels).cpu().numpy() + 1
    model = MLPClassification(1 if n_class == 2 else n_class, num_input=train_features[0].view(-1).size(0)).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-5)
    accuracy = []
    num_batch = train_labels.size(0) // batch_size
    for _ in range(num_epoch):
        perm = torch.randperm(train_labels.size(0)).cuda()
        for i in range(num_batch):
            idxs = perm[i * batch_size: min((i + 1) * batch_size, train_labels.size(0))]
            pred = model(train_features[idxs].cuda()).squeeze(-1)
            if n_class == 2:
                loss = F.binary_cross_entropy_with_logits(pred, train_labels[idxs].cuda())
                accuracy.append(
                    torch.mean(((pred > 0).float() == train_labels[idxs].cuda().float()).float()).data.cpu().numpy())
            else:
                loss = F.cross_entropy(pred, train_labels[idxs].cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    if plot:
        plt.plot(accuracy)
        plt.show()
    return model


def train_resnet18(train_features, train_labels, plot=False, num_epoch=500, batch_size=250, n_class=None):
    if n_class is None:
        n_class = torch.max(train_labels).cpu().numpy() + 1
    # model = Resnet18Classification(1 if n_class == 2 else n_class).cuda()
    model = Resnet18Classification(n_class).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-5)
    accuracy = []
    num_batch = train_labels.size(0) // batch_size
    num_preds = torch.sum((train_labels.unsqueeze(-1) == torch.arange(model.num_output)).float(), dim=0)
    for _ in range(num_epoch):
        perm = torch.randperm(train_labels.size(0)).cuda()
        for i in range(num_batch):
            idxs = perm[i * batch_size: min((i + 1) * batch_size, train_labels.size(0))]
            pred = model(train_features[idxs].cuda()).squeeze(-1)
            # if n_class == 2:
            #     loss = F.binary_cross_entropy_with_logits(pred, train_labels[idxs].cuda())
            #     accuracy.append(
            #         torch.mean(((pred > 0).float() == train_labels[idxs].cuda().float()).float()).data.cpu().numpy())
            # else:
            loss = F.cross_entropy(pred, train_labels[idxs].cuda(), weight=1. / torch.clip(num_preds.cuda(), min=1e-6))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    if plot:
        plt.plot(accuracy)
        plt.show()
    return model
