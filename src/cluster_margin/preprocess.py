from src.model import *
import os
from scipy.spatial.distance import cdist
from medmnist import INFO, PathMNIST


def hac(clusters, c_dists):
    num_empty = 0
    while c_dists.shape[0] / float(len(clusters) - num_empty) < 2:
        num_empty += 1
        num_elem = np.array([float(len(c)) for c in clusters])
        i, j = np.unravel_index(np.argmin(c_dists), c_dists.shape)
        assert num_elem[i] != 0. and num_elem[j] != 0.
        c_dists[i] = (c_dists[i] * num_elem[i] + c_dists[j] * num_elem[j]) / (num_elem[i] + num_elem[j])
        c_dists[:, i] = (c_dists[:, i] * num_elem[i] + c_dists[:, j] * num_elem[j]) / (num_elem[i] + num_elem[j])
        c_dists[j] = float("inf")
        c_dists[:, j] = float("inf")
        clusters[i] = clusters[i] + clusters[j]
        clusters[j] = []
    new_clusters = []
    for c in clusters:
        if len(c) != 0:
            new_clusters.append(c)
    print(len(new_clusters))
    return new_clusters


if __name__ == "__main__":
    data_name = "medmnist"
    if data_name == "cifar":
        dataset_class = datasets.cifar.CIFAR10
    elif data_name == "cifar100":
        dataset_class = datasets.cifar.CIFAR100
    elif data_name == "svhn":
        dataset_class = lambda root, train, download, transform: datasets.svhn.SVHN(root,
                                                                                    split="train" if train else "test",
                                                                                    download=download,
                                                                                    transform=transform)
    elif data_name == "medmnist":
        T = transforms.Compose(
            [transforms.Resize((32, 32)),
             transforms.ToTensor(),
             # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        def dataset_class(root, train, download, transform):
            return PathMNIST("train" if train else "test", transform=T, download=download, root=root)

    if os.path.exists("./%s_resnet18_dists" % data_name):
        dist = np.load("./%s_resnet18_dists" % data_name)
        print("Loaded dist.")
        print(dist.shape)
        features = torch.load(open("./%s_resnet18_features" % data_name, "rb"))
        print("Loaded features.")
        print(features.shape)
    else:
        if os.path.exists("./%s_resnet18_features" % data_name):
            features = torch.load(open("./%s_resnet18_features" % data_name, "rb"))
            print("Loaded features.")
        else:
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            dataset = dataset_class("./data/", train=True, download=True, transform=transform)
            loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=40)
            imgs, labels = next(iter(loader))
            if imgs.size(0) <= 50000:
                imgs, labels = imgs.cuda(), labels.cuda()
                model = Resnet18Feature().cuda()
                features = model(imgs).data.squeeze()
            else:
                features = []
                model = Resnet18Feature().cuda()
                for i in range(5):
                    imgs = imgs[i * 30000:(i + 1) * 30000].cuda()
                    with torch.no_grad():
                        features.append(model(imgs).data.squeeze())
                features = torch.cat(features, dim=0)
                del model, imgs
                torch.cuda.empty_cache()
            print(features.size(), labels.size())
            torch.save(features, open("./%s_resnet18_features" % data_name, "wb"))

        dist = torch.cdist(features.cpu(), features.cpu()).numpy()
        print(dist.shape)
        np.save(open("./%s_resnet18_dists" % data_name, "wb"), dist)

    features = features.data.cpu().numpy()
    np.random.seed(12345)
    slice = np.random.permutation(dist.shape[0])[:1000]
    c_dists = np.array(dist)[slice, :][:, slice]
    print(c_dists.shape)
    for i in range(c_dists.shape[0]):
        c_dists[i, i] = float("inf")
    clusters = hac([[i] for i in range(c_dists.shape[0])], c_dists)
    centers = []
    for i, c in enumerate(clusters):
        center = np.mean(features[slice[c]], axis=0)
        centers.append(center)
    centers = np.array(centers)
    dist = cdist(features, centers)
    cluster_idxs = np.argmin(dist, axis=1)
    print(len(cluster_idxs), np.max(cluster_idxs))
    np.save(open("./%s_resnet18_clusters" % data_name, "wb"), cluster_idxs)
