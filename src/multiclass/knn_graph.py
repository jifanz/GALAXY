from src.graph import *
import faiss
import faiss.contrib.torch_utils


def create_knn_graph(features, labels, name, n_order=1):
    """
    Construct k Nearest Neighbor graphs based on the features.
    :param labels: If K classes, each elements of labels takes 0, ..., K-1.
    :return: a MultiLabelGraph
    """
    num_classes = int(np.max(labels)) + 1
    res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatL2(features.size(1))
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    gpu_index_flat.add(features)
    _, nn_idxs = gpu_index_flat.search(features, n_order + 1)
    nodes = []
    for idx in range(features.size(0)):
        nodes.append(Node(idx, labels[idx], idx))
    for i, nn_idx in enumerate(nn_idxs):
        for j in nn_idx:
            if j != i:
                nodes[i].add_neighbors([nodes[j]])
                nodes[j].add_neighbors([nodes[i]])
    return MultiLabelGraph(nodes, labels, name, num_classes)
