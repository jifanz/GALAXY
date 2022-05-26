from src.graph import *


def create_linear_graphs(scores, labels, name, n_order=1):
    """
    Construct linear graphs based on the sorted scores along each class.
    :param labels: If K classes, each elements of labels takes 0, ..., K-1.
    :return: a MultiLinearGraph
    """
    most_confident = np.max(scores, axis=1).reshape((-1, 1))
    scores = scores - most_confident + 1e-8 * most_confident
    num_classes = int(np.max(labels)) + 1
    graphs = []
    for c in range(num_classes):
        sorted_idx = np.argsort(scores[:, c])
        nodes = []
        label_class = (labels == c).astype(float) * 2 - 1
        for idx in sorted_idx:
            nodes.append(Node(idx, label_class[idx], idx))
        for order in range(1, n_order + 1):
            for i in range(order):
                nodes[i].add_neighbors([nodes[i + order]])
                nodes[-i - 1].add_neighbors([nodes[-i - order - 1]])
            for i in range(order, len(nodes) - order):
                nodes[i].add_neighbors([nodes[i - order], nodes[i + order]])
        graphs.append(Graph(nodes, name + "class_%d" % c))
    return MultiLinearGraph(graphs, n_order)
