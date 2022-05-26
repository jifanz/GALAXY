from src.graph import *


def create_linear_from_list(labels, name, n_order=1):
    """
    Construct a linear graph based on the sequential labels.
    :param labels: -1's and 1's
    :return: a linear graph
    """
    assert len(labels) >= 3, "Linear graph must have more than 3 nodes."
    nodes = []
    idxs = [0 for _ in labels]
    current_idx = 0
    for i, label in enumerate(labels):
        if label == -1:
            idxs[i] = current_idx
            current_idx += 1
    for i, label in enumerate(labels):
        if label == 1:
            idxs[i] = current_idx
            current_idx += 1
    for i, label in enumerate(labels):
        nodes.append(Node(idxs[i], label, i))
    for order in range(1, n_order + 1):
        for i in range(order):
            nodes[i].add_neighbors([nodes[i + order]])
            nodes[-i - 1].add_neighbors([nodes[-i - order - 1]])
        for i in range(order, len(nodes) - order):
            nodes[i].add_neighbors([nodes[i - order], nodes[i + order]])

    return Graph(nodes, name)


def create_rotating_linear_from_list(labels, name, num_rotation):
    """
    Construct a list of linear graph based on the sequential labels but permutated vertex labels.
    :param labels: -1's and 1's
    :return: a list of linear graphs
    """
    assert len(labels) >= 3, "Linear graph must have more than 3 nodes."
    graphs = []
    for graph_idx in range(num_rotation):
        nodes = []
        idxs = [0 for _ in labels]
        current_idx = (graph_idx * 27) % len(labels)
        for i, label in enumerate(labels):
            if label == -1:
                idxs[i] = current_idx
                current_idx = (current_idx + 1) % len(labels)
        for i, label in enumerate(labels):
            if label == 1:
                idxs[i] = current_idx
                current_idx = (current_idx + 1) % len(labels)
        assert current_idx == (graph_idx * 27) % len(labels)
        for i, label in enumerate(labels):
            nodes.append(Node(idxs[i], label, i))
        nodes[0].add_neighbors([nodes[1]])
        nodes[-1].add_neighbors([nodes[-2]])
        for i in range(1, len(nodes) - 1):
            nodes[i].add_neighbors([nodes[i - 1], nodes[i + 1]])
        graphs.append(Graph(nodes, name + ("_%d" % graph_idx)))
    return graphs
