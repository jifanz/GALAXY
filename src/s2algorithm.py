from src.graph import *
import numpy as np


def s2query(graph):
    dist, path = graph.shortest_shortest_path()
    if path is None:
        query = graph.not_queried[np.random.randint(len(graph.not_queried))]
        # print("Random Sample")
    else:
        assert len(path) > 2, "Found path connecting oppositely label nodes."
        query = path[len(path) // 2]
        # print([n.idx for n in path])
    return query.idx


def bisection_query(graph):
    dist, path = graph.shortest_shortest_path()
    if path is None:
        query = graph.not_queried[np.random.randint(len(graph.not_queried))]
        return None, query.idx
    else:
        assert len(path) > 2, "Found path connecting oppositely label nodes."
        query = path[len(path) // 2]
        return query.idx, query.idx


def run_s2(graph, init_batch_size, num_iter):
    gt_errors = []
    init_batch = np.random.choice(graph.num_nodes, size=init_batch_size, replace=False)
    queries = list(init_batch)
    for idx in init_batch:
        print("queried", idx)
        graph.label(idx)
        graph.nn_pred()
        gt_errors.append(graph.gt_error())
