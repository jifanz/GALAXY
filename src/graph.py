import numpy as np
from queue import PriorityQueue, Queue


class Node:
    def __init__(self, idx, label, loc):
        self.idx = idx
        self.label = label
        self.queried = False
        self.loc = loc
        self.neighbors = set()

    def add_neighbors(self, neighbors):
        for n in neighbors:
            self.neighbors.add(n)

    def set_neighbors(self, neighbors):
        self.neighbors = set(neighbors)


class Graph:
    def __init__(self, nodes, name):
        self.name = name
        self.nodes = set(nodes)
        self.node_list = list(nodes)
        self.num_nodes = len(nodes)
        self.node_dict = {node.idx: node for node in self.nodes}
        self.edges = []
        for node in nodes:
            for neighbor in node.neighbors:
                if node.idx < neighbor.idx:
                    self.edges.append((node, neighbor))
        self.edges = set(self.edges)
        self.queried = []
        self.not_queried = list(nodes)
        self.hamming_error = 0
        self.cut_error = 0
        self.preds = None
        self.labels = np.array([0 for _ in nodes])
        for node in nodes:
            self.labels[node.idx] = node.label

    def label(self, idx):
        node = self.node_dict[idx]
        node.queried = True
        self.queried.append(node)
        self.not_queried.remove(node)

        new_neighbors = []
        for neighbor in node.neighbors:
            if neighbor.queried and neighbor.label != node.label:
                self.cut_error += 1
                if node.idx < neighbor.idx:
                    self.edges.remove((node, neighbor))
                else:
                    self.edges.remove((neighbor, node))
                neighbor.neighbors.remove(node)
            else:
                new_neighbors.append(neighbor)
        node.set_neighbors(new_neighbors)

        if self.preds is not None and self.preds[idx] != node.label:
            self.hamming_error += 1

    def nn_pred(self):
        self.preds = np.random.choice([-1, 1], len(self.nodes))
        queue = Queue()
        predicted = set(self.queried)
        for n in self.queried:
            self.preds[n.idx] = n.label
            queue.put(n)
        while not queue.empty():
            n = queue.get()
            for neighbor in n.neighbors:
                if neighbor not in predicted:
                    queue.put(neighbor)
                    self.preds[neighbor.idx] = self.preds[n.idx]
                    predicted.add(neighbor)
        return self.preds

    def gt_error(self):
        return np.sum((np.array(self.preds) != self.labels).astype(int))

    def pred_cut(self):
        self.pred_cut_error = self.cut_error
        for (n1, n2) in self.edges:
            if self.preds[n1.idx] != self.preds[n2.idx]:
                self.pred_cut_error += 1
        self.pred_cut_error += (self.pred_cut_error - self.cut_error) / float(len(self.nodes))

    def shortest_shortest_path(self):
        queue = PriorityQueue()
        count = 0
        dist = {}
        path_prev = {}
        positive_queried = set()
        negative_queried = set()
        for node in self.queried:
            if node.label == 1:
                positive_queried.add(node)
            else:
                negative_queried.add(node)

        for node in self.nodes:
            if node in positive_queried:
                dist[node] = 0
                queue.put((0, count, node))
                count += 1
            else:
                dist[node] = 2 * len(self.nodes)
            path_prev[node] = None

        while not queue.empty():
            _, _, node = queue.get()
            for neighbor in node.neighbors:
                new_dist = dist[node] + 1
                if new_dist < dist[neighbor]:
                    dist[neighbor] = new_dist
                    path_prev[neighbor] = node
                    queue.put((new_dist, count, neighbor))
                    count += 1
                if neighbor in negative_queried:
                    # We can do this because edge weights are all 1's
                    current = neighbor
                    path = [neighbor]
                    while path_prev[current] is not None:
                        current = path_prev[current]
                        path.append(current)
                    return new_dist, path
        return float('inf'), None


class MultiLinearGraph():
    def __init__(self, graphs, n_order):
        self.graphs = graphs
        self.n_order = n_order
        self.nodes = graphs[0].nodes
        self.node_list = graphs[0].node_list
        self.num_nodes = len(graphs[0].nodes)
        self.node_dict = graphs[0].node_dict
        self.not_queried = graphs[0].not_queried
        self.s2_iteration = 0

    def label(self, idx):
        for graph in self.graphs:
            graph.label(idx)

    def shortest_shortest_path(self, increment=True):
        min_dist = float('inf')
        min_path = None
        # for graph in self.graphs:
        #     dist, path = graph.shortest_shortest_path()
        #     if dist < min_dist:
        #         min_dist = dist
        #         min_path = path
        graph = self.graphs[self.s2_iteration % len(self.graphs)]
        dist, path = graph.shortest_shortest_path()
        if dist < min_dist:
            min_dist = dist
            min_path = path
        if min_path is None and increment and self.n_order < len(self.node_list):
            self.n_order += 1
            order = self.n_order
            for graph in self.graphs:
                for i in range(order):
                    self.check_add_neighbor(graph.node_list[i], graph.node_list[i + order], graph)
                    self.check_add_neighbor(graph.node_list[-i - 1], graph.node_list[-i - order - 1], graph)
                for i in range(order, len(graph.node_list) - order):
                    self.check_add_neighbor(graph.node_list[i], graph.node_list[i - order], graph)
                    self.check_add_neighbor(graph.node_list[i], graph.node_list[i + order], graph)
            return self.shortest_shortest_path(increment=False)
        self.s2_iteration += 1
        return min_dist, min_path

    @staticmethod
    def check_add_neighbor(n1: Node, n2: Node, graph: Graph):
        if (n1.label == n2.label) or (not n1.queried) or (not n2.queried):
            n1.add_neighbors([n2])
            if n1.idx < n2.idx:
                graph.edges.add((n1, n2))


class MultiLabelGraph():
    def __init__(self, nodes, labels, name, num_classes=10):
        self.graph = Graph(nodes, name)
        self.nodes = self.graph.nodes
        self.node_list = nodes
        self.num_nodes = self.graph.num_nodes
        self.node_dict = self.graph.node_dict
        self.labels = labels
        self.num_classes = 10
        self.not_queried = self.graph.not_queried

    def label(self, idx):
        self.graph.label(idx)

    def nn_pred(self):
        self.graph.nn_pred()

    def gt_error(self):
        return self.graph.gt_error()

    def shortest_shortest_path(self):
        if self.num_classes == 10:
            splits = [([0, 1, 2, 3, 4], [5, 6, 7, 8, 9]),
                      ([0, 1, 7, 8, 9], [5, 6, 2, 3, 4]),
                      ([0, 6, 2, 3, 9], [5, 1, 7, 8, 4]),
                      ([0, 6, 2, 8, 9], [5, 1, 7, 3, 4])]
        else:
             raise ValueError("Unexpected number of classes.")
        min_dist = float('inf')
        min_path = None
        for (s1, s2) in splits:
            for node, label in zip(self.node_list, self.labels):
                if label in s1:
                    node.label = -1
                else:
                    node.label = 1
            dist, path = self.graph.shortest_shortest_path()
            if dist < min_dist:
                min_dist = dist
                min_path = path

        for node, label in zip(self.node_list, self.labels):
            node.label = label
        return min_dist, min_path
