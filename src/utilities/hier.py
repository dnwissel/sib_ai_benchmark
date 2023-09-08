import itertools
import numpy as np
import pandas as pd
import torch
import networkx as nx


class Encoder:
    def __init__(self, G_full, roots_label):
        self.G_idx = None
        self.G_label = None
        self.G_full = G_full

        self.roots_idx = None
        self.idx_label = None
        self.node_map = None
        self.roots_label = roots_label
        self.predecessor_dict = {}
        self.successor_dict = {}

    def fit(self, y):
        # print(y.unique())`
        ancestors = [nx.ancestors(self.G_full,n) for n in y.unique()]
        ancestors = set(itertools.chain(*ancestors))
        self.G_label = self.G_full.subgraph(ancestors | set(y.unique()))
        # self.G_full.remove_nodes_from([n for n in self.G_full if n not in (ancestors | set(y.unique()))])
        # self.G_label = self.G_full
        # Check root node
        root_list = []
        for node in self.G_label.nodes():
            if self.G_label.out_degree(node) > 0 and self.G_label.in_degree(node) == 0:
                root_list.append(node)
        if len(root_list) == 1:
            root = root_list[0]
            nodes = sorted(self.G_label.nodes(), key=lambda x: (nx.shortest_path_length(self.G_label, root, x), x)) # sort with x if same shorted path len
            self.node_map = dict(zip(nodes, range(len(nodes))))
            # print( self.node_map)
        else:
            raise ValueError('More than one root found')

        adjacency_matrix = np.array(nx.adjacency_matrix(self.G_label, nodelist=nodes).todense())
        self.G_idx = nx.DiGraph(adjacency_matrix)
        self.roots_idx = [v for k, v in  self.node_map.items() if k in self.roots_label]

        self.label_idx = np.array(list(map( self.node_map.get, y.unique())))

        for n in self.G_idx.nodes:
            self.predecessor_dict[n] = list(self.G_idx.predecessors(n))
            self.successor_dict[n] = list(self.G_idx.successors(n))

        return self

    def transform(self, y, is_idx=True):
        # print(y.unique())
        return self._encode_y(y, is_idx)


    #TODO: novel label in test
    def _encode_y(self, nodes, is_idx):
        num_class = len(self.G_idx.nodes())
        Y = []
        if is_idx:
            for node in nodes:
                # print(node, self.G_idx.nodes())
                y_ = np.zeros(num_class)
                if  node in self.G_idx.nodes():
                    y_[[ a for a in nx.ancestors(self.G_idx, node)]] = 1
                    y_[node] = 1
                Y.append(y_)
        else:
            labels = nodes
            for label in labels:
                y_ = np.zeros(num_class)
                if  self.node_map.get(label) is not None:
                    y_[[ self.node_map.get(a) for a in nx.ancestors(self.G_label, label)]] = 1
                    y_[ self.node_map[label]] = 1
                    Y.append(y_)
        Y = np.stack(Y)
        return Y
    
    
    def get_R(self, tensor=True):
        # Compute matrix of ancestors R
        # Given n classes, R is an (n x n) matrix where R_ij = 1 if class i is ancestor of class j
        nodes = self.G_idx.nodes()
        num_nodes = len(nodes)
        R = np.zeros((num_nodes, num_nodes))
        np.fill_diagonal(R, 1)

        for i in range(num_nodes):
            ancestors = list(nx.ancestors(self.G_idx, i))
            if ancestors:
                R[i, ancestors] = 1

        if tensor:
            R = torch.tensor(R)
            #Transpose to get the descendants for each node
            R = R.transpose(1, 0)
            R = R.unsqueeze(0)
            return R
        return R


    def get_lossMask(self):
        # Compute loss mask for Cont. sigmoid
        nodes = self.G_idx.nodes()
        num_nodes = len(nodes)
        loss_mask = np.zeros((num_nodes, num_nodes))
        np.fill_diagonal(loss_mask, 1)

        for i in range(num_nodes):
            ancestors = list(nx.ancestors(self.G_idx, i))
            children = list(self.G_idx.successors(i))
            parents = list(self.G_idx.predecessors(i))
            for a in ancestors:
                children = list(self.G_idx.successors(a))
                if children:
                    loss_mask[i, children] = 1

            if children:
                loss_mask[i, children] = 1
            if parents:
                loss_mask[i, parents] = 1

        loss_mask = torch.tensor(loss_mask)
        # loss_mask = loss_mask.unsqueeze(0).to(device)
        # loss_mask = loss_mask.to(device)
        return loss_mask
