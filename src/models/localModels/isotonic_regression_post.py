#################################################################
#   Hierarchical classification via isotonic regression
#################################################################
import sys
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
# from sklearn.isotonic import isotonic_regression
from quadprog import solve_qp
import networkx as nx

from utilities.hier import get_R

class IsotonicRegressionPost():
    def __init__(
            self,
            encoder,
            base_learner=LogisticRegression(),
        ):
        self.encoder = encoder
        self.trained_classifiers = None
        self.base_learner = base_learner
       
    def fit(self, X, y):
        self._fit_base_learner(X, y)
        return self
    
    def _fit_base_learner(self, X, y):
        encoded_y = self.encoder.transform(y)
        node_indicators = encoded_y.T
        self.trained_classifiers = np.zeros(encoded_y.shape[1])
        for idx, node_y in enumerate(node_indicators):
            unique_node_y = np.unique(node_y)
            if len(unique_node_y) == 1:
                self.trained_classifiers[idx] = unique_node_y[0]
            else:
                self.trained_classifiers[idx] = self.base_learner.fit(X, node_y)

    def run_IR(self, probas):
        nodes = self.encoder.G_idx.nodes()
        num_nodes = len(nodes)
        G = np.zeros((num_nodes, num_nodes))
        np.fill_diagonal(G, 2)

        b = 0
        C = self._get_C()
        probas_post = []
        for row in probas:
            sol = solve_qp(G, row * 2, C.T, b)
            probas_post.append(sol[0])

    def _get_C(self):
        nodes = self.encoder.G_idx.nodes()
        num_nodes = len(nodes)
        C = []
        for i in range(num_nodes):
            successors = list(self.encoder.G_idx.successors(i))
            for child in successors:
                row = np.zeros(len(num_nodes))
                row[i] = 1
                row[child] = -1
                C.append(row)
        return C
    
    def predict_proba(self):
        pass

    def predict(self, X):
        pass
        