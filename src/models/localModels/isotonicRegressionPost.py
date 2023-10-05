import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
# from sklearn.isotonic import isotonic_regression
# from quadprog import solve_qp
from qpsolvers import solve_qp
from scipy import sparse

import networkx as nx
from models.wrapper import WrapperLocal
from sklearn.base import clone


class IsotonicRegressionPost:
    def __init__(
            self,
            encoder=None,
            base_learner=LogisticRegression(max_iter=1000, class_weight='balanced'),
            # base_learner=SVC(max_iter=10000,  C=0.02, class_weight='balanced', probability=True, gamma='auto'),
        ):
        self.encoder = encoder
        self.trained_classifiers = None
        self.base_learner = base_learner
        self.predict_path = False
    
    def set_predictPath(self, val):
        self.predict_path = val
        
    def fit(self, X, y):
        self._fit_base_learner(X, y)
        return self
    
    def _fit_base_learner(self, X, y):
        encoded_y = self.encoder.transform(y)
        node_indicators = encoded_y.T
        self.trained_classifiers = np.zeros(encoded_y.shape[1], dtype=object)
        for idx, node_y in enumerate(node_indicators):
            unique_node_y = np.unique(node_y)
            if len(unique_node_y) == 1:
                cls = int(unique_node_y[0])
            else:
                cls = clone(self.base_learner)
                cls = cls.fit(X, node_y)

            self.trained_classifiers[idx] = cls

        # print(self.trained_classifiers)
        

    def run_IR(self, probas):
        """ ref to https://qpsolvers.github.io/qpsolvers/quadratic-programming.html"""
        nodes = self.encoder.G_idx.nodes()
        num_nodes = len(nodes)
        P = np.zeros((num_nodes, num_nodes))
        np.fill_diagonal(P, 1)

        C = self._get_C(nodes)
        G = sparse.csc_matrix(-1 * C)
        P = sparse.csc_matrix(P)

        h = np.zeros(C.shape[0])
        lb = np.zeros(C.shape[1])
        ub = np.ones(C.shape[1])
        probas_post = []
        for row in probas:
            q = -1 * row.T
            x = solve_qp(0.5 * P, q, G=G, h=h,lb=lb, ub=ub, solver="osqp")
            probas_post.append(x)
        # print(x - row)
        return np.array(probas_post)


    def _get_C(self, nodes):
        """
        Constraint matrix for quadratic prog, ensure that, pi < pj, if j is a parent of i.
        """
        num_nodes = len(nodes)
        C = []
        for i in range(num_nodes):
            successors = list(self.encoder.G_idx.successors(i))
            for child in successors:
                row = np.zeros(num_nodes)
                row[i] = 1.0
                row[child] = -1.0
                C.append(row)
        # print(np.array(C).shape, num_nodes)
        return np.array(C)
    
    def set_encoder(self, encoder):
        self.encoder = encoder

    def predict_proba(self, X, raw=False):
        probas = []
        for cls in self.trained_classifiers:
            if isinstance(cls, int):
                col = np.repeat([cls], X.shape[0])
            else:
                col = cls.predict_proba(X)[:, 1] # proba for pos class
            probas.append(col)
        if raw:
            return np.array(probas).T

        probas =  self.run_IR(np.array(probas).T)
        return probas


    def predict(self, X, threshold=0.5):
        probas = self.predict_proba(X)
        if self.predict_path:
            # preds = []
            # for cls in self.trained_classifiers:
            #     if isinstance(cls, int):
            #         col = np.repeat([cls], X.shape[0])
            #     else:
            #         col = cls.predict(X) # proba for pos class
            #     preds.append(col)
            # return np.array(preds).T
            return probas > threshold
        return self._inference_1(probas)

    #TODO: refactor to a func
    def _inference_1(self, probas, threshold=0.5):
        predicted = probas > threshold
        y_pred = np.zeros(predicted.shape[0])
        for row_idx, row in enumerate(predicted):
            counts = row[self.encoder.label_idx].sum()
            if counts < 2:
                idx = np.argmax(probas[row_idx, self.encoder.label_idx])
                y_pred[row_idx] = self.encoder.label_idx[idx]
            else:
                labels = self.encoder.label_idx[row[self.encoder.label_idx].tolist()]
                # if counts == 0:
                #     labels = self.encoder.label_idx
                mask = np.argsort(probas[row_idx, labels])
                # print(mask)
                labels_sorted = [labels[i] for i in mask]
                preds = []
                while len(labels_sorted) != 0:
                    ancestors = nx.ancestors(self.encoder.G_idx, labels_sorted[0])
                    path = [labels_sorted[0]]
                    preds.append(labels_sorted[0])
                    if ancestors is not None:
                        path += list(ancestors)
                    labels_sorted = [i for i in labels_sorted if i not in path]
                idx = np.argmax(probas[row_idx, preds])
                y_pred[row_idx] = preds[idx]
        return y_pred.astype(int)
    
    #TODO refactor to a module
    def _inference_2(self, probas, threshold=0.5):
        """Select index of the last 1 on the path as the preds"""
        predicted = probas > threshold
        y_pred = np.zeros(predicted.shape[0])

        for row_idx, row in enumerate(predicted):
            for idx in range(len(row) - 1 , -1, -1):
                if row[idx] and idx in self.encoder.label_idx:
                    y_pred[row_idx] = idx
                    break
        return y_pred


params = dict(
        name='IsotonicRegressionPost',
        model=IsotonicRegressionPost(),
)

# Please don't change this line
wrapper = WrapperLocal(**params)


if __name__ == "__main__":
    pass