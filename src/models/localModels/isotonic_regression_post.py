import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
# from sklearn.isotonic import isotonic_regression
from quadprog import solve_qp
import networkx as nx
from models.wrapper import Wrapper


class IsotonicRegressionPost:
    def __init__(
            self,
            encoder=None,
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
        C = self._get_C(nodes)
        probas_post = []
        for row in probas:
            sol = solve_qp(G, row * 2, C.T, b)
            probas_post.append(sol[0])
        return np.array(probas_post)


    def _get_C(self, nodes):
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
    
    def set_encoder(self, encoder):
        self.encoder = encoder

    def predict_proba(self, X):
        probas = []
        for cls in self.trained_classifiers:
            if isinstance(cls, int):
                col = np.repeat([cls], X.shape[0])
            else:
                col = cls.predict_proba(X)
            probas.append(col)
        return np.array(probas).T


    def predict(self, X):
        probas = self.predict_proba(X)
        return self._inference(probas)


    def _inference(self, probas):
        predicted = probas > 0.5
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
        return y_pred
        
params = dict(
        name='IR',
        model=IsotonicRegressionPost(),
)

# Please don't change this line
wrapper = Wrapper(**params)


if __name__ == "__main__":
    pass