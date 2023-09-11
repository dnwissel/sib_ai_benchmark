import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
# from sklearn.isotonic import isotonic_regression
from quadprog import solve_qp
import networkx as nx
from models.wrapper import WrapperLocal


class CascadedLRPost:
    def __init__(
            self,
            encoder=None,
            base_learner=LogisticRegression(max_iter=1000),
        ):
        self.encoder = encoder
        self.trained_classifiers = None
        self.base_learner = base_learner
        self.node_indicators = None
       
    def fit(self, X, y):
        self._fit_base_learner(X, y)
        return self
    
    def _fit_base_learner(self, X, y):
        encoded_y = self.encoder.transform(y)
        self.node_indicators = encoded_y.T
        self.trained_classifiers = np.zeros(encoded_y.shape[1], dtype=object)
        for idx, node_y in enumerate(self.node_indicators):
            unique_node_y = np.unique(node_y)
            if len(unique_node_y) == 1:
                cls = int(unique_node_y[0])
            else:
                cls = self.base_learner.fit(X, node_y)
            self.trained_classifiers[idx] = cls

    # def fit_LR(self, probas):
    #     R = self.encoder.get_R()
    #     self.LR_cls = [] # train for each node
    #     n_sample = probas.shape(0)
    #     n_node = probas.shape(1)

    #     LR = LogisticRegression(max_iter=1000)
    #     for node in range(n_node):
    #         node_y = self.node_indicators[:, node]
    #         unique_node_y = np.unique(node_y)
    #         if node in self.encoder.root_idx:
    #             cls = 1
    #         elif len(unique_node_y) == 1:
    #             cls = int(unique_node_y[0])
    #         else:
    #             mask = R[node, :]
    #             anc_probas = probas[:, mask]
    #             cls = LR.fit(anc_probas, node_y)
    #         self.LR_cls[node] = cls


    # def predict_LR(self):
    #     pass
    

    def _coumpute_marginals(self, anc_matrix, log_marginal_probas, log_proba, idx):
        log_mp = log_marginal_probas[idx] 
        if log_mp is not None:
            return log_mp
        
        anc_mask = anc_matrix[idx, :]
        idx = np.arange(len(log_marginal_probas))
        anc_idxs = idx[anc_mask]
        sum = 0
        for idx in anc_idxs:
            anc_proba = log_marginal_probas[idx]
            if anc_proba is not None:
                sum += anc_proba
            else:
                sum += self._coumpute_marginals(anc_matrix, log_marginal_probas, log_proba, idx)

        return sum


    def get_marginal_proba(self, log_probas_full):
        """Algo in CELLO paper, actually AND logic"""
        # Compute the marginals for each label by multiplying all of the conditional
        # probabilities from that label up to the root of the DAG
        anc_matrix = self.encoder.get_R().T

        marginal_probas_full = []
        for log_probas in log_probas_full:
            log_marginal_probas = np.full(shape=len(log_probas), fill_value=None)
            roots_idx = self.encoder.roots_idx
            log_marginal_probas[roots_idx] = 0
            for idx, log_proba in enumerate(log_probas):
                self._coumpute_marginals(anc_matrix, log_marginal_probas, log_proba, idx)
            
            marginal_probas = np.exp(log_marginal_probas)
            marginal_probas_full.append(marginal_probas)


    def set_encoder(self, encoder):
        self.encoder = encoder


    def predict_log_proba(self, X):
        probas = []
        for cls in self.trained_classifiers:
            if isinstance(cls, int):
                col = np.repeat(np.log([cls]), X.shape[0]) # TODO log
            else:
                col = cls.predict_log_proba(X)[:, 1] # proba for pos class
            probas.append(col)
        probas =  np.array(probas).T
        return probas


    def predict(self, X):
        log_probas = self.predict_log_proba(X)
        marginal_probas_full = self.get_marginal_proba(log_probas)
        return self._inference(marginal_probas_full)

    #TODO: refactor to a func
    def _inference(self, probas, threshold=0.5):
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
        
params = dict(
        name='CascadedLRPost',
        model=CascadedLRPost(),
)

# Please don't change this line
wrapper = WrapperLocal(**params)


if __name__ == "__main__":
    pass