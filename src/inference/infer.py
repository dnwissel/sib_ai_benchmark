from functools import partial

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from scipy import sparse

from qpsolvers import solve_qp


def run_IR(probas, encoder):
    """ 
    ref to https://qpsolvers.github.io/qpsolvers/quadratic-programming.html
    """

    nodes = encoder.G_idx.nodes()
    num_nodes = len(nodes)
    P = np.zeros((num_nodes, num_nodes))
    np.fill_diagonal(P, 1)

    C = _get_C(nodes, encoder)
    G = sparse.csc_matrix(-1 * C)
    P = sparse.csc_matrix(P)

    h = np.zeros(C.shape[0])
    lb = np.zeros(C.shape[1])
    ub = np.ones(C.shape[1])
    probas_post = []

    cnt = 0
    for row in probas:
        # print(row)
        q = -1 * row.T
        x = solve_qp(0.5 * P, q, G=G, h=h, lb=lb, ub=ub, solver="osqp")
        probas_post.append(x)

        # plot
        plot = False
        if plot and cnt % 1000 == 0:
            # print(x - row)
            optimal = x
            diff = x - row
            num_nodes = len(x)
            x = range(num_nodes)
            # Create a figure and a grid of subplots
            fig, axs = plt.subplots(3, 1)

            # Plot on each subplot
            axs[0].plot(x, row, marker='o', linestyle='-')
            axs[1].plot(x, optimal, marker='o', linestyle='-')
            axs[2].plot(x, diff, marker='o', linestyle='-')

            # Set y-label for each subplot
            axs[0].set_ylabel('raw')
            axs[1].set_ylabel('optimal')
            axs[2].set_ylabel('diff')

            plt.xlabel('Node_index')
            # plt.ylabel('value')
            # plt.title('Plot of a Sequence of Numbers')

            for ax in axs[:2]:
                ax.axhline(y=0.5, color='r', linestyle='dotted')
            axs[2].axhline(y=0, color='g', linestyle='dotted')

            plt.subplots_adjust(hspace=0.4)

            plt.savefig(f'figure_{cnt}')
            plt.close()
        cnt += 1

    return np.array(probas_post)


def _get_C(nodes, encoder):
    """
    Constraint matrix for quadratic prog, ensure that, pi < pj, if j is a parent of i.
    """
    num_nodes = len(nodes)
    C = []
    for i in range(num_nodes):
        successors = list(encoder.G_idx.successors(i))
        for child in successors:
            row = np.zeros(num_nodes)
            row[i] = 1.0
            row[child] = -1.0
            C.append(row)
    # print(np.array(C)[:5])
    return np.array(C)


def _lhs_dp(node, en, row, memo):
    value = memo[node]
    if value != -1:
        return value
    s_prime_pos = list(
        map(partial(_lhs_dp, en=en, row=row, memo=memo), en.predecessor_dict[node]))
    lh = row[node] * (1 - torch.prod(1 - torch.tensor(s_prime_pos)))
    memo[node] = lh
    return lh


def infer_cs(probas, encoder):
    y_pred = np.zeros(probas.shape[0])
    y_probas = np.zeros(probas.shape[0])
    num_nodes = probas.shape[1]

    for row_idx, row in enumerate(probas):
        memo = np.zeros(num_nodes) - 1

        lhs = np.zeros(len(encoder.label_idx))
        for root in encoder.roots_idx:
            memo[root] = 1.0

        for idx, label in enumerate(encoder.label_idx):
            lh_ = _lhs_dp(label, encoder, row, memo)
            lh_children = np.prod(1 - row[list(encoder.successor_dict[label])])
            lhs[idx] = lh_ * lh_children
        y_pred[row_idx] = encoder.label_idx[np.argmax(lhs)]

        lhs_norm = lhs / lhs.sum()
        y_probas[row_idx] = np.max(lhs_norm)
    return y_pred, y_probas


def infer_path_cs(probas, encoder):
    probas_margin = np.zeros(probas.shape)
    num_nodes = probas.shape[1]

    for row_idx, row in enumerate(probas):
        memo = np.zeros(num_nodes) - 1
        lhs = np.zeros(num_nodes)

        for root in encoder.roots_idx:
            # print(root)
            lhs[root] = 1.0
            memo[root] = 1.0

        for label in range(num_nodes):
            if label in encoder.roots_idx:
                continue

            lh_ = _lhs_dp(label, encoder, row, memo)
            lh_children = np.prod(1 - row[list(encoder.successor_dict[label])])
            # lhs[label] = lh_ * lh_children
            lhs[label] = lh_
        probas_margin[row_idx] = lhs
    return probas_margin


def infer_1(probas, encoder, threshold=0.5):
    """
    Identify multiple paths. Compare the probability of the last predicted node of those paths
    """
    predicted = probas > threshold
    y_pred = np.zeros(predicted.shape[0])
    for row_idx, row in enumerate(predicted):
        counts = row[encoder.label_idx].sum()
        if counts < 2:
            idx = np.argmax(probas[row_idx, encoder.label_idx])
            y_pred[row_idx] = encoder.label_idx[idx]
        else:
            labels = encoder.label_idx[row[encoder.label_idx].tolist()]
            # if counts == 0:
            #     labels = encoder.label_idx
            mask = np.argsort(probas[row_idx, labels])

            labels_sorted = [labels[i] for i in mask]
            preds = []
            while len(labels_sorted) != 0:
                ancestors = nx.ancestors(encoder.G_idx, labels_sorted[0])
                path = [labels_sorted[0]]
                preds.append(labels_sorted[0])
                if ancestors is not None:
                    path += list(ancestors)
                labels_sorted = [i for i in labels_sorted if i not in path]
            idx = np.argmax(probas[row_idx, preds])
            y_pred[row_idx] = preds[idx]
    return y_pred.astype(int)


def infer_2(probas, encoder, threshold=0.5):
    """
    Select index of the last 1 on the path as the preds
    """
    predicted = probas > threshold
    y_pred = np.zeros(predicted.shape[0])

    for row_idx, row in enumerate(predicted):
        for idx in range(len(row) - 1, -1, -1):
            # idx = len(row) - 1 - ridx
            if row[idx] and idx in encoder.label_idx:
                y_pred[row_idx] = idx
                break
    return y_pred
