from hiclass.metrics import f1, recall, precision
import numpy as np
import torch


def f1_hier(y_true, y_pred, en):
    y_true_encoded = en.transform(y_true)

    y_true_labels = to_labels(y_true_encoded[:, en.idx_to_eval])
    y_pred_labels = to_labels(y_pred[:, en.idx_to_eval])

    # print(y_true_labels, y_pred_labels)
    score = f1(y_true_labels, y_pred_labels)
    return score


def recall_hier(y_true, y_pred, en):
    y_true_encoded = en.transform(y_true)
    # print(y_true_encoded, y_pred)

    y_true_labels = to_labels(y_true_encoded[:, en.idx_to_eval])
    y_pred_labels = to_labels(y_pred[:, en.idx_to_eval])
    score = recall(y_true_labels, y_pred_labels)
    return score


def precision_hier(y_true, y_pred, en):
    y_true_encoded = en.transform(y_true)
    # print(y_true_encoded, y_pred)

    y_true_labels = to_labels(y_true_encoded[:, en.idx_to_eval])
    y_pred_labels = to_labels(y_pred[:, en.idx_to_eval])
    score = precision(y_true_labels, y_pred_labels)
    return score


def f1_hier_report(y_true, y_pred, idx_to_eval):
    # y_true_encoded = en.transform(y_true)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # print(y_true.shape)
    # print(y_pred.shape)
    y_true_labels = to_labels(y_true[:, idx_to_eval])
    y_pred_labels = to_labels(y_pred[:, idx_to_eval])

    # print(y_true_labels, y_pred_labels)
    score = f1(y_true_labels, y_pred_labels)
    return score


def to_labels(y):
    labels = []
    idxs = np.arange(y.shape[1])
    for row in y:
        if torch.is_tensor(row):
            row = row.numpy()

        row = row.astype('bool')
        labels.append(idxs[row])
    return labels
