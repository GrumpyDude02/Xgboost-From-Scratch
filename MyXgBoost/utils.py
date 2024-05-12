import numpy as np


def mse_score(y, lhs, rhs):
    lhs_std = y[lhs].std()
    rhs_std = y[rhs].std()
    return len(lhs) * lhs_std + len(rhs) * rhs_std


def gini_index(y):
    labels = np.unique(y)
    gini = 0
    for c in labels:
        p_cls = len(y[y == c]) / len(y)
        gini += p_cls**2
    return 1 - gini


def gini_score(y, lhs_idxs, rhs_idxs):
    weight_l = len(y[lhs_idxs]) / len(y)
    weight_r = len(y[rhs_idxs]) / len(y)
    return weight_l * gini_index(y[lhs_idxs]) + weight_r * gini_index(y[rhs_idxs]) - gini_index(y)


def entropy_index(y):
    labels = np.unique(y)
    entropy = 0
    for c in labels:
        p_cls = len(y[y == c]) / len(y)
        entropy -= p_cls * np.log2(p_cls)
    return entropy


def entropy_score(y, lhs_idxs, rhs_idxs):
    weight_l = len(lhs_idxs) / len(y)
    weight_r = len(rhs_idxs) / len(y)
    return weight_l * entropy_index(y[lhs_idxs]) + weight_r * entropy_index(y[rhs_idxs]) - entropy_index(y)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
