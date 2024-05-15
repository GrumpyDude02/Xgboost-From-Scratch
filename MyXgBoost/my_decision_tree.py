import numpy as np, MyXgBoost.utils as utils, pandas as pd
from MyXgBoost._tree import _BaseTree


class MyDecisionTree(_BaseTree):
    """Criterion can be:
    "gini",
    "entropy",
    "mse" """

    def _classification_leaf_value(y):
        uniques, count = np.unique(y, return_counts=True)
        return uniques[np.argmax(count)]

    def _regression_leaf_value(y):
        return np.mean(y)

    Criterion = {
        "gini": (utils.gini_score, _classification_leaf_value),
        "entropy": (utils.entropy_score, _classification_leaf_value),
        "mse": (None, _regression_leaf_value),
    }

    def __init__(self, max_depth=5, min_sample_leaf=10, criterion="mse") -> None:
        super().__init__(max_depth, min_sample_leaf)
        self.loss_function, self._calculate_leaf_value = MyDecisionTree.Criterion.get(criterion, (None, None))
        if self._calculate_leaf_value is None:
            raise Exception("Udefined Criterion")
        elif self.loss_function is None:
            self._split_function = self._fast_find_split_regression
        else:
            self._split_function = self._find_split

    def _fast_find_split_regression(self, X, y, feature):
        x = X.values[:, feature]
        sort_idx = np.argsort(x)
        sort_y, sort_x = y.values[sort_idx], x[sort_idx]
        sum_y, n = sort_y.sum(), len(sort_y)
        sum_y_right, n_right = sum_y, n
        sum_y_left, n_left = 0.0, 0
        best_split = None
        best_score = float("inf")

        for i in range(n - self.min_sample_leaf):
            y_i, x_i, x_i_next = sort_y[i], sort_x[i], sort_x[i + 1]
            sum_y_left += y_i
            sum_y_right -= y_i
            n_left += 1
            n_right -= 1
            if n_left < self.min_sample_leaf or x_i == x_i_next:
                continue
            curr_score = -(sum_y_left**2) / n_left - (sum_y_right**2) / n_right + (sum_y**2) / n
            if curr_score < best_score:
                best_score = curr_score
                best_split = (x_i + x_i_next) / 2

        return {
            "feature": feature,
            "split": best_split,
            "score": best_score,
        }

    def _find_split(self, X, y, feature):
        x = X.values[:, feature]
        y_vls = y.values
        sort_idxs = np.argsort(x)  # indexes that sort x
        lhs = 0
        split = None
        score = float("inf")
        for i in range(len(sort_idxs) - self.min_sample_leaf):
            lhs += 1
            if lhs < self.min_sample_leaf or x[sort_idxs[i]] == x[sort_idxs[i + 1]]:
                continue
            curr_score = self.loss_function(y_vls, sort_idxs[: lhs + 1], sort_idxs[lhs + 1 :])
            if curr_score < score:
                score = curr_score
                split = (x[sort_idxs[i]] + x[sort_idxs[i + 1]]) * 0.5
        return {"feature": feature, "split": split, "score": score}
