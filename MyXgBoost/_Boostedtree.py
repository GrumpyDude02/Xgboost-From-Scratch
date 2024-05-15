from ._tree import _BaseTree
import numpy as np, pandas as pd


class _BoostedTreeRegressor(_BaseTree):

    def __init__(self, X: pd.DataFrame, boost_parameters, gradients, hessians) -> None:
        super().__init__(boost_parameters["max_depth"], None)
        self.global_X = X
        self.global_g = gradients
        self.global_h = hessians
        self.boost_parameters = boost_parameters
        self.gradiants, self.hessians = gradients, hessians
        self.min_child_weight = boost_parameters["min_child_weight"]
        self.gamma = boost_parameters["gamma"]
        self.lambda_ = boost_parameters["lambda"]
        self.eps = self.boost_parameters["eps"]
        method = self.boost_parameters["tree_method"]
        if method == "exact":
            self.find_split = self._exact_find_split
        elif method == "approx":
            self.find_split = self._weighted_quantile_sketch
        else:
            raise TypeError("Undifined Method")
        self.root = self._build_tree(X, gradients, hessians, self.max_depth)

    def _calculate_leaf_value(self, y):
        g, h = y
        return -g.sum() / (h.sum() + self.lambda_)

    def _build_tree(self, X, g, h, depth):
        if depth == 0:
            return _BoostedTreeRegressor.Node(value=self._calculate_leaf_value((g, h)))
        score = float("-inf")
        feature, split = None, None
        direction = "left"
        missing_idxs = None
        for i in range(X.shape[1]):
            data = self.find_split(X, g, h, i)
            if data["score"] > score:
                feature = data["feature"]
                split = data["split"]
                score = data["score"]
                missing_idxs = data["missing_idxs"]
                direction = data["direction"]
        if score == float("-inf"):
            return _BoostedTreeRegressor.Node(value=self._calculate_leaf_value((g, h)))
        x = X.iloc[:, feature]
        x = x[x != 0]
        left_idxs = x <= split
        right_idxs = x > split
        if missing_idxs is not None:
            if direction == "left":
                np.concatenate((left_idxs, missing_idxs))
            else:
                np.concatenate((right_idxs, missing_idxs))
        left = self._build_tree(X[left_idxs], g[left_idxs], h[left_idxs], depth - 1)
        right = self._build_tree(X[right_idxs], g[right_idxs], h[right_idxs], depth - 1)
        return _BoostedTreeRegressor.Node(left, right, split=split, feature=feature)

    def _exact_find_split(self, X: pd.DataFrame, g, h, feature):
        score, split, direction = float("-inf"), None, "left"
        x = X.values[:, feature]
        idxs = np.argsort(x[x != 0])
        missing_idxs = np.where(x == 0)
        x_sort = x[idxs]
        h_sum, g_sum = h.sum(), g.sum()
        left_idxs = 0

        g_cumsum, h_cumsum = np.cumsum(g[idxs]), np.cumsum(h[idxs])
        n = len(g_cumsum) - 1
        for i in range(n):
            if h_cumsum[i] < self.min_child_weight or x_sort[i + 1] == x_sort[i]:
                continue
            rhs_h = h_sum - h_cumsum[i]
            if rhs_h < self.min_child_weight:
                break
            rhs_g = g_sum - g_cumsum[i]

            right_score = (
                0.5
                * (
                    g_cumsum[i] ** 2 / (h_cumsum[i] + self.lambda_)
                    + rhs_g**2 / (rhs_h + self.lambda_)
                    - g_sum**2 / (h_sum + self.lambda_)
                )
                - self.gamma
            )
            if right_score > score:
                left_idxs = i
                split = (x_sort[i] + x_sort[i + 1]) * 0.5
                direction = "right"
                score = right_score

        for i in range(n, 1):
            if h_cumsum[i] < self.min_child_weight or x_sort[i - 1] == x_sort[i]:
                continue
            lhs_h = h_sum - h_cumsum[i]

            if lhs_h < self.min_child_weight:
                break
            lhs_g = g_sum - g_cumsum[i]
            left_score = (
                0.5
                * (
                    g_cumsum[i] ** 2 / (h_cumsum[i] + self.lambda_)
                    + lhs_g**2 / (lhs_h + self.lambda_)
                    - g_sum**2 / (h_sum + self.lambda_)
                )
                - self.gamma
            )
            if right_score > score:
                left_idxs = i
                split = (x_sort[i] + x_sort[i - 1]) * 0.5
                direction = "left"
                score = left_score

        return {"feature": feature, "split": split, "score": score, "direction": direction, "missing_idxs": missing_idxs}

    def _sparse_score(self, g_sum, h_sum, right_g, left_g, right_h, left_h, nan_g_sum, nan_h_sum):
        # excuse the repetition, function calls are slow
        right_score = (
            0.5
            * (
                left_g**2 / (left_h + self.lambda_)
                + (right_g + nan_g_sum) ** 2 / ((right_h + nan_g_sum) + self.lambda_)
                - g_sum**2 / (h_sum + self.lambda_)
            )
            - self.gamma
        )
        left_score = (
            0.5
            * (
                (left_g + nan_g_sum) ** 2 / (left_h + nan_g_sum + self.lambda_)
                + right_g**2 / (right_h + self.lambda_)
                - g_sum**2 / (h_sum + self.lambda_)
            )
            - self.gamma
        )
        return max(left_score, right_score)

    def _calculate_score(self, g_sum, h_sum, right_g, left_g, right_h, left_h, nan_g_sum: float = 0, nan_h_sum: float = 0):
        return (
            0.5
            * (
                left_g**2 / (left_h + self.lambda_)
                + right_g**2 / (right_h + self.lambda_)
                - g_sum**2 / (h_sum + self.lambda_)
            )
            - self.gamma
        )
