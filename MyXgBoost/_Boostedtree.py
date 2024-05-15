from ._tree import _BaseTree
import numpy as np, pandas as pd


class _BoostedTreeRegressor(_BaseTree):

    def __init__(self, X: pd.DataFrame, boost_parameters: dict, gradients: np.ndarray, hessians: np.ndarray) -> None:
        super().__init__(boost_parameters["max_depth"], None)
        self.global_X = X
        self.global_g = gradients
        self.global_h = hessians
        self.boost_parameters = boost_parameters
        self.gradiants, self.hessians = gradients, hessians
        self.min_sample_leaf = boost_parameters["min_sample_leaf"]
        self.min_child_weight = boost_parameters["min_child_weight"]
        self.gamma = boost_parameters["gamma"]
        self.lambda_ = boost_parameters["lambda"]
        self.eps = self.boost_parameters["eps"]
        method = self.boost_parameters["tree_method"]
        if method == "exact":
            self.find_split = self._exact_find_split
        else:
            raise TypeError("Undifined Method")
        self.root = self._build_tree(X, gradients, hessians, self.max_depth)

    def _calculate_leaf_value(self, g: np.ndarray, h: np.ndarray):
        return -g.sum() / (h.sum() + self.lambda_)

    def _build_tree(self, X: pd.DataFrame, g: np.ndarray, h: np.ndarray, depth: int):
        if depth == 0:
            return _BoostedTreeRegressor.Node(value=self._calculate_leaf_value(g, h))
        score = float("-inf")
        feature, split = None, None
        for i in range(X.shape[1]):
            data = self.find_split(X, g, h, i)
            if data["score"] > score:
                feature = data["feature"]
                split = data["split"]
                score = data["score"]
                left_idxs = data["left_idxs"]
                right_idxs = data["right_idxs"]
        if score == float("-inf"):
            return _BoostedTreeRegressor.Node(value=self._calculate_leaf_value(g, h))

        left = self._build_tree(X.iloc[left_idxs], g[left_idxs], h[left_idxs], depth - 1)
        right = self._build_tree(X.iloc[right_idxs], g[right_idxs], h[right_idxs], depth - 1)
        return _BoostedTreeRegressor.Node(left, right, split=split, feature=feature)

    def _exact_find_split(self, X: pd.DataFrame, g, h, feature):
        score, split, direction = float("-inf"), None, "left"
        x = X.values[:, feature]
        idxs = np.argsort(x[x != 0])
        missing_idxs = np.where(x == 0)[0]
        x_sort = x[idxs]
        h_sum, g_sum = h.sum(), g.sum()
        split_idx = 0
        left_idxs, right_idxs = None, None

        g_cumsum, h_cumsum = np.cumsum(g[idxs]), np.cumsum(h[idxs])
        n = len(g_cumsum) - self.min_sample_leaf - 1
        for i in range(n):
            if h_cumsum[i] < self.min_child_weight or i < self.min_sample_leaf or x_sort[i + 1] == x_sort[i]:
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
                split_idx = i
                split = (x_sort[i] + x_sort[i + 1]) * 0.5
                direction = "right"
                score = right_score

        for i in range(n, 1):
            if h_cumsum[i] < self.min_child_weight or x_sort[i - 1] == x_sort[i] or i < self.min_sample_leaf:
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
            if left_score > score:
                split_idx = i
                split = (x_sort[i] + x_sort[i - 1]) * 0.5
                direction = "left"
                score = left_score
        if direction == "left":
            left_idxs = np.concatenate((idxs[:split_idx], missing_idxs))
            right_idxs = idxs[split_idx:]
        else:
            right_idxs = np.concatenate((idxs[split_idx:], missing_idxs))
            left_idxs = idxs[:split_idx]
        return {"feature": feature, "split": split, "score": score, "left_idxs": left_idxs, "right_idxs": right_idxs}
