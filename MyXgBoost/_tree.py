import numpy as np, pandas as pd


class _BaseTree:

    class Node:
        def __init__(self, left=None, right=None, value=None, split=None, feature=None) -> None:
            self.split = split
            self.value = value
            self.feature = feature
            self.left = left
            self.right = right

    def __init__(self, max_depth, min_sample_leaf) -> None:
        self.min_sample_leaf = min_sample_leaf
        self.max_depth = max_depth
        self.root = None
        self._split_function = None

    def _predict_row(self, x, node: Node):
        if node.value is not None:
            return node.value
        if x.iloc[node.feature] <= node.split:
            return self._predict_row(x, node.left)
        return self._predict_row(x, node.right)

    def _build_tree(self, X, y, depth):
        score = float("inf")
        left_idxs, right_idxs, feature, split = None, None, None, None
        if depth == 0 or len(X) < self.min_sample_leaf:
            return _BaseTree.Node(value=self._calculate_leaf_value(y))
        for i in range(X.shape[1]):
            data = self._split_function(X, y, i)
            if data["score"] < score:
                feature = data["feature"]
                split = data["split"]
                score = data["score"]
        if score == float("inf"):
            return _BaseTree.Node(value=self._calculate_leaf_value(y))
        x = X.iloc[:, feature]
        left_idxs = x <= split
        right_idxs = x > split
        left = self._build_tree(X[left_idxs], y[left_idxs], depth - 1)
        right = self._build_tree(X[right_idxs], y[right_idxs], depth - 1)
        return _BaseTree.Node(left, right, split=split, feature=feature)

    def fit(self, X, y):
        self.root = self._build_tree(X, y, self.max_depth)

    def predict(self, X):
        return np.array([self._predict_row(x, self.root) for _, x in X.iterrows()])

    def _calculate_leaf_value(self, y):
        raise Exception("Not Implemented")


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
        for i in range(X.shape[1]):
            data = self.find_split(X, g, h, i)
            if data["score"] > score:
                feature = data["feature"]
                split = data["split"]
                score = data["score"]
        if score == float("-inf"):
            return _BoostedTreeRegressor.Node(value=self._calculate_leaf_value((g, h)))
        x = X.iloc[:, feature]
        left_idxs = x <= split
        right_idxs = x > split
        left = self._build_tree(X[left_idxs], g[left_idxs], h[left_idxs], depth - 1)
        right = self._build_tree(X[right_idxs], g[right_idxs], h[right_idxs], depth - 1)
        return _BoostedTreeRegressor.Node(left, right, split=split, feature=feature)

    def _weighted_quantile_sketch(self, X: pd.DataFrame, g, h, feature):
        score = float("-inf")
        split = None
        x = X.values[:, feature]
        idxs = np.argsort(x)
        h_sum, g_sum = h.sum(), g.sum()
        nan_flag = np.isnan(x[idxs]).sum() == 0
        if nan_flag:
            nan_g_sum, nan_h_sum = 0, 0
            calc_score = self._calculate_score
        else:
            nan_idxs = idxs[len(idxs) - np.isnan(x[idxs]).sum() :]
            idxs = idxs[: -np.isnan(x[idxs]).sum()]
            nan_g_sum, nan_h_sum = g[nan_idxs].sum(), h[nan_idxs].sum()
            calc_score = self._sparse_score
        inv_h_sum = 1 / (h_sum + 1e-16)
        g_cumsum, h_cumsum, x_sort = np.cumsum(g[idxs]), np.cumsum(h[idxs]), x[idxs]

        diff = np.abs(h_cumsum[:-1] * inv_h_sum - h_cumsum[1:] * inv_h_sum)
        precomputed_indices = np.where(diff < self.eps)[0]

        for i in precomputed_indices:
            if h_cumsum[i] < self.min_child_weight:
                continue
            rhs_h_sum = h_sum - h_cumsum[i]
            if rhs_h_sum < self.min_child_weight:
                break
            rhs_g_sum = g_sum - g_cumsum[i]
            curr_score = calc_score(g_sum, h_sum, rhs_g_sum, g_cumsum[i], rhs_h_sum, h_cumsum[i], nan_g_sum, nan_h_sum)
            if curr_score > score:
                score = curr_score
                split = (x_sort[i] + x_sort[i + 1]) * 0.5

        return {"feature": feature, "split": split, "score": score}

    def _exact_find_split(self, X: pd.DataFrame, g, h, feature):
        score = float("-inf")
        split = None
        x = X.values[:, feature]
        h_sum, g_sum = h.sum(), g.sum()
        idxs = np.argsort(x)
        nan_flag = np.isnan(x[idxs]).sum() == 0
        if nan_flag:
            nan_g_sum, nan_h_sum = 0, 0
            calc_score = self._calculate_score
        else:
            calc_score = self._sparse_score
            nan_idxs = idxs[len(idxs) - np.isnan(x[idxs]).sum() :]
            idxs = idxs[: -np.isnan(x[idxs]).sum()]
            nan_g_sum, nan_h_sum = g[nan_idxs].sum(), h[nan_idxs].sum()
        g_cumsum, h_cumsum, x_sort = np.cumsum(g[idxs]), np.cumsum(h[idxs]), x[idxs]
        n = len(g_cumsum) - 1
        for i in range(n):
            if h_cumsum[i] < self.min_child_weight or x_sort[i + 1] == x_sort[i]:
                continue
            rhs_h_sum = h_sum - h_cumsum[i]
            if rhs_h_sum < self.min_child_weight:
                break
            rhs_g_sum = g_sum - g_cumsum[i]
            curr_score = calc_score(g_sum, h_sum, rhs_g_sum, g_cumsum[i], rhs_h_sum, h_cumsum[i], nan_g_sum, nan_h_sum)
            if curr_score > score:
                score = curr_score
                split = (x_sort[i + 1] + x_sort[i]) * 0.5

        return {"feature": feature, "split": split, "score": score}

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
