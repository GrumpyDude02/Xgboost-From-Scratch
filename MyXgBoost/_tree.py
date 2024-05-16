import numpy as np, pandas as pd


class _BaseTree:

    class Node:
        def __init__(self, left=None, right=None, value=None, split=None, feature=None, score=None) -> None:
            self.split = split
            self.value = value
            self.feature = feature
            self.left = left
            self.right = right
            self.score = score

    def __init__(self, max_depth: int, min_sample_leaf: int) -> None:
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
        return _BaseTree.Node(left, right, split=split, feature=feature, score=score)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Node:
        self.root = self._build_tree(X, y, self.max_depth)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([self._predict_row(x, self.root) for _, x in X.iterrows()])

    def _calculate_leaf_value(self, y):
        raise Exception("Not Implemented")
