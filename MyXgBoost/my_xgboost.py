import numpy as np, pandas as pd, math
import MyXgBoost.objectives as objective
from MyXgBoost._Boostedtree import _BoostedTreeRegressor
from matplotlib import pyplot as plt, axes as axes
from matplotlib.backends.backend_agg import FigureCanvasAgg

# TODO [[maybe]]: implement softmax for multiclassification
font_size = 10


class MyXgbModel:
    """
        Parameters: default dict consists of : {
        "learning_rate": 0.1,
        "max_depth": 5,
        "subsample": 0.8,
        "min_sample_leaf":20,
        "lambda": 1.5,
        "gamma": 0.0,
        "min_child_weight": 25.0,
        "base_prediction": 0.5,
        "objective": "reg:squarederror",
        "tree_method": "exact",
        "n_estimators": 20,
        "eps": 0.03,
        "verbosity": False,
    }
    """

    objectives = {
        "reg:squarederror": objective.SquaredErrorObjective(),
        "binary:logistic": None,
    }

    default_parameters = {
        "learning_rate": 0.1,
        "max_depth": 5,
        "subsample": 0.8,
        "lambda": 1.5,
        "gamma": 0.0,
        "min_child_weight": 25.0,
        "min_sample_leaf": 20,
        "base_prediction": 0.5,
        "objective": "reg:squarederror",
        "tree_method": "exact",
        "n_estimators": 20,
        "eps": 0.03,
        "verbosity": False,
    }

    def __init__(self, parameters: dict = None, seed: int = None) -> None:
        self.parameters = MyXgbModel.default_parameters if parameters is None else parameters
        if parameters is not None:
            for key in MyXgbModel.default_parameters.keys():
                val = parameters.get(key)
                self.parameters[key] = val if val is not None else MyXgbModel.default_parameters[key]
            assert self.parameters["learning_rate"] >= 0 and self.parameters["learning_rate"] <= 1
            assert self.parameters["subsample"] >= 0 and self.parameters["subsample"] <= 1
            assert self.parameters["eps"] >= 0 and self.parameters["eps"] <= 1
        obj = MyXgbModel.objectives.get(self.parameters["objective"])
        if obj is None:
            raise ValueError("Undefined objective function. This model currently only supports regression tasks.")
        self.objective: objective.Objective = obj
        self.parameters = MyXgbModel.parameters if parameters is None else parameters
        self.verbose = self.parameters["verbosity"]
        self.rounds = self.parameters["n_estimators"]
        self.rng = np.random.default_rng(seed=seed)

    def fit(self, X: pd.DataFrame, y: pd.Series, X_val=None, y_val=None) -> None:
        offset = []
        offset_val = []
        curr_pred_val = None
        learning_rate = self.parameters["learning_rate"]
        size = math.floor(self.parameters["subsample"] * len(X))
        curr_pred = self.parameters["base_prediction"] * np.ones(shape=len(y))
        self.trees: list[_BoostedTreeRegressor] = []
        for i in range(self.rounds):
            if self.parameters["subsample"] < 1:
                idxs = self.rng.choice(len(y), size=size, replace=False)

            gradients = self.objective.gradient(y.values, curr_pred)
            hessians = self.objective.hessian(y.values, curr_pred)
            if idxs is not None:
                gradients, hessians = gradients[idxs], hessians[idxs]
                X_ = X.iloc[idxs]
            else:
                X_ = X
            tree = _BoostedTreeRegressor(
                X_,
                boost_parameters=self.parameters,
                gradients=gradients,
                hessians=hessians,
            )
            curr_pred += learning_rate * tree.predict(X)
            self.trees.append(tree)
            offset.append(self.objective.loss(y, curr_pred))
            if X_val is not None and y_val is not None:
                if curr_pred_val is None:
                    curr_pred_val = self.parameters["base_prediction"] * np.ones(shape=len(y_val))

                curr_pred_val += learning_rate * tree.predict(X_val)
                offset_val.append(self.objective.loss(y_val, curr_pred_val))
            if self.verbose:
                print(f"Iteration {i+1}: Loss: {self.objective.loss(y,curr_pred)}")

        return {"error_train": offset, "rounds": i + 1, "error_val": offset_val}

    def predict(self, X) -> np.ndarray:
        return self.objective.convert(
            self.parameters["base_prediction"]
            + self.parameters["learning_rate"] * np.sum([tree.predict(X) for tree in self.trees], axis=0)
        )

    def get_error_data(self, X: pd.DataFrame, y: pd.Series) -> dict:
        if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
            raise Exception("X dataset must of type pandas Dataframe and y must be type of pandas Series")
        if self.trees == []:
            raise Exception("No dataset was fitted, this method was designed to run after fitting a dataset")
        errors = []
        X_ = X.values
        y_ = y.values
        base = self.base_score * np.ones(shape=len(y))
        for e in self.estimators:
            base += self.learning_rate * e.predict(X_)
            errors.append(self.loss_function(base, y_))

        return {"rounds": len(self.estimators), "error": errors}

    def plot_tree(self, ax=None, num_trees=0):
        try:
            tree = self.trees[num_trees]
        except IndexError:
            raise Exception("Tree number exceeds the number of trees created")
        if ax is None:
            fig, ax = plt.subplots(figsize=(200, 100))
        self._rec_plot_tree(ax, tree.root, tree.max_depth, direction="left")

        return ax

    def _rec_plot_tree(
        self,
        ax: axes.Axes,
        node: _BoostedTreeRegressor.Node,
        depth,
        direction,
        x=0.5,
        y=1.0,
        dx=1,
        dy=0.1,
        level_width=0.2,
        font_size=10,
    ):
        if node.value is not None:
            offset = 0.015 if direction == "left" else 0
            ax.text(
                x,
                y - offset,
                f"Leaf: {node.value:.2f}",
                ha="center",
                va="center",
                fontsize=font_size,
                bbox=dict(facecolor="lightgreen", alpha=1, boxstyle="round"),
            )
        else:
            ax.text(
                x,
                y,
                f"X{node.feature} <= {node.split:.2f}\nGain: {node.score:.2f}",
                ha="center",
                va="center",
                fontsize=font_size,
                bbox=dict(facecolor="lightblue", alpha=1, boxstyle="round"),
            )
            new_level_width = level_width / 2
            new_y = y - dy
            new_dx = dx / 2

            ax.plot([x, x - new_dx], [y + dy / 24, new_y + dy / 24], "k-")  # Line to left child
            ax.plot([x, x + new_dx], [y + dy / 24, new_y + dy / 24], "k-")  # Line to right child

            ax.text(x - new_dx / 2 - 0.01, new_y + dy / 2, "Yes", ha="right", va="bottom", fontsize=font_size)
            ax.text(x + new_dx / 2 + 0.01, new_y + dy / 2, "No", ha="left", va="bottom", fontsize=font_size)

            self._rec_plot_tree(
                ax,
                node.left,
                depth - 1,
                "left",
                x - new_dx,
                new_y,
                new_dx,
                dy,
                level_width=new_level_width,
                font_size=font_size * 0.9,
            )
            self._rec_plot_tree(
                ax,
                node.right,
                depth - 1,
                "right",
                x + new_dx,
                new_y,
                new_dx,
                dy,
                level_width=new_level_width,
                font_size=font_size * 0.9,
            )
        ax.axis("off")
