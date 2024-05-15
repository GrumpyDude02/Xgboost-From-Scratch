import numpy as np, pandas as pd, math
import MyXgBoost.objectives as Objective
from MyXgBoost._Boostedtree import _BoostedTreeRegressor

# TODO [[maybe]]: implement softmax for multiclassification


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
        "reg:squarederror": Objective.SquaredErrorObjective(),
        "binary:logistic": None,
    }

    parameters = {
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
        self.parameters = MyXgbModel.parameters if parameters is None else parameters
        if parameters is not None:
            for key in MyXgbModel.parameters.keys():
                val = parameters.get(key)
                self.parameters[key] = val if val is not None else MyXgbModel.parameters[key]
            assert self.parameters["learning_rate"] >= 0 and self.parameters["learning_rate"] <= 1
            assert self.parameters["subsample"] >= 0 and self.parameters["subsample"] <= 1
            assert self.parameters["eps"] >= 0 and self.parameters["eps"] <= 1
        obj = MyXgbModel.objectives.get(self.parameters["objective"])
        if obj is None:
            raise ValueError("Undefined objective function. This model currently only supports regression tasks.")
        self.objective = obj
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
        self.trees = []
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
        return self.objective.activation_function(
            self.parameters["base_prediction"]
            + self.parameters["learning_rate"] * np.sum([tree.predict(X) for tree in self.trees], axis=0)
        )


# model.fit(X_train, y_train, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=True)
