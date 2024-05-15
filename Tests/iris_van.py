from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from MyXgBoost import MyDecisionTree
import numpy as np, pandas as pd
from .timer import timer


@timer
def run(params):
    mapping = {True: 1, False: 0}
    X, y = load_iris(as_frame=True, return_X_y=True)
    X["class"] = y
    X = pd.concat([X, pd.get_dummies(X["class"], prefix="class")], axis=1)
    for i in range(0, 3):
        X[f"class_{i}"] = X[f"class_{i}"].map(mapping)
    X.drop(columns=["class"], inplace=True)
    y = X.pop("sepal width (cm)")
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=43)
    m = MyDecisionTree(max_depth=params["max_depth"], min_sample_leaf=params["min_sample_leaf"])
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    print(f"MSE:  {mean_squared_error(y_test, pred)}")
    print(f"MAE:  {mean_absolute_error(y_test, pred)}")
