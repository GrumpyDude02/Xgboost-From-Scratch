from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .timer import timer
from MyXgBoost import MyDecisionTree
import numpy as np, pandas as pd


@timer
def run(params):
    max_depth = params["max_depth"]
    min_sample_leaf = params["min_sample_leaf"]
    X, y = fetch_california_housing(as_frame=True, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=43)
    m = MyDecisionTree(max_depth=max_depth, min_sample_leaf=min_sample_leaf)
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    print(f"MSE:  {mean_squared_error(y_test, pred)}")
    print(f"MAE:  {mean_absolute_error(y_test, pred)}")
