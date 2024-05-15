from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .timer import timer
from MyXgBoost import MyXgbModel
import xgboost as xgb
import numpy as np, pandas as pd


@timer
def run(method_criterion: str, verbose=False, rounds=10):
    X, y = fetch_california_housing(as_frame=True, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=43)
    m = MyXgbModel(
        seed=43,
        parameters={
            "subsample": 0.8,
            "base_prediction": np.mean(y.values),
            "learning_rate": 0.4,
            "n_estimators": rounds,
            "objective": "reg:squarederror",
            "tree_method": method_criterion,
            "verbosity": verbose,
        },
    )
    og = xgb.XGBModel(
        max_depth=5, learning_rate=0.4, n_estimators=rounds, objective="reg:squarederror", reg_lambda=1.5, subsample=0.8
    )
    og.fit(X_train, y_train)
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    print(f"MSE:  {mean_squared_error(y_test, pred)}")
    print(f"MAE:  {mean_absolute_error(y_test, pred)}")
    print("----------------------original xgboost--------------------------")
    pred = og.predict(X_test)
    print(f"MSE:  {mean_squared_error(y_test, pred)}")
    print(f"MAE:  {mean_absolute_error(y_test, pred)}")
