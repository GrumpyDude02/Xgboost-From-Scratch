from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .timer import timer
from MyXgBoost import MyXgbModel
import xgboost as xgb
import numpy as np
from matplotlib import pyplot as plt


@timer
def run(params):
    X, y = fetch_california_housing(as_frame=True, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=43)
    params["base_prediction"] = np.mean(y)
    m = MyXgbModel(
        seed=43,
        parameters=params,
    )
    og = xgb.XGBModel(
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        n_estimators=params["n_estimators"],
        objective=params["objective"],
        reg_lambda=params["lambda"],
        subsample=params["subsample"],
        base_score=params["base_prediction"],
    )
    og.fit(X_train, y_train)
    metrics = m.fit(X_train, y_train, X_test, y_test)
    pred = m.predict(X_test)
    print("---------------------------My xgboost---------------------------")
    print(f"MSE:  {mean_squared_error(y_test, pred)}")
    print(f"MAE:  {mean_absolute_error(y_test, pred)}")
    print("----------------------original xgboost--------------------------")
    pred = og.predict(X_test)
    print(f"MSE:  {mean_squared_error(y_test, pred)}")
    print(f"MAE:  {mean_absolute_error(y_test, pred)}")
    plt.legend()
    plt.show()
