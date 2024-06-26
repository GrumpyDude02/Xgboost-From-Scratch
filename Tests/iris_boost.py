from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from MyXgBoost import MyXgbModel
import numpy as np, pandas as pd
from matplotlib import pyplot as plt


def run(params):
    mapping = {True: 1, False: 0}
    X, y = load_iris(as_frame=True, return_X_y=True)
    X["class"] = y
    X = pd.concat([X, pd.get_dummies(X["class"], prefix="class")], axis=1)
    for i in range(0, 3):
        X[f"class_{i}"] = X[f"class_{i}"].map(mapping)
    X.drop(columns=["class"], inplace=True)
    y = X.pop("petal width (cm)")
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.77, random_state=37)
    m = MyXgbModel(seed=43, parameters=params)
    errors = m.fit(X_train, y_train, X_test, y_test)
    plt.plot(np.arange(1, errors["rounds"] + 1), errors["error_train"], label="Train dataset error")
    plt.plot(np.arange(1, errors["rounds"] + 1), errors["error_val"], label="Validation dataset error")
    plt.xlabel("Rounds/Iterations")
    plt.ylabel("Error")
    plt.legend()
    m.plot_importance("weight")
    m.plot_tree(num_trees=params["n_estimators"] - 1)
    pred = m.predict(X_test)
    print(f"MSE:  {mean_squared_error(y_test, pred)}")
    print(f"MAE:  {mean_absolute_error(y_test, pred)}")
    plt.show()
