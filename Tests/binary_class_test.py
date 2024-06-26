import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from MyXgBoost import MyXgbModel
from .timer import timer


@timer
def run(params):
    # Generate synthetic binary classification dataset
    assert params["objective"] != "reg:squarederror"
    X, y = make_classification(n_samples=10000, n_features=20, n_classes=2, random_state=42)

    X = pd.DataFrame(X)
    y = pd.Series(y)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

    params = params

    m = MyXgbModel(parameters=params, seed=42)

    # Train the model
    m.fit(X_train, y_train)

    # Predict on the test set
    y_pred = m.predict(X_test)

    print(y_pred)
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
