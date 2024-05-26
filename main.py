from Tests import (
    iris_boost,
    california_housing_boost,
    california_housing_van,
    iris_van,
    binary_class_test,
    california_housing_customvsog,
    iris_ogvscustom,
)


def main():
    parameters = {
        "learning_rate": 0.4,
        "max_depth": 5,
        "subsample": 0.8,
        "min_sample_leaf": 20,
        "lambda": 1.5,
        "gamma": 0.0,
        "min_child_weight": 25.0,
        "base_prediction": 0.5,
        "objective": "binary:logistic",
        "tree_method": "exact",
        "n_estimators": 10,
        "base_prediction": 0.5,
        "verbosity": False,
    }
    binary_class_test.run(parameters)


if __name__ == "__main__":
    main()
