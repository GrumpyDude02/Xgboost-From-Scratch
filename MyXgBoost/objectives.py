import numpy as np
from MyXgBoost.utils import sigmoid


class SquaredErrorObjective:

    def loss(self, y: np.ndarray, pred: np.ndarray) -> np.ndarray:
        return np.mean((y - pred) ** 2)

    def gradient(self, y: np.ndarray, pred: np.ndarray) -> np.ndarray:
        return pred - y

    def hessian(self, y: np.ndarray, pred: np.ndarray) -> np.ndarray:
        return np.ones(len(y))

    def activation_function(self, pred: np.ndarray) -> np.ndarray:
        return pred


class ClassificationObjective:

    def loss(self, y: np.ndarray, pred: np.ndarray) -> np.ndarray:
        preds = sigmoid(pred)
        return -np.mean(y * np.log(preds) + (1 - y) * np.log(1 - preds))

    def gradient(self, y: np.ndarray, preds: np.ndarray) -> np.ndarray:
        preds = sigmoid(preds)
        return preds - y

    def hessian(self, y: np.ndarray, preds: np.ndarray) -> np.ndarray:
        preds = sigmoid(preds)
        return preds * (1 - preds)

    def activation_function(self, preds: np.ndarray) -> np.ndarray:
        predicted_probas = sigmoid(np.full((preds.shape[0], 1), 1).flatten().astype("float64") + preds)
        final_preds = np.where(predicted_probas > np.mean(predicted_probas), 1, 0)
        return final_preds
