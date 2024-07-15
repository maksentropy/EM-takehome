import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from typing import Dict


class CarbonRegressor:
    """
    Class wraper for a ML model.
    """

    def __init__(self, model: Pipeline) -> None:
        self.model = model

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        self.model.fit(X_train, y_train.ravel())

    def predict(self, X_predict: np.ndarray) -> np.ndarray:
        return self.model.predict(X_predict)

    def score(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Wrapper for scoring accuracy of a model.
        """
        pred = self.model.predict(X)

        metrics = {
            "mae": mean_absolute_error(y_pred=pred, y_true=y),
            "mape": mean_absolute_percentage_error(y_pred=pred, y_true=y),
        }
        return metrics
