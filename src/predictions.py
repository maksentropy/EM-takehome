import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
import yaml
from data_processing import DataProcesser
from model import CarbonRegressor


with open("config.yaml", "r") as file:
    settings = yaml.safe_load(file)


def train_model() -> CarbonRegressor:
    """
    The function retreives training data with DataProcesser and trains model.
    """

    data_train_path = settings["path_train"]
    data_obj = DataProcesser()
    data_obj.read_data(path=data_train_path)

    target = settings["target"]
    data_obj.dropna_columns(columns=target)
    features = settings["features"]
    data_obj.fill_na(features=features, method="median")

    X = data_obj.get_columns_as_np(columns=features)
    y = data_obj.get_columns_as_np(columns=target)

    reg = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("gradient_boosting", GradientBoostingRegressor()),
        ]
    )
    model = CarbonRegressor(model=reg)
    model.train_model(X, y)

    return model


def make_predictions(model: CarbonRegressor) -> np.ndarray:
    """
    The function retreives test data with DataProcesser and makes predictions with model.
    """

    data_path = settings["path_test"]
    data_obj = DataProcesser()
    data_obj.read_data(path=data_path)

    features = settings["features"]
    data_obj.fill_na(features=features, method="median")
    X = data_obj.get_columns_as_np(columns=features)

    predictions = model.predict(X[-24:, :])

    return predictions


if __name__ == "__main__":
    model = train_model()
    predictions = make_predictions(model=model)
    print(predictions)
