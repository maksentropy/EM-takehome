import pandas as pd
import numpy as np


class DataProcesser:
    """
    Class for handling pandas data.
    """

    def __init__(self, data=None) -> None:
        self.data = data

    def read_data(self, path: str) -> None:
        """
        Method for retreiving data from a source. In this case, it reads
        a csv file in specified path.
        """
        self.data = pd.read_csv(path)

    def get_columns_as_np(self, columns) -> np.ndarray:
        """
        Method for retreiving data for given columns
        """
        return self.data[columns].values

    def dropna_columns(self, columns=None) -> None:
        """
        Method for dropping rows if in the sepcified columns there are NaNs.
        """
        if columns is None:
            self.data.dropna(inplace=True)
        else:
            self.data.dropna(subset=columns, inplace=True)

    def fill_na(self, features=None, method=None) -> None:
        """
        Method for handling missing values in the data.
        Currently supports only inserting median value for specified feature set assming it's numeric.
        Otherwise uses all columns with NaNs.
        """
        if method is None:
            return

        if method == "median":

            if features is None:
                features = self.data.columns[self.data.isna().any()].tolist()

            for feature in features:
                fill_value = self.data[feature].median()
                self.data.loc[self.data[feature].isna(), feature] = fill_value
