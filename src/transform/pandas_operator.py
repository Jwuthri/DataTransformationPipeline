from typing import List, Any

from sklearn.base import BaseEstimator

import pandas as pd


class DataFrameColumnsSelection(BaseEstimator):

    def __init__(self, columns: List[str]) -> None:
        self.columns = columns

    def fit(self, x: Any, y: Any = None) -> __qualname__:
        return self

    def transform(self, x: Any) -> pd.DataFrame:
        return x[self.columns]


class DataFrameTextFormat(BaseEstimator):

    def __init__(self, columns: List[str], format: str = "lower") -> None:
        assert format.lower() in ["lower", "upper", "capitalize"]
        self.columns = columns
        self.format = format.lower()

    def fit(self, x: Any, y: Any = None) -> __qualname__:
        return self

    def transform(self, x: Any) -> pd.DataFrame:
        x[self.columns] = getattr(x[self.columns].str, self.format)()

        return x


class DataFrameDropEmptyRows(BaseEstimator):

    def __init__(self, columns: List[str]) -> None:
        self.columns = columns

    def fit(self, x: Any, y: Any = None) -> __qualname__:
        return self

    def transform(self, x: Any) -> pd.DataFrame:
        return x[x[self.columns].notnull()]


class DataFrameTextLength(BaseEstimator):

    def __init__(self, text_column: str, length_column: str) -> None:
        self.text_column = text_column
        self.length_column = length_column

    def fit(self, x: Any, y: Any = None) -> __qualname__:
        return self

    def transform(self, x: Any) -> pd.DataFrame:
        x[self.length_column] = x[self.text_column].str.len()
    
        return x


class DataFrameTextNumberWords(BaseEstimator):

    def __init__(self, text_column: str, number_column: str) -> None:
        self.text_column = text_column
        self.number_column = number_column

    def fit(self, x: Any, y: Any = None) -> __qualname__:
        return self

    def transform(self, x: Any) -> pd.DataFrame:
        x[self.number_column] = x[self.text_column].str.split().str.len()

        return x
