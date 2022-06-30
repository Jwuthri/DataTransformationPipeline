from typing import List, Dict, Any

from sklearn.base import BaseEstimator

import pandas as pd


class DataFrameColumnsSelection(BaseEstimator):

    def __init__(self, columns: List[str]) -> None:
        self.columns = columns

    def fit(self, x: Any, y: Any = None) -> __qualname__:
        return self

    def transform(self, x: Any) -> pd.DataFrame:
        return x[self.columns]


class DataFrameDropColumns(BaseEstimator):

    def __init__(self, columns: List[str]) -> None:
        self.columns = columns

    def fit(self, x: Any, y: Any = None) -> __qualname__:
        return self

    def transform(self, x: Any) -> pd.DataFrame:
        return x.drop(columns=self.columns)


class PandasRenameColumns(BaseEstimator):

    def __init__(self, columns_mapping: Dict[str, str]) -> None:
        self.colucolumns_mappingmns = columns_mapping

    def fit(self, x: Any, y: Any = None) -> __qualname__:
        return self

    def transform(self, x: Any) -> pd.DataFrame:
        return x.rename(columns=self.columns_mapping)


class DataFrameTextFormat(BaseEstimator):

    def __init__(self, column: str, format: str = "lower") -> None:
        assert format.lower() in ["lower", "upper", "capitalize"]
        self.column = column
        self.format = format.lower()

    def fit(self, x: Any, y: Any = None) -> __qualname__:
        return self

    def transform(self, x: Any) -> pd.DataFrame:
        x[self.column] = getattr(x[self.column].str, self.format)()

        return x


class DataFrameDropEmptyRows(BaseEstimator):

    def __init__(self, column: str) -> None:
        self.column = column

    def fit(self, x: Any, y: Any = None) -> __qualname__:
        return self

    def transform(self, x: Any) -> pd.DataFrame:
        return x[x[self.column].notnull()]


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


class DataFrameValueFrequency(BaseEstimator):
    pass
