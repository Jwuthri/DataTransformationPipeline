from typing import List, Dict, Any
import re

from sklearn.base import BaseEstimator

import pandas as pd


class DataFrameColumnsSelection(BaseEstimator):

    def __init__(self, columns: List[str]) -> None:
        self.columns = columns

    def fit(self, x: Any, y: Any = None) -> __qualname__:
        return self

    def transform(self, x: Any) -> pd.DataFrame:
        return x[self.columns]


class DataFrameColumnsDrop(BaseEstimator):

    def __init__(self, columns: List[str]) -> None:
        self.columns = columns

    def fit(self, x: Any, y: Any = None) -> __qualname__:
        return self

    def transform(self, x: Any) -> pd.DataFrame:
        return x.drop(columns=self.columns)


class DataFrameColumnsRename(BaseEstimator):

    def __init__(self, columns_mapping: Dict[str, str]) -> None:
        self.columns_mapping = columns_mapping

    def fit(self, x: Any, y: Any = None) -> __qualname__:
        return self

    def transform(self, x: Any) -> pd.DataFrame:
        return x.rename(columns=self.columns_mapping)


class DataFrameTextFormat(BaseEstimator):

    def __init__(self, text_column: str, new_column: str = None, format: str = "lower") -> None:
        assert format.lower() in ["lower", "upper", "capitalize"]
        self.format = format.lower()
        self.text_column = text_column
        if new_column is None:
            self.new_column = text_column
        else:
            self.new_column = new_column

    def fit(self, x: Any, y: Any = None) -> __qualname__:
        return self

    def transform(self, x: Any) -> pd.DataFrame:
        x[self.new_column] = getattr(x[self.text_column].str, self.format)()

        return x


class DataFrameDropEmptyRows(BaseEstimator):

    def __init__(self, text_column: str, new_column: str = None) -> None:
        self.text_column = text_column
        if new_column is None:
            self.new_column = text_column
        else:
            self.new_column = new_column

    def fit(self, x: Any, y: Any = None) -> __qualname__:
        return self

    def transform(self, x: Any) -> pd.DataFrame:
        return x[x[self.text_column].notnull()]


class DataFrameTextLength(BaseEstimator):

    def __init__(self, text_column: str, new_column: str = None) -> None:
        self.text_column = text_column
        if new_column is None:
            self.new_column = text_column
        else:
            self.new_column = new_column

    def fit(self, x: Any, y: Any = None) -> __qualname__:
        return self

    def transform(self, x: Any) -> pd.DataFrame:
        x[self.new_column] = x[self.text_column].str.len()
    
        return x


class DataFrameTextNumberWords(BaseEstimator):

    def __init__(self, text_column: str, new_column: str = None) -> None:
        self.text_column = text_column
        if new_column is None:
            self.new_column = text_column
        else:
            self.new_column = new_column

    def fit(self, x: Any, y: Any = None) -> __qualname__:
        return self

    def transform(self, x: Any) -> pd.DataFrame:
        x[self.new_column] = x[self.text_column].str.split().str.len()

        return x


class DataFrameValueFrequency(BaseEstimator):

    def __init__(self, text_column: str, new_column: str = None) -> None:
        self.text_column = text_column
        if new_column is None:
            self.new_column = text_column
        else:
            self.new_column = new_column

    def fit(self, x: Any, y: Any = None) -> __qualname__:
        return self

    def transform(self, x: Any) -> pd.DataFrame:
        x[self.new_column] = x.groupby(self.text_column)[self.text_column].transform("count")

        return x


class DataFrameExplodeColumn(BaseEstimator):

    def __init__(self, text_column: str, new_column: str = None) -> None:
        self.text_column = text_column
        if new_column is None:
            self.new_column = text_column
        else:
            self.new_column = new_column

    def fit(self, x: Any, y: Any = None) -> __qualname__:
        return self

    def transform(self, x: Any) -> pd.DataFrame:
        x = x.explode(self.text_column)

        return x


class DataFrameQueryFilter(BaseEstimator):

    def __init__(self, text_column: str, new_column: str = None, query: str = "< 10") -> None:
        self.text_column = text_column
        self.query = query
        if new_column is None:
            self.new_column = text_column
        else:
            self.new_column = new_column

    def fit(self, x: Any, y: Any = None) -> __qualname__:
        return self

    def transform(self, x: Any) -> pd.DataFrame:
        x = x.query(f"{self.text_column} {self.query}")

        return x


class DataFrameDeDuplicatesSpace(BaseEstimator):

    def __init__(self, text_column: str, new_column: str = None) -> None:
        self.text_column = text_column
        if new_column is None:
            self.new_column = text_column
        else:
            self.new_column = new_column

    def fit(self, x: Any, y: Any = None) -> __qualname__:
        return self

    @staticmethod
    def remove_multiple_spaces(text: str):
        return re.sub(" {2,}", " ", text)

    def transform(self, x: Any) -> pd.DataFrame:
        x[self.new_column] = x[self.text_column].map(self.remove_multiple_spaces)

        return x
        

class DatFrameMergeColumns(BaseEstimator):
    pass


class DataFrameInplodeColumn(BaseEstimator):
    pass
