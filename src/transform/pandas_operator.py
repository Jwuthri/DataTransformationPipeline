from typing import List, Dict, Any

from sklearn.base import BaseEstimator

import pandas as pd


class DataFrameReadCsv(BaseEstimator):
    def __init__(self, path: str) -> None:
        self.path = path

    def fit(self, x: Any, y: Any = None) -> __qualname__:
        return self

    def transform(self, x: Any) -> pd.DataFrame:
        return pd.read_csv(self.path)


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
    def __init__(self, text_column: str) -> None:
        self.text_column = text_column

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
    def __init__(self, text_column: str) -> None:
        self.text_column = text_column

    def fit(self, x: Any, y: Any = None) -> __qualname__:
        return self

    def transform(self, x: Any) -> pd.DataFrame:
        x = x.explode(self.text_column)

        return x


class DataFrameQueryFilter(BaseEstimator):
    def __init__(self, text_column: str, query: str) -> None:
        self.text_column = text_column
        self.query = query

    def fit(self, x: Any, y: Any = None) -> __qualname__:
        return self

    def transform(self, x: Any) -> pd.DataFrame:
        x = x.query(f"{self.text_column} {self.query}")

        return x


class DataFrameInplodeColumn(BaseEstimator):
    def __init__(self, key_column: str, agg_column: str) -> None:
        self.key_column = key_column
        self.agg_column = agg_column

    def fit(self, x: Any, y: Any = None) -> __qualname__:
        return self

    def transform(self, x: Any) -> pd.DataFrame:
        x = x.groupby([self.key_column]).agg({self.agg_column: lambda x: x.tolist()}).reset_index()

        return x


class DataFrameToCsv(BaseEstimator):
    def __init__(self, output_path: str) -> None:
        self.output_path = output_path

    def fit(self, x, y=None) -> __qualname__:
        return self

    def transform(self, x) -> pd.DataFrame:
        x.to_csv(self.output_path, index=False)

        return x
