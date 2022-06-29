from typing import Any

import contractions
from pycld2 import detect
from sklearn.base import BaseEstimator

import pandas as pd


class NlpDetectLang(BaseEstimator):

    def __init__(self, text_column: str, lang_column: str) -> None:
        self.text_column = text_column
        self.lang_column = lang_column

    def fit(self, x: Any, y: Any = None) -> __qualname__:
        return self

    @staticmethod
    def detect_language(message: str) -> str:
        printable_message = "".join(x for x in message if x.isprintable())
        reliable, _, detected_language = detect(printable_message)
        if reliable:
            return detected_language[0][0]
        else:
            return 'UNKNOWN'

    def transform(self, x: Any) -> pd.DataFrame:
        x[self.lang_column] = x[self.text_column].map(self.detect_language)

        return x


class NlpTextExpansion(BaseEstimator):

    def __init__(self, text_column: str, lang_column: str) -> None:
        self.text_column = text_column
        self.lang_column = lang_column

    def fit(self, x: Any, y: Any = None) -> __qualname__:
        return self

    def transform(self, x: Any) -> pd.DataFrame:
        x[self.lang_column] = x[self.text_column].apply(lambda txt: contractions.fix(txt))

        return x
  

class NlpRemoveStopwords(BaseEstimator):
    pass


class NlpTextToToken(BaseEstimator):
    pass


class NlpWordLemmatizer(BaseEstimator):
    pass


class NlpWordStemmer(BaseEstimator):
    pass

