from typing import Any, Dict

import emot
import spacy
import contractions

from pycld2 import detect
from sklearn.base import BaseEstimator

import pandas as pd


class NlpDetectLanguage(BaseEstimator):

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


class NlpWordExpansion(BaseEstimator):

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


class NlpTextToWords(BaseEstimator):
    pass


class NlpSpeechTagging(BaseEstimator):

    def __init__(self, column: str, new_column: str) -> None:
        self.column = column
        self.new_column = new_column
    
    def fit(self, x: Any, y: Any = None) -> __qualname__:
        self.nlp = spacy.load("en_core_web_sm")

        return self

    def pos(self, text: str) -> Dict[str, str]:
        doc = self.nlp(text)
        pos_tagging = []
        for token in doc:
            pos_tagging.append(
                [token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.sentiment, token.shape_, token.is_alpha, token.is_stop]
            )

        return pd.DataFrame(pos_tagging, columns=['token', 'lemma', 'pos', 'tag', 'dependency', 'sentiment', 'shape', 'is_alpha', 'is_stopwords']).to_dict('records')


    def transform(self, x: Any) -> pd.DataFrame:
        x[self.new_column] = x[self.column].map(self.pos)

        return x


class NlpWordLemmatizer(BaseEstimator):
    
    def __init__(self, column: str) -> None:
        self.column = column

    def fit(self, x: Any, y: Any = None) -> __qualname__:
        self.nlp = spacy.load("en_core_web_sm")

        return self
    
    def lemmatize(self, text: str) -> str:
        doc = self.nlp(text)
        tokens = [token.lemma_ for token in doc]

        return " ".join(tokens)

    def transform(self, x: Any) -> pd.DataFrame:
        x[self.column] = x[self.column].map(self.lemmatize)

        return x


class NlpReplaceEmojis(BaseEstimator):

    def __init__(self, column: str, how: str = "replace") -> None:
        self.column = column
        self.how = how

    def fit(self, x: Any, y: Any = None) -> __qualname__:
        self.emot_obj = emot.core.emot()

        return self

    def clean_emojis(self, text: str) -> str:
        emojis = self.emot_obj.emoji(text)
        if emojis['flag']:
            for index in range(len(emojis) - 1, -1, -1):
                target_value = emojis['mean'][index] if self.how == "replace" else ""
                start, end = emojis['location'][index]
                text = text[:start] + target_value + text[end:]

        return text

    def transform(self, x: Any) -> pd.DataFrame:
        x[self.column] = x[self.column].map(self.clean_emojis)

        return x


class NlpReplaceEmoticons(BaseEstimator):

    def __init__(self, column: str, how: str = "replace") -> None:
        self.column = column
        self.how = how

    def fit(self, x: Any, y: Any = None) -> __qualname__:
        self.emot_obj = emot.core.emot()

        return self

    def clean_emoticons(self, text: str) -> str:
        emoticons = self.emot_obj.emoticons(text)
        if emoticons['flag']:
            for index in range(len(emoticons) - 1, -1, -1):
                target_value = emoticons['mean'][index] if self.how == "replace" else ""
                start, end = emoticons['location'][index]
                text = text[:start] + target_value + text[end:]

        return text

    def transform(self, x: Any) -> pd.DataFrame:
        x[self.column] = x[self.column].map(self.clean_emoticons)

        return x
