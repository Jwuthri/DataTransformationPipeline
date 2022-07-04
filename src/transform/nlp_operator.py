from typing import Any, Dict, List

import emot
import spacy
import contractions

from pycld2 import detect
from sklearn.base import BaseEstimator

import pandas as pd


class NlpDetectLanguage(BaseEstimator):

    def __init__(self, text_column: str, new_column: str = None) -> None:
        self.text_column = text_column
        if new_column is None:
            self.new_column = text_column
        else:
            self.new_column = new_column

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
        x[self.new_column] = x[self.text_column].map(self.detect_language)

        return x


class NlpWordExpansion(BaseEstimator):

    def __init__(self, text_column: str, new_column: str = None) -> None:
        self.text_column = text_column
        self.new_column = new_column

    def fit(self, x: Any, y: Any = None) -> __qualname__:
        return self

    def transform(self, x: Any) -> pd.DataFrame:
        x[self.new_column] = x[self.text_column].apply(contractions.fix)

        return x
  

class NlpRemoveStopwords(BaseEstimator):
    
    def __init__(self, text_column: str, new_column: str = None) -> None:
        self.nlp = None
        self.text_column = text_column
        if new_column is None:
            self.new_column = text_column
        else:
            self.new_column = new_column

    def fit(self, x: Any, y: Any = None) -> __qualname__:
        self.nlp = spacy.load("en_core_web_sm")

        return self
    
    def remove_stopwords(self, text: str) -> str:
        doc = self.nlp(text)
        text_no_stopwords = []
        for token in doc:
            if not token.is_stop:
                text_no_stopwords.append(token.text_with_ws)

        return "".join(text_no_stopwords)

    def transform(self, x: Any) -> pd.DataFrame:
        x[self.new_column] = x[self.text_column].apply(self.remove_stopwords)

        return x


class NlpTextToSentences(BaseEstimator):

    def __init__(self, text_column: str, new_column: str = None) -> None:
        self.nlp = None
        self.text_column = text_column
        if new_column is None:
            self.new_column = text_column
        else:
            self.new_column = new_column

    def fit(self, x: Any, y: Any = None) -> __qualname__:
        self.nlp = spacy.load("en_core_web_sm")

        return self
    
    def text_to_sentences(self, text: str) -> List[str]:
        return [sentence.text for sentence in self.nlp(text).sents]

    def transform(self, x: Any) -> pd.DataFrame:
        x[self.new_column] = x[self.text_column].apply(self.text_to_sentences)

        return x


class NlpTextToWords(BaseEstimator):

    def __init__(self, text_column: str, new_column: str = None) -> None:
        self.nlp = None
        self.text_column = text_column
        if new_column is None:
            self.new_column = text_column
        else:
            self.new_column = new_column

    def fit(self, x: Any, y: Any = None) -> __qualname__:
        self.nlp = spacy.load("en_core_web_sm")

        return self
    
    def text_to_tokens(self, text: str) -> List[str]:
        return [token.text for token in self.nlp(text)]

    def transform(self, x: Any) -> pd.DataFrame:
        x[self.new_column] = x[self.text_column].apply(self.text_to_tokens)

        return x


class NlpSpeechTagging(BaseEstimator):

    def __init__(self, text_column: str, new_column: str = None) -> None:
        self.nlp = None
        self.text_column = text_column
        if new_column is None:
            self.new_column = text_column
        else:
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
        x[self.new_column] = x[self.text_column].map(self.pos)

        return x


class NlpWordLemmatizer(BaseEstimator):
    
    def __init__(self, text_column: str, new_column: str = None) -> None:
        self.nlp = None
        self.text_column = text_column
        if new_column is None:
            self.new_column = text_column
        else:
            self.new_column = new_column

    def fit(self, x: Any, y: Any = None) -> __qualname__:
        self.nlp = spacy.load("en_core_web_sm")

        return self
    
    def lemmatize(self, text: str) -> str:
        doc = self.nlp(text)
        tokens = [token.lemma_ for token in doc]

        return " ".join(tokens)

    def transform(self, x: Any) -> pd.DataFrame:
        x[self.new_column] = x[self.text_column].map(self.lemmatize)

        return x


class NlpReplaceEmojis(BaseEstimator):

    def __init__(self, text_column: str, new_column: str = None, how: str = "replace") -> None:
        self.emot_obj = None
        self.text_column = text_column
        if new_column is None:
            self.new_column = text_column
        else:
            self.new_column = new_column
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
        x[self.new_column] = x[self.text_column].map(self.clean_emojis)

        return x


class NlpReplaceEmoticons(BaseEstimator):

    def __init__(self, text_column: str, new_column: str = None, how: str = "replace") -> None:
        self.emot_obj = None
        self.text_column = text_column
        if new_column is None:
            self.new_column = text_column
        else:
            self.new_column = new_column
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
        x[self.new_column] = x[self.text_column].map(self.clean_emoticons)

        return x


class NlpRemoveCharacterRepetition(BaseEstimator):
    pass


class NlpRemoveWordRepetition(BaseEstimator):
    pass


    # @staticmethod
    # def _replace_group(match):
    #     char, repetition = match.groups()

    #     return char
    # def replace_char_rep(self, text: str):
    #     char_rep = re.compile(r"(\S)(\1{2,})")

    #     return char_rep.sub(self._replace_group, text)

    # def replace_words_rep(self, text: str):
    #     word_rep = re.compile(r"(\b\w+\W+)(\1{2,})")

    #     return word_rep.sub(self._replace_group, text)
