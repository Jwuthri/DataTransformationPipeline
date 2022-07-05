import re
from typing import Any, Dict, List

import emot
import spacy
import contractions

from pycld2 import detect
from sklearn.base import BaseEstimator

import pandas as pd


class NlpDetectLanguage(BaseEstimator):
    """It's a wrapper for the detect_language function from the langdetect library."""

    def __init__(self, text_column: str, new_column: str = None) -> None:
        """
        The function takes in a text column and a new column name, and if the new column name is not
        specified, it will use the text column name as the new column name
        
        :param text_column: The column containing the text you want to clean
        :type text_column: str
        :param new_column: The name of the new column that will be created. If not specified, the new
        column will have the same name as the text column
        :type new_column: str
        """
        self.text_column = text_column
        if new_column is None:
            self.new_column = text_column
        else:
            self.new_column = new_column

    def fit(self, x: Any, y: Any = None) -> __qualname__:
        return self

    @staticmethod
    def detect_language(message: str) -> str:
        """
        It takes a string as input, removes all non-printable characters, and then uses the detect()
        function from the langdetect library to detect the language of the string
        
        :param message: str
        :type message: str
        :return: The language of the message.
        """
        printable_message = "".join(x for x in message if x.isprintable())
        reliable, _, detected_language = detect(printable_message)
        if reliable:
            return detected_language[0][0]
        else:
            return 'UNKNOWN'

    def transform(self, x: Any) -> pd.DataFrame:
        """
        The function takes a dataframe, and a column name, and returns a new dataframe with a new column
        that contains the language of the text in the original column
        
        :param x: Any - the dataframe that will be passed to the transform method
        :type x: Any
        :return: A dataframe with a new column called 'language'
        """
        x[self.new_column] = x[self.text_column].map(self.detect_language)

        return x


class NlpWordExpansion(BaseEstimator):

    def __init__(self, text_column: str, new_column: str = None) -> None:
        """
        This function takes in a text column and a new column name and returns a None
        
        :param text_column: The column containing the text you want to process
        :type text_column: str
        :param new_column: The name of the new column that will contain the cleaned text. If not
        specified, the name of the new column will be the same as the original text column
        :type new_column: str
        """
        self.text_column = text_column
        self.new_column = new_column

    def fit(self, x: Any, y: Any = None) -> __qualname__:
        return self

    def transform(self, x: Any) -> pd.DataFrame:
        """
        Remove the word's contractions from a dataframe column
        
        :param x: Any - the dataframe that you want to transform
        :type x: Any
        :return: A dataframe with the new column added.
        """
        x[self.new_column] = x[self.text_column].apply(contractions.fix)

        return x
  

class NlpRemoveStopwords(BaseEstimator):
    """It's a class that takes a list of stopwords and removes them from a list of words."""

    def __init__(self, text_column: str, new_column: str = None) -> None:
        """
        This function takes in a text column and a new column name and returns a new column with the new
        column name
        
        :param text_column: The column in the dataframe that contains the text to be processed
        :type text_column: str
        :param new_column: The name of the new column that will be created in the dataframe
        :type new_column: str
        """
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
        """
        It takes a string of text, creates a spaCy document, iterates through the tokens in the
        document, and if the token is not a stopword, it adds the token to a list. 
        
        The function then returns the list as a string
        
        :param text: The text to be processed
        :type text: str
        :return: A string
        """
        doc = self.nlp(text)
        text_no_stopwords = []
        for token in doc:
            if not token.is_stop:
                text_no_stopwords.append(token.text_with_ws)

        return "".join(text_no_stopwords)

    def transform(self, x: Any) -> pd.DataFrame:
        """
        The function takes in a dataframe, and a column name, and returns a dataframe with a new column
        that has the stopwords removed from the original column
        
        :param x: Any - the dataframe that will be passed to the transform method
        :type x: Any
        :return: A dataframe with the new column added.
        """
        x[self.new_column] = x[self.text_column].apply(self.remove_stopwords)

        return x


class NlpTextToSentences(BaseEstimator):
    """It takes a string of text and returns a list of sentences."""

    def __init__(self, text_column: str, new_column: str = None) -> None:
        """
        This function takes in a text column and a new column name and returns a new column with the new
        column name
        
        :param text_column: The column in the dataframe that contains the text to be processed
        :type text_column: str
        :param new_column: The name of the new column that will be created in the dataframe
        :type new_column: str
        """
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
        """
        It takes a string of text and returns a list of strings, where each string is a sentence from
        the original text
        
        :param text: The text to be split into sentences
        :type text: str
        :return: A list of strings.
        """
        return [sentence.text for sentence in self.nlp(text).sents]

    def transform(self, x: Any) -> pd.DataFrame:
        x[self.new_column] = x[self.text_column].apply(self.text_to_sentences)

        return x


class NlpTextToWords(BaseEstimator):
    """It takes a string of text, and returns a list of words."""

    def __init__(self, text_column: str, new_column: str = None) -> None:
        """
        This function takes in a text column and a new column name and returns a new column with the new
        column name
        
        :param text_column: The column in the dataframe that contains the text to be processed
        :type text_column: str
        :param new_column: The name of the new column that will be created in the dataframe
        :type new_column: str
        """
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
        """
        It takes a string of text and returns a list of tokens
        
        :param text: The text to be tokenized
        :type text: str
        :return: A list of tokens
        """
        return [token.text for token in self.nlp(text)]

    def transform(self, x: Any) -> pd.DataFrame:
        """
        The function takes a dataframe, and a column of text, and returns a dataframe with a new column
        of tokens
        
        :param x: Any - the dataframe that will be passed to the transform function
        :type x: Any
        :return: A dataframe with the new column added.
        """
        x[self.new_column] = x[self.text_column].apply(self.text_to_tokens)

        return x


class NlpSpeechTagging(BaseEstimator):
    """It's a wrapper for a scikit-learn estimator that takes a list of strings as input and returns a list of strings as output."""

    def __init__(self, text_column: str, new_column: str = None) -> None:
        """
        This function takes in a text column and a new column name and returns a new column with the new
        column name
        
        :param text_column: The column in the dataframe that contains the text to be processed
        :type text_column: str
        :param new_column: The name of the new column that will be created in the dataframe
        :type new_column: str
        """
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
        """
        The function takes a string as input, and returns a dictionary of the tokens, lemmas, parts of
        speech, tags, dependencies, sentiment, shape, is_alpha, and is_stopwords
        
        :param text: The text to be processed
        :type text: str
        :return: A dictionary of records
        """
        doc = self.nlp(text)
        pos_tagging = []
        for token in doc:
            pos_tagging.append(
                [token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.sentiment, token.shape_, token.is_alpha, token.is_stop]
            )

        return pd.DataFrame(pos_tagging, columns=['token', 'lemma', 'pos', 'tag', 'dependency', 'sentiment', 'shape', 'is_alpha', 'is_stopwords']).to_dict('records')

    def transform(self, x: Any) -> pd.DataFrame:
        """
        The function takes a dataframe, and a column of text, and returns a dataframe with a new column
        of text that has been transformed by the function
        
        :param x: Any - the dataframe that you want to transform
        :type x: Any
        :return: A dataframe with the new column added.
        """
        x[self.new_column] = x[self.text_column].map(self.pos)

        return x


class NlpWordLemmatizer(BaseEstimator):
    """It's a wrapper for the NLTK WordNetLemmatizer class that implements the scikit-learn transformer API."""

    def __init__(self, text_column: str, new_column: str = None) -> None:
        """
        This function takes in a text column and a new column name and returns a new column with the new
        column name
        
        :param text_column: The column in the dataframe that contains the text you want to process
        :type text_column: str
        :param new_column: The name of the new column that will be created in the dataframe
        :type new_column: str
        """
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
        """
        The function takes a string as input, and returns a string as output. 
        
        The input string is passed to the nlp object, which is a spaCy object. 
        
        The nlp object returns a doc object, which is a spaCy object. 
        
        The doc object is iterated over, and each token is lemmatized. 
        
        The lemmatized tokens are returned as a string.
        
        :param text: The text to be lemmatized
        :type text: str
        :return: A string of lemmatized tokens
        """
        doc = self.nlp(text)
        tokens = [token.lemma_ for token in doc]

        return " ".join(tokens)

    def transform(self, x: Any) -> pd.DataFrame:
        """
        The function takes in a dataframe, and returns a dataframe with a new column that is the
        lemmatized version of the text column
        
        :param x: Any - the dataframe that you want to transform
        :type x: Any
        :return: A dataframe with a new column called 'lemmatized_text'
        """
        x[self.new_column] = x[self.text_column].map(self.lemmatize)

        return x


class NlpReplaceEmojis(BaseEstimator):
    """Replaces emojis with their textual description"""

    def __init__(self, text_column: str, new_column: str = None, how: str = "replace") -> None:
        """
        The function takes in a text column, a new column, and a how parameter. If the new column is not
        specified, the new column is set to the text column. The how parameter is set to replace by
        default.
        
        :param text_column: The column in the dataframe that contains the text you want to clean
        :type text_column: str
        :param new_column: The name of the new column that will be created. If None, the name of the
        text_column will be used
        :type new_column: str
        :param how: replace, append, prepend, defaults to replace
        :type how: str (optional)
        """
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
        """
        It takes a string and returns a string with emojis replaced with their meanings or removed
        
        :param text: The text to be cleaned
        :type text: str
        :return: A dictionary with the following keys:
        """
        emojis = self.emot_obj.emoji(text)
        if emojis['flag']:
            for index in range(len(emojis) - 1, -1, -1):
                target_value = emojis['mean'][index] if self.how == "replace" else ""
                start, end = emojis['location'][index]
                text = text[:start] + target_value + text[end:]

        return text

    def transform(self, x: Any) -> pd.DataFrame:
        """
        It takes a dataframe, and for each row in the dataframe, it takes the text in the column
        specified by the text_column parameter, and replaces all emojis with the text specified by the
        replace_with parameter
        
        :param x: Any - the dataframe you want to transform
        :type x: Any
        :return: A dataframe with the new column added.
        """
        x[self.new_column] = x[self.text_column].map(self.clean_emojis)

        return x


class NlpReplaceEmoticons(BaseEstimator):
    """Replaces emoticons with their corresponding words."""

    def __init__(self, text_column: str, new_column: str = None, how: str = "replace") -> None:
        """
        The function takes in a text column, a new column, and a how parameter. If the new column is not
        specified, the new column is set to the text column. The how parameter is set to replace by
        default.
        
        :param text_column: The column in the dataframe that contains the text you want to clean
        :type text_column: str
        :param new_column: The name of the new column that will be created. If None, the name of the
        text_column will be used
        :type new_column: str
        :param how: replace, append, prepend, defaults to replace
        :type how: str (optional)
        """
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
        """
        It takes a string as input, finds all the emoticons in the string, and replaces them with their
        meaning (if the how parameter is set to "replace") or with an empty string

        :param text: The text to be cleaned
        :type text: str
        :return: A dictionary with the following keys:
        """
        emoticons = self.emot_obj.emoticons(text)
        if emoticons['flag']:
            for index in range(len(emoticons) - 1, -1, -1):
                target_value = emoticons['mean'][index] if self.how == "replace" else ""
                start, end = emoticons['location'][index]
                text = text[:start] + target_value + text[end:]

        return text

    def transform(self, x: Any) -> pd.DataFrame:
        """
        It takes a dataframe, and for each row in the dataframe, it takes the text in the column
        specified by the text_column parameter, and replaces the emoticons in that text with the
        corresponding word in the dictionary specified by the emoticons parameter
        
        :param x: Any - the dataframe you want to transform
        :type x: Any
        :return: A dataframe with the new column added.
        """
        x[self.new_column] = x[self.text_column].map(self.clean_emoticons)

        return x


class NlpDeDuplicatesSpace(BaseEstimator):

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


class NlpReplaceWordRepetition(BaseEstimator):

    def __init__(self, text_column: str, new_column: str = None) -> None:
        self.text_column = text_column
        if new_column is None:
            self.new_column = text_column
        else:
            self.new_column = new_column

    def fit(self, x: Any, y: Any = None) -> __qualname__:
        return self

    @staticmethod
    def _replace_group(match):
        char, _ = match.groups()

        return char

    def replace_words_rep(self, text: str) -> str:
        word_rep = re.compile(r"(\b\w+\W+)(\1{2,})")
        
        return word_rep.sub(self._replace_group, text)

    def transform(self, x: Any) -> pd.DataFrame:
        x[self.new_column] = x[self.text_column].map(self.replace_words_rep)

        return x



class NlpRemoveCharRepetition(BaseEstimator):

    def __init__(self, text_column: str, new_column: str = None) -> None:
        self.text_column = text_column
        if new_column is None:
            self.new_column = text_column
        else:
            self.new_column = new_column

    def fit(self, x: Any, y: Any = None) -> __qualname__:
        return self

    @staticmethod
    def _replace_group(match):
        char, _ = match.groups()

        return char
        
    def replace_char_rep(self, text: str) -> str:
        char_rep = re.compile(r"(\S)(\1{2,})")
        
        return char_rep.sub(self._replace_group, text)

    def transform(self, x: Any) -> pd.DataFrame:
        x[self.new_column] = x[self.text_column].map(self.replace_char_rep)

        return x
