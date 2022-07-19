import pytest

from src.fixtures.data import FIXTURE_DF
from src.transform.nlp_operator import *


@pytest.fixture(scope="module")
def dataset():
    return FIXTURE_DF


def test_NlpDetectLanguage(dataset):
    dataset = dataset.copy()
    pipe = NlpDetectLanguage(text_column="text", new_column="lang")
    pipe.fit(dataset)
    output = pipe.transform(dataset)
    assert output.to_dict() == {
        "id": {0: 1, 1: 2, 2: 3},
        "type": {0: "drama", 1: "comedy", 2: "thriller"},
        "useless": {0: 0, 1: 0, 2: 0},
        "text": {
            0: "first think another Disney movie, might good, it's kids movie.",
            1: "Put aside Dr. House repeat missed, Desperate Housewives (new) watch one.",
            2: "big fan Stephen King's work, film made even greater fan King. Pet Sematary Creed family.",
        },
        "polarity": {0: 1, 1: 0, 2: 1},
        "lang": {0: "ENGLISH", 1: "ENGLISH", 2: "ENGLISH"},
    }


def test_NlpWordExpansion(dataset):
    dataset = dataset.copy()
    pipe = NlpWordExpansion(text_column="text")
    pipe.fit(dataset)
    output = pipe.transform(dataset)
    assert output.to_dict() == {
        "id": {0: 1, 1: 2, 2: 3},
        "type": {0: "drama", 1: "comedy", 2: "thriller"},
        "useless": {0: 0, 1: 0, 2: 0},
        "text": {
            0: "first think another Disney movie, might good, it's kids movie.",
            1: "Put aside Dr. House repeat missed, Desperate Housewives (new) watch one.",
            2: "big fan Stephen King's work, film made even greater fan King. Pet Sematary Creed family.",
        },
        "polarity": {0: 1, 1: 0, 2: 1},
        None: {
            0: "first think another Disney movie, might good, it is kids movie.",
            1: "Put aside Dr. House repeat missed, Desperate Housewives (new) watch one.",
            2: "big fan Stephen King's work, film made even greater fan King. Pet Sematary Creed family.",
        },
    }


def test_NlpRemoveStopwords(dataset):
    dataset = dataset.copy()
    pipe = NlpRemoveStopwords(text_column="text")
    pipe.fit(dataset)
    output = pipe.transform(dataset)
    assert output.to_dict() == {
        "id": {0: 1, 1: 2, 2: 3},
        "type": {0: "drama", 1: "comedy", 2: "thriller"},
        "useless": {0: 0, 1: 0, 2: 0},
        "text": {
            0: "think Disney movie, good, kids movie.",
            1: "aside Dr. House repeat missed, Desperate Housewives (new) watch .",
            2: "big fan Stephen Kingwork, film greater fan King. Pet Sematary Creed family.",
        },
        "polarity": {0: 1, 1: 0, 2: 1},
    }


def test_NlpTextToSentences(dataset):
    dataset = dataset.copy()
    pipe = NlpTextToSentences(text_column="text", new_column="sentences")
    pipe.fit(dataset)
    output = pipe.transform(dataset)
    assert output.to_dict() == {
        "id": {0: 1, 1: 2, 2: 3},
        "type": {0: "drama", 1: "comedy", 2: "thriller"},
        "useless": {0: 0, 1: 0, 2: 0},
        "text": {
            0: "first think another Disney movie, might good, it's kids movie.",
            1: "Put aside Dr. House repeat missed, Desperate Housewives (new) watch one.",
            2: "big fan Stephen King's work, film made even greater fan King. Pet Sematary Creed family.",
        },
        "polarity": {0: 1, 1: 0, 2: 1},
        "sentences": {
            0: ["first think another Disney movie, might good, it's kids movie."],
            1: ["Put aside Dr. House repeat missed, Desperate Housewives (new) watch one."],
            2: ["big fan Stephen King's work, film made even greater fan King.", "Pet Sematary Creed family."],
        },
    }


def test_NlpTextToWords(dataset):
    dataset = dataset.copy()
    pipe = NlpTextToWords(text_column="text", new_column="words")
    pipe.fit(dataset)
    output = pipe.transform(dataset)
    assert output.to_dict() == {
        "id": {0: 1, 1: 2, 2: 3},
        "type": {0: "drama", 1: "comedy", 2: "thriller"},
        "useless": {0: 0, 1: 0, 2: 0},
        "text": {
            0: "first think another Disney movie, might good, it's kids movie.",
            1: "Put aside Dr. House repeat missed, Desperate Housewives (new) watch one.",
            2: "big fan Stephen King's work, film made even greater fan King. Pet Sematary Creed family.",
        },
        "polarity": {0: 1, 1: 0, 2: 1},
        "words": {
            0: [
                "first",
                "think",
                "another",
                "Disney",
                "movie",
                ",",
                "might",
                "good",
                ",",
                "it",
                "'s",
                "kids",
                "movie",
                ".",
            ],
            1: [
                "Put",
                "aside",
                "Dr.",
                "House",
                "repeat",
                "missed",
                ",",
                "Desperate",
                "Housewives",
                "(",
                "new",
                ")",
                "watch",
                "one",
                ".",
            ],
            2: [
                "big",
                "fan",
                "Stephen",
                "King",
                "'s",
                "work",
                ",",
                "film",
                "made",
                "even",
                "greater",
                "fan",
                "King",
                ".",
                "Pet",
                "Sematary",
                "Creed",
                "family",
                ".",
            ],
        },
    }


def test_NlpWordLemmatizer(dataset):
    dataset = dataset.copy()
    pipe = NlpWordLemmatizer(text_column="text")
    pipe.fit(dataset)
    output = pipe.transform(dataset)
    assert output.to_dict() == {
        "id": {0: 1, 1: 2, 2: 3},
        "type": {0: "drama", 1: "comedy", 2: "thriller"},
        "useless": {0: 0, 1: 0, 2: 0},
        "text": {
            0: "first think another Disney movie , might good , it be kid movie .",
            1: "put aside Dr. House repeat miss , Desperate Housewives ( new ) watch one .",
            2: "big fan Stephen King 's work , film make even great fan king . Pet Sematary Creed family .",
        },
        "polarity": {0: 1, 1: 0, 2: 1},
    }


def test_NlpReplaceEmojis(dataset):
    dataset = dataset.copy()
    pipe = NlpReplaceEmojis(text_column="text")
    pipe.fit(dataset)
    output = pipe.transform(dataset)
    assert output.to_dict() == {
        "id": {0: 1, 1: 2, 2: 3},
        "type": {0: "drama", 1: "comedy", 2: "thriller"},
        "useless": {0: 0, 1: 0, 2: 0},
        "text": {
            0: "first think another Disney movie, might good, it's kids movie.",
            1: "Put aside Dr. House repeat missed, Desperate Housewives (new) watch one.",
            2: "big fan Stephen King's work, film made even greater fan King. Pet Sematary Creed family.",
        },
        "polarity": {0: 1, 1: 0, 2: 1},
    }


def test_NlpReplaceEmoticons(dataset):
    dataset = dataset.copy()
    pipe = NlpReplaceEmoticons(text_column="text")
    pipe.fit(dataset)
    output = pipe.transform(dataset)
    assert output.to_dict() == {
        "id": {0: 1, 1: 2, 2: 3},
        "type": {0: "drama", 1: "comedy", 2: "thriller"},
        "useless": {0: 0, 1: 0, 2: 0},
        "text": {
            0: "first think another Disney movie, might good, it's kids movie.",
            1: "Put aside Dr. House repeat missed, Desperate Housewives (new) watch one.",
            2: "big fan Stephen King's work, film made even greater fan King. Pet Sematary Creed family.",
        },
        "polarity": {0: 1, 1: 0, 2: 1},
    }


def test_NlpDeDuplicatesSpace(dataset):
    dataset = dataset.copy()
    pipe = NlpDeDuplicatesSpace(text_column="text")
    pipe.fit(dataset)
    output = pipe.transform(dataset)
    assert output.to_dict() == {
        "id": {0: 1, 1: 2, 2: 3},
        "type": {0: "drama", 1: "comedy", 2: "thriller"},
        "useless": {0: 0, 1: 0, 2: 0},
        "text": {
            0: "first think another Disney movie, might good, it's kids movie.",
            1: "Put aside Dr. House repeat missed, Desperate Housewives (new) watch one.",
            2: "big fan Stephen King's work, film made even greater fan King. Pet Sematary Creed family.",
        },
        "polarity": {0: 1, 1: 0, 2: 1},
    }


def test_NlpReplaceWordRepetition(dataset):
    dataset = dataset.copy()
    pipe = NlpReplaceWordRepetition(text_column="text")
    pipe.fit(dataset)
    output = pipe.transform(dataset)
    assert output.to_dict() == {
        "id": {0: 1, 1: 2, 2: 3},
        "type": {0: "drama", 1: "comedy", 2: "thriller"},
        "useless": {0: 0, 1: 0, 2: 0},
        "text": {
            0: "first think another Disney movie, might good, it's kids movie.",
            1: "Put aside Dr. House repeat missed, Desperate Housewives (new) watch one.",
            2: "big fan Stephen King's work, film made even greater fan King. Pet Sematary Creed family.",
        },
        "polarity": {0: 1, 1: 0, 2: 1},
    }


def test_NlpRemoveCharRepetition(dataset):
    dataset = dataset.copy()
    pipe = NlpRemoveCharRepetition(text_column="text")
    pipe.fit(dataset)
    output = pipe.transform(dataset)
    assert output.to_dict() == {
        "id": {0: 1, 1: 2, 2: 3},
        "type": {0: "drama", 1: "comedy", 2: "thriller"},
        "useless": {0: 0, 1: 0, 2: 0},
        "text": {
            0: "first think another Disney movie, might good, it's kids movie.",
            1: "Put aside Dr. House repeat missed, Desperate Housewives (new) watch one.",
            2: "big fan Stephen King's work, film made even greater fan King. Pet Sematary Creed family.",
        },
        "polarity": {0: 1, 1: 0, 2: 1},
    }


def test_NlpSpeechTagging(dataset):
    dataset = dataset.copy()
    pipe = NlpSpeechTagging(text_column="text", new_column="pos")
    pipe.fit(dataset)
    output = pipe.transform(dataset)
    assert output.to_dict() == {
        "id": {0: 1, 1: 2, 2: 3},
        "type": {0: "drama", 1: "comedy", 2: "thriller"},
        "useless": {0: 0, 1: 0, 2: 0},
        "text": {
            0: "first think another Disney movie, might good, it's kids movie.",
            1: "Put aside Dr. House repeat missed, Desperate Housewives (new) watch one.",
            2: "big fan Stephen King's work, film made even greater fan King. Pet Sematary Creed family.",
        },
        "polarity": {0: 1, 1: 0, 2: 1},
        "pos": {
            0: [
                {
                    "token": "first",
                    "lemma": "first",
                    "pos": "ADV",
                    "tag": "RB",
                    "dependency": "advmod",
                    "sentiment": 0.0,
                    "shape": "xxxx",
                    "is_alpha": True,
                    "is_stopwords": True,
                },
                {
                    "token": "think",
                    "lemma": "think",
                    "pos": "VERB",
                    "tag": "VB",
                    "dependency": "ROOT",
                    "sentiment": 0.0,
                    "shape": "xxxx",
                    "is_alpha": True,
                    "is_stopwords": False,
                },
                {
                    "token": "another",
                    "lemma": "another",
                    "pos": "DET",
                    "tag": "DT",
                    "dependency": "det",
                    "sentiment": 0.0,
                    "shape": "xxxx",
                    "is_alpha": True,
                    "is_stopwords": True,
                },
                {
                    "token": "Disney",
                    "lemma": "Disney",
                    "pos": "PROPN",
                    "tag": "NNP",
                    "dependency": "compound",
                    "sentiment": 0.0,
                    "shape": "Xxxxx",
                    "is_alpha": True,
                    "is_stopwords": False,
                },
                {
                    "token": "movie",
                    "lemma": "movie",
                    "pos": "NOUN",
                    "tag": "NN",
                    "dependency": "nsubj",
                    "sentiment": 0.0,
                    "shape": "xxxx",
                    "is_alpha": True,
                    "is_stopwords": False,
                },
                {
                    "token": ",",
                    "lemma": ",",
                    "pos": "PUNCT",
                    "tag": ",",
                    "dependency": "punct",
                    "sentiment": 0.0,
                    "shape": ",",
                    "is_alpha": False,
                    "is_stopwords": False,
                },
                {
                    "token": "might",
                    "lemma": "might",
                    "pos": "AUX",
                    "tag": "MD",
                    "dependency": "aux",
                    "sentiment": 0.0,
                    "shape": "xxxx",
                    "is_alpha": True,
                    "is_stopwords": True,
                },
                {
                    "token": "good",
                    "lemma": "good",
                    "pos": "ADJ",
                    "tag": "JJ",
                    "dependency": "ccomp",
                    "sentiment": 0.0,
                    "shape": "xxxx",
                    "is_alpha": True,
                    "is_stopwords": False,
                },
                {
                    "token": ",",
                    "lemma": ",",
                    "pos": "PUNCT",
                    "tag": ",",
                    "dependency": "punct",
                    "sentiment": 0.0,
                    "shape": ",",
                    "is_alpha": False,
                    "is_stopwords": False,
                },
                {
                    "token": "it",
                    "lemma": "it",
                    "pos": "PRON",
                    "tag": "PRP",
                    "dependency": "nsubj",
                    "sentiment": 0.0,
                    "shape": "xx",
                    "is_alpha": True,
                    "is_stopwords": True,
                },
                {
                    "token": "'s",
                    "lemma": "be",
                    "pos": "AUX",
                    "tag": "VBZ",
                    "dependency": "ccomp",
                    "sentiment": 0.0,
                    "shape": "'x",
                    "is_alpha": False,
                    "is_stopwords": True,
                },
                {
                    "token": "kids",
                    "lemma": "kid",
                    "pos": "NOUN",
                    "tag": "NNS",
                    "dependency": "compound",
                    "sentiment": 0.0,
                    "shape": "xxxx",
                    "is_alpha": True,
                    "is_stopwords": False,
                },
                {
                    "token": "movie",
                    "lemma": "movie",
                    "pos": "VERB",
                    "tag": "VBP",
                    "dependency": "attr",
                    "sentiment": 0.0,
                    "shape": "xxxx",
                    "is_alpha": True,
                    "is_stopwords": False,
                },
                {
                    "token": ".",
                    "lemma": ".",
                    "pos": "PUNCT",
                    "tag": ".",
                    "dependency": "punct",
                    "sentiment": 0.0,
                    "shape": ".",
                    "is_alpha": False,
                    "is_stopwords": False,
                },
            ],
            1: [
                {
                    "token": "Put",
                    "lemma": "put",
                    "pos": "VERB",
                    "tag": "VB",
                    "dependency": "advcl",
                    "sentiment": 0.0,
                    "shape": "Xxx",
                    "is_alpha": True,
                    "is_stopwords": True,
                },
                {
                    "token": "aside",
                    "lemma": "aside",
                    "pos": "ADV",
                    "tag": "RB",
                    "dependency": "advmod",
                    "sentiment": 0.0,
                    "shape": "xxxx",
                    "is_alpha": True,
                    "is_stopwords": False,
                },
                {
                    "token": "Dr.",
                    "lemma": "Dr.",
                    "pos": "PROPN",
                    "tag": "NNP",
                    "dependency": "compound",
                    "sentiment": 0.0,
                    "shape": "Xx.",
                    "is_alpha": False,
                    "is_stopwords": False,
                },
                {
                    "token": "House",
                    "lemma": "House",
                    "pos": "PROPN",
                    "tag": "NNP",
                    "dependency": "compound",
                    "sentiment": 0.0,
                    "shape": "Xxxxx",
                    "is_alpha": True,
                    "is_stopwords": False,
                },
                {
                    "token": "repeat",
                    "lemma": "repeat",
                    "pos": "NOUN",
                    "tag": "NN",
                    "dependency": "nsubj",
                    "sentiment": 0.0,
                    "shape": "xxxx",
                    "is_alpha": True,
                    "is_stopwords": False,
                },
                {
                    "token": "missed",
                    "lemma": "miss",
                    "pos": "VERB",
                    "tag": "VBN",
                    "dependency": "ccomp",
                    "sentiment": 0.0,
                    "shape": "xxxx",
                    "is_alpha": True,
                    "is_stopwords": False,
                },
                {
                    "token": ",",
                    "lemma": ",",
                    "pos": "PUNCT",
                    "tag": ",",
                    "dependency": "punct",
                    "sentiment": 0.0,
                    "shape": ",",
                    "is_alpha": False,
                    "is_stopwords": False,
                },
                {
                    "token": "Desperate",
                    "lemma": "Desperate",
                    "pos": "PROPN",
                    "tag": "NNP",
                    "dependency": "compound",
                    "sentiment": 0.0,
                    "shape": "Xxxxx",
                    "is_alpha": True,
                    "is_stopwords": False,
                },
                {
                    "token": "Housewives",
                    "lemma": "Housewives",
                    "pos": "PROPN",
                    "tag": "NNP",
                    "dependency": "nsubj",
                    "sentiment": 0.0,
                    "shape": "Xxxxx",
                    "is_alpha": True,
                    "is_stopwords": False,
                },
                {
                    "token": "(",
                    "lemma": "(",
                    "pos": "PUNCT",
                    "tag": "-LRB-",
                    "dependency": "punct",
                    "sentiment": 0.0,
                    "shape": "(",
                    "is_alpha": False,
                    "is_stopwords": False,
                },
                {
                    "token": "new",
                    "lemma": "new",
                    "pos": "ADJ",
                    "tag": "JJ",
                    "dependency": "amod",
                    "sentiment": 0.0,
                    "shape": "xxx",
                    "is_alpha": True,
                    "is_stopwords": False,
                },
                {
                    "token": ")",
                    "lemma": ")",
                    "pos": "PUNCT",
                    "tag": "-RRB-",
                    "dependency": "punct",
                    "sentiment": 0.0,
                    "shape": ")",
                    "is_alpha": False,
                    "is_stopwords": False,
                },
                {
                    "token": "watch",
                    "lemma": "watch",
                    "pos": "VERB",
                    "tag": "VB",
                    "dependency": "ROOT",
                    "sentiment": 0.0,
                    "shape": "xxxx",
                    "is_alpha": True,
                    "is_stopwords": False,
                },
                {
                    "token": "one",
                    "lemma": "one",
                    "pos": "NUM",
                    "tag": "CD",
                    "dependency": "dobj",
                    "sentiment": 0.0,
                    "shape": "xxx",
                    "is_alpha": True,
                    "is_stopwords": True,
                },
                {
                    "token": ".",
                    "lemma": ".",
                    "pos": "PUNCT",
                    "tag": ".",
                    "dependency": "punct",
                    "sentiment": 0.0,
                    "shape": ".",
                    "is_alpha": False,
                    "is_stopwords": False,
                },
            ],
            2: [
                {
                    "token": "big",
                    "lemma": "big",
                    "pos": "ADJ",
                    "tag": "JJ",
                    "dependency": "amod",
                    "sentiment": 0.0,
                    "shape": "xxx",
                    "is_alpha": True,
                    "is_stopwords": False,
                },
                {
                    "token": "fan",
                    "lemma": "fan",
                    "pos": "NOUN",
                    "tag": "NN",
                    "dependency": "compound",
                    "sentiment": 0.0,
                    "shape": "xxx",
                    "is_alpha": True,
                    "is_stopwords": False,
                },
                {
                    "token": "Stephen",
                    "lemma": "Stephen",
                    "pos": "PROPN",
                    "tag": "NNP",
                    "dependency": "compound",
                    "sentiment": 0.0,
                    "shape": "Xxxxx",
                    "is_alpha": True,
                    "is_stopwords": False,
                },
                {
                    "token": "King",
                    "lemma": "King",
                    "pos": "PROPN",
                    "tag": "NNP",
                    "dependency": "poss",
                    "sentiment": 0.0,
                    "shape": "Xxxx",
                    "is_alpha": True,
                    "is_stopwords": False,
                },
                {
                    "token": "'s",
                    "lemma": "'s",
                    "pos": "PART",
                    "tag": "POS",
                    "dependency": "case",
                    "sentiment": 0.0,
                    "shape": "'x",
                    "is_alpha": False,
                    "is_stopwords": True,
                },
                {
                    "token": "work",
                    "lemma": "work",
                    "pos": "NOUN",
                    "tag": "NN",
                    "dependency": "nsubj",
                    "sentiment": 0.0,
                    "shape": "xxxx",
                    "is_alpha": True,
                    "is_stopwords": False,
                },
                {
                    "token": ",",
                    "lemma": ",",
                    "pos": "PUNCT",
                    "tag": ",",
                    "dependency": "punct",
                    "sentiment": 0.0,
                    "shape": ",",
                    "is_alpha": False,
                    "is_stopwords": False,
                },
                {
                    "token": "film",
                    "lemma": "film",
                    "pos": "NOUN",
                    "tag": "NN",
                    "dependency": "nsubj",
                    "sentiment": 0.0,
                    "shape": "xxxx",
                    "is_alpha": True,
                    "is_stopwords": False,
                },
                {
                    "token": "made",
                    "lemma": "make",
                    "pos": "VERB",
                    "tag": "VBD",
                    "dependency": "ROOT",
                    "sentiment": 0.0,
                    "shape": "xxxx",
                    "is_alpha": True,
                    "is_stopwords": True,
                },
                {
                    "token": "even",
                    "lemma": "even",
                    "pos": "ADV",
                    "tag": "RB",
                    "dependency": "advmod",
                    "sentiment": 0.0,
                    "shape": "xxxx",
                    "is_alpha": True,
                    "is_stopwords": True,
                },
                {
                    "token": "greater",
                    "lemma": "great",
                    "pos": "ADJ",
                    "tag": "JJR",
                    "dependency": "amod",
                    "sentiment": 0.0,
                    "shape": "xxxx",
                    "is_alpha": True,
                    "is_stopwords": False,
                },
                {
                    "token": "fan",
                    "lemma": "fan",
                    "pos": "NOUN",
                    "tag": "NN",
                    "dependency": "compound",
                    "sentiment": 0.0,
                    "shape": "xxx",
                    "is_alpha": True,
                    "is_stopwords": False,
                },
                {
                    "token": "King",
                    "lemma": "king",
                    "pos": "NOUN",
                    "tag": "NN",
                    "dependency": "dobj",
                    "sentiment": 0.0,
                    "shape": "Xxxx",
                    "is_alpha": True,
                    "is_stopwords": False,
                },
                {
                    "token": ".",
                    "lemma": ".",
                    "pos": "PUNCT",
                    "tag": ".",
                    "dependency": "punct",
                    "sentiment": 0.0,
                    "shape": ".",
                    "is_alpha": False,
                    "is_stopwords": False,
                },
                {
                    "token": "Pet",
                    "lemma": "Pet",
                    "pos": "PROPN",
                    "tag": "NNP",
                    "dependency": "compound",
                    "sentiment": 0.0,
                    "shape": "Xxx",
                    "is_alpha": True,
                    "is_stopwords": False,
                },
                {
                    "token": "Sematary",
                    "lemma": "Sematary",
                    "pos": "PROPN",
                    "tag": "NNP",
                    "dependency": "compound",
                    "sentiment": 0.0,
                    "shape": "Xxxxx",
                    "is_alpha": True,
                    "is_stopwords": False,
                },
                {
                    "token": "Creed",
                    "lemma": "Creed",
                    "pos": "PROPN",
                    "tag": "NNP",
                    "dependency": "compound",
                    "sentiment": 0.0,
                    "shape": "Xxxxx",
                    "is_alpha": True,
                    "is_stopwords": False,
                },
                {
                    "token": "family",
                    "lemma": "family",
                    "pos": "NOUN",
                    "tag": "NN",
                    "dependency": "ROOT",
                    "sentiment": 0.0,
                    "shape": "xxxx",
                    "is_alpha": True,
                    "is_stopwords": False,
                },
                {
                    "token": ".",
                    "lemma": ".",
                    "pos": "PUNCT",
                    "tag": ".",
                    "dependency": "punct",
                    "sentiment": 0.0,
                    "shape": ".",
                    "is_alpha": False,
                    "is_stopwords": False,
                },
            ],
        },
    }
