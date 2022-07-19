import pytest

from src.transform.nlp_operator import *


@pytest.fixture(scope='module')
def dataset():
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "type": ['drama', 'comedy', 'thriller'],
            "useless": [0, 0, 0],
            "text": [
                "I aaaaam so so so happy. first think another Disney movie, might good, it's kids movie.",
                "Put aside Dr. House repeat missed, Desperate Housewives (new) watch one.",
                "big fan Stephen King's work, film made even greater fan King. Pet Sematary Creed family.",
            ],
            "polarity": [1, 0, 1]
        }
    )


def test_NlpDetectLanguage(dataset):
    pipe = NlpDetectLanguage(text_column="text", new_column="lang")
    pipe.fit(dataset)
    output = pipe.transform(dataset)
    assert output.to_dict() == {'id': {0: 1, 1: 2, 2: 3}, 'type': {0: 'drama', 1: 'comedy', 2: 'thriller'}, 'useless': {0: 0, 1: 0, 2: 0}, 'text': {0: "I aaaaam so so so happy. first think another Disney movie, might good, it's kids movie.", 1: 'Put aside Dr. House repeat missed, Desperate Housewives (new) watch one.', 2: "big fan Stephen King's work, film made even greater fan King. Pet Sematary Creed family."}, 'polarity': {0: 1, 1: 0, 2: 1}, 'lang': {0: 'ENGLISH', 1: 'ENGLISH', 2: 'ENGLISH'}}


def test_NlpWordExpansion(dataset):
    pipe = NlpWordExpansion(text_column="text")
    pipe.fit(dataset)
    output = pipe.transform(dataset)
    assert output.to_dict() == {'id': {0: 1, 1: 2, 2: 3}, 'type': {0: 'drama', 1: 'comedy', 2: 'thriller'}, 'useless': {0: 0, 1: 0, 2: 0}, 'text': {0: "I aaaaam so so so happy. first think another Disney movie, might good, it's kids movie.", 1: 'Put aside Dr. House repeat missed, Desperate Housewives (new) watch one.', 2: "big fan Stephen King's work, film made even greater fan King. Pet Sematary Creed family."}, 'polarity': {0: 1, 1: 0, 2: 1}, 'lang': {0: 'ENGLISH', 1: 'ENGLISH', 2: 'ENGLISH'}, None: {0: 'I aaaaam so so so happy. first think another Disney movie, might good, it is kids movie.', 1: 'Put aside Dr. House repeat missed, Desperate Housewives (new) watch one.', 2: "big fan Stephen King's work, film made even greater fan King. Pet Sematary Creed family."}}


def test_NlpRemoveStopwords(dataset):
    pipe = NlpRemoveStopwords(text_column="text")
    pipe.fit(dataset)
    output = pipe.transform(dataset)
    assert output.to_dict() == {'id': {0: 1, 1: 2, 2: 3}, 'type': {0: 'drama', 1: 'comedy', 2: 'thriller'}, 'useless': {0: 0, 1: 0, 2: 0}, 'text': {0: 'aaaaam happy. think Disney movie, good, kids movie.', 1: 'aside Dr. House repeat missed, Desperate Housewives (new) watch .', 2: 'big fan Stephen Kingwork, film greater fan King. Pet Sematary Creed family.'}, 'polarity': {0: 1, 1: 0, 2: 1}, 'lang': {0: 'ENGLISH', 1: 'ENGLISH', 2: 'ENGLISH'}, None: {0: 'I aaaaam so so so happy. first think another Disney movie, might good, it is kids movie.', 1: 'Put aside Dr. House repeat missed, Desperate Housewives (new) watch one.', 2: "big fan Stephen King's work, film made even greater fan King. Pet Sematary Creed family."}}


def test_NlpTextToSentences(dataset):
    pipe = NlpTextToSentences(text_column="text", new_column="sentences")
    pipe.fit(dataset)
    output = pipe.transform(dataset)
    assert output.to_dict() == {'id': {0: 1, 1: 2, 2: 3}, 'type': {0: 'drama', 1: 'comedy', 2: 'thriller'}, 'useless': {0: 0, 1: 0, 2: 0}, 'text': {0: 'aaaaam happy. think Disney movie, good, kids movie.', 1: 'aside Dr. House repeat missed, Desperate Housewives (new) watch .', 2: 'big fan Stephen Kingwork, film greater fan King. Pet Sematary Creed family.'}, 'polarity': {0: 1, 1: 0, 2: 1}, 'lang': {0: 'ENGLISH', 1: 'ENGLISH', 2: 'ENGLISH'}, None: {0: 'I aaaaam so so so happy. first think another Disney movie, might good, it is kids movie.', 1: 'Put aside Dr. House repeat missed, Desperate Housewives (new) watch one.', 2: "big fan Stephen King's work, film made even greater fan King. Pet Sematary Creed family."}, 'sentences': {0: ['aaaaam happy.', 'think Disney movie, good, kids movie.'], 1: ['aside Dr. House repeat missed, Desperate Housewives (new) watch .'], 2: ['big fan Stephen Kingwork, film greater fan King.', 'Pet Sematary Creed family.']}}


def test_NlpTextToWords(dataset):
    pipe = NlpTextToWords(text_column="text", new_column="words")
    pipe.fit(dataset)
    output = pipe.transform(dataset)
    assert output.to_dict() == {'id': {0: 1, 1: 2, 2: 3}, 'type': {0: 'drama', 1: 'comedy', 2: 'thriller'}, 'useless': {0: 0, 1: 0, 2: 0}, 'text': {0: 'aaaaam happy. think Disney movie, good, kids movie.', 1: 'aside Dr. House repeat missed, Desperate Housewives (new) watch .', 2: 'big fan Stephen Kingwork, film greater fan King. Pet Sematary Creed family.'}, 'polarity': {0: 1, 1: 0, 2: 1}, 'lang': {0: 'ENGLISH', 1: 'ENGLISH', 2: 'ENGLISH'}, None: {0: 'I aaaaam so so so happy. first think another Disney movie, might good, it is kids movie.', 1: 'Put aside Dr. House repeat missed, Desperate Housewives (new) watch one.', 2: "big fan Stephen King's work, film made even greater fan King. Pet Sematary Creed family."}, 'sentences': {0: ['aaaaam happy.', 'think Disney movie, good, kids movie.'], 1: ['aside Dr. House repeat missed, Desperate Housewives (new) watch .'], 2: ['big fan Stephen Kingwork, film greater fan King.', 'Pet Sematary Creed family.']}, 'words': {0: ['aaaaam', 'happy', '.', 'think', 'Disney', 'movie', ',', 'good', ',', 'kids', 'movie', '.'], 1: ['aside', 'Dr.', 'House', 'repeat', 'missed', ',', 'Desperate', 'Housewives', '(', 'new', ')', 'watch', '.'], 2: ['big', 'fan', 'Stephen', 'Kingwork', ',', 'film', 'greater', 'fan', 'King', '.', 'Pet', 'Sematary', 'Creed', 'family', '.']}}


# def test_NlpWordLemmatizer(dataset):
#     pipe = NlpWordLemmatizer(text_column="text")
#     pipe.fit(dataset)
#     output = pipe.transform(dataset)
#     breakpoint()
#     assert output.to_dict() == {'id': {0: 1, 1: 2, 2: 3}, 'type': {0: 'drama', 1: 'comedy', 2: 'thriller'}, 'useless': {0: 0, 1: 0, 2: 0}, 'text': {0: "first think another Disney movie, might good, it's kids movie.", 1: 'Put aside Dr. House repeat missed, Desperate Housewives (new) watch one.', 2: "big fan Stephen King's work, film made even greater fan King. Pet Sematary Creed family."}, 'polarity': {0: 1, 1: 0, 2: 1}, 'lang': {0: 'ENGLISH', 1: 'ENGLISH', 2: 'ENGLISH'}}


def test_NlpReplaceEmojis(dataset):
    pipe = NlpReplaceEmojis(text_column="text")
    pipe.fit(dataset)
    output = pipe.transform(dataset)
    # breakpoint()
    assert output.to_dict() == {'id': {0: 1, 1: 2, 2: 3}, 'type': {0: 'drama', 1: 'comedy', 2: 'thriller'}, 'useless': {0: 0, 1: 0, 2: 0}, 'text': {0: "first think another Disney movie, might good, it's kids movie.", 1: 'Put aside Dr. House repeat missed, Desperate Housewives (new) watch one.', 2: "big fan Stephen King's work, film made even greater fan King. Pet Sematary Creed family."}, 'polarity': {0: 1, 1: 0, 2: 1}, 'lang': {0: 'ENGLISH', 1: 'ENGLISH', 2: 'ENGLISH'}}


def test_NlpReplaceEmoticons(dataset):
    pipe = NlpReplaceEmoticons(text_column="text")
    pipe.fit(dataset)
    output = pipe.transform(dataset)
    # breakpoint()
    assert output.to_dict() == {'id': {0: 1, 1: 2, 2: 3}, 'type': {0: 'drama', 1: 'comedy', 2: 'thriller'}, 'useless': {0: 0, 1: 0, 2: 0}, 'text': {0: "first think another Disney movie, might good, it's kids movie.", 1: 'Put aside Dr. House repeat missed, Desperate Housewives (new) watch one.', 2: "big fan Stephen King's work, film made even greater fan King. Pet Sematary Creed family."}, 'polarity': {0: 1, 1: 0, 2: 1}, 'lang': {0: 'ENGLISH', 1: 'ENGLISH', 2: 'ENGLISH'}}


def test_NlpDeDuplicatesSpace(dataset):
    pipe = NlpDeDuplicatesSpace(text_column="text")
    pipe.fit(dataset)
    output = pipe.transform(dataset)
    # breakpoint()
    assert output.to_dict() == {'id': {0: 1, 1: 2, 2: 3}, 'type': {0: 'drama', 1: 'comedy', 2: 'thriller'}, 'useless': {0: 0, 1: 0, 2: 0}, 'text': {0: "first think another Disney movie, might good, it's kids movie.", 1: 'Put aside Dr. House repeat missed, Desperate Housewives (new) watch one.', 2: "big fan Stephen King's work, film made even greater fan King. Pet Sematary Creed family."}, 'polarity': {0: 1, 1: 0, 2: 1}, 'lang': {0: 'ENGLISH', 1: 'ENGLISH', 2: 'ENGLISH'}}


def test_NlpReplaceWordRepetition(dataset):
    pipe = NlpReplaceWordRepetition(text_column="text")
    pipe.fit(dataset)
    output = pipe.transform(dataset)
    # breakpoint()
    assert output.to_dict() == {'id': {0: 1, 1: 2, 2: 3}, 'type': {0: 'drama', 1: 'comedy', 2: 'thriller'}, 'useless': {0: 0, 1: 0, 2: 0}, 'text': {0: "first think another Disney movie, might good, it's kids movie.", 1: 'Put aside Dr. House repeat missed, Desperate Housewives (new) watch one.', 2: "big fan Stephen King's work, film made even greater fan King. Pet Sematary Creed family."}, 'polarity': {0: 1, 1: 0, 2: 1}, 'lang': {0: 'ENGLISH', 1: 'ENGLISH', 2: 'ENGLISH'}}


def test_NlpRemoveCharRepetition(dataset):
    pipe = NlpRemoveCharRepetition(text_column="text")
    pipe.fit(dataset)
    output = pipe.transform(dataset)
    # breakpoint()
    assert output.to_dict() == {'id': {0: 1, 1: 2, 2: 3}, 'type': {0: 'drama', 1: 'comedy', 2: 'thriller'}, 'useless': {0: 0, 1: 0, 2: 0}, 'text': {0: "first think another Disney movie, might good, it's kids movie.", 1: 'Put aside Dr. House repeat missed, Desperate Housewives (new) watch one.', 2: "big fan Stephen King's work, film made even greater fan King. Pet Sematary Creed family."}, 'polarity': {0: 1, 1: 0, 2: 1}, 'lang': {0: 'ENGLISH', 1: 'ENGLISH', 2: 'ENGLISH'}}


# def test_NlpSpeechTagging(dataset):
#     pipe = NlpSpeechTagging(text_column="text", new_column="pos")
#     pipe.fit(dataset)
#     output = pipe.transform(dataset)
#     assert output.to_dict() == {'id': {0: 1, 1: 2, 2: 3}, 'type': {0: 'drama', 1: 'comedy', 2: 'thriller'}, 'useless': {0: 0, 1: 0, 2: 0}, 'text': {0: 'aaaaam happy. think Disney movie, good, kids movie.', 1: 'aside Dr. House repeat missed, Desperate Housewives (new) watch .', 2: 'big fan Stephen Kingwork, film greater fan King. Pet Sematary Creed family.'}, 'polarity': {0: 1, 1: 0, 2: 1}, 'lang': {0: 'ENGLISH', 1: 'ENGLISH', 2: 'ENGLISH'}, None: {0: 'I aaaaam so so so happy. first think another Disney movie, might good, it is kids movie.', 1: 'Put aside Dr. House repeat missed, Desperate Housewives (new) watch one.', 2: "big fan Stephen King's work, film made even greater fan King. Pet Sematary Creed family."}, 'sentences': {0: ['aaaaam happy.', 'think Disney movie, good, kids movie.'], 1: ['aside Dr. House repeat missed, Desperate Housewives (new) watch .'], 2: ['big fan Stephen Kingwork, film greater fan King.', 'Pet Sematary Creed family.']}, 'words': {0: ['aaaaam', 'happy', '.', 'think', 'Disney', 'movie', ',', 'good', ',', 'kids', 'movie', '.'], 1: ['aside', 'Dr.', 'House', 'repeat', 'missed', ',', 'Desperate', 'Housewives', '(', 'new', ')', 'watch', '.'], 2: ['big', 'fan', 'Stephen', 'Kingwork', ',', 'film', 'greater', 'fan', 'King', '.', 'Pet', 'Sematary', 'Creed', 'family', '.']}, 'pos': {0: [{'token': 'aaaaam', 'lemma': 'aaaaam', 'pos': 'INTJ', 'tag': 'UH', 'dependency': 'ROOT', 'sentiment': 0.0, 'shape': 'xxxx', 'is_alpha': True, 'is_stopwords': False}, {'token': 'happy', 'lemma': 'happy', 'pos': 'ADJ', 'tag': 'JJ', 'dependency': 'amod', 'sentiment': 0.0, 'shape': 'xxxx', 'is_alpha': True, 'is_stopwords': False}, {'token': '.', 'lemma': '.', 'pos': 'PUNCT', 'tag': '.', 'dependency': 'punct', 'sentiment': 0.0, 'shape': '.', 'is_alpha': False, 'is_stopwords': False}, {'token': 'think', 'lemma': 'think', 'pos': 'VERB', 'tag': 'VB', 'dependency': 'ROOT', 'sentiment': 0.0, 'shape': 'xxxx', 'is_alpha': True, 'is_stopwords': False}, {'token': 'Disney', 'lemma': 'Disney', 'pos': 'PROPN', 'tag': 'NNP', 'dependency': 'compound', 'sentiment': 0.0, 'shape': 'Xxxxx', 'is_alpha': True, 'is_stopwords': False}, {'token': 'movie', 'lemma': 'movie', 'pos': 'PROPN', 'tag': 'NNP', 'dependency': 'nsubj', 'sentiment': 0.0, 'shape': 'xxxx', 'is_alpha': True, 'is_stopwords': False}, {'token': ',', 'lemma': ',', 'pos': 'PUNCT', 'tag': ',', 'dependency': 'punct', 'sentiment': 0.0, 'shape': ',', 'is_alpha': False, 'is_stopwords': False}, {'token': 'good', 'lemma': 'good', 'pos': 'ADJ', 'tag': 'JJ', 'dependency': 'amod', 'sentiment': 0.0, 'shape': 'xxxx', 'is_alpha': True, 'is_stopwords': False}, {'token': ',', 'lemma': ',', 'pos': 'PUNCT', 'tag': ',', 'dependency': 'punct', 'sentiment': 0.0, 'shape': ',', 'is_alpha': False, 'is_stopwords': False}, {'token': 'kids', 'lemma': 'kid', 'pos': 'NOUN', 'tag': 'NNS', 'dependency': 'compound', 'sentiment': 0.0, 'shape': 'xxxx', 'is_alpha': True, 'is_stopwords': False}, {'token': 'movie', 'lemma': 'movie', 'pos': 'VERB', 'tag': 'VBP', 'dependency': 'ccomp', 'sentiment': 0.0, 'shape': 'xxxx', 'is_alpha': True, 'is_stopwords': False}, {'token': '.', 'lemma': '.', 'pos': 'PUNCT', 'tag': '.', 'dependency': 'punct', 'sentiment': 0.0, 'shape': '.', 'is_alpha': False, 'is_stopwords': False}], 1: [{'token': 'aside', 'lemma': 'aside', 'pos': 'ADV', 'tag': 'RB', 'dependency': 'advmod', 'sentiment': 0.0, 'shape': 'xxxx', 'is_alpha': True, 'is_stopwords': False}, {'token': 'Dr.', 'lemma': 'Dr.', 'pos': 'PROPN', 'tag': 'NNP', 'dependency': 'compound', 'sentiment': 0.0, 'shape': 'Xx.', 'is_alpha': False, 'is_stopwords': False}, {'token': 'House', 'lemma': 'House', 'pos': 'PROPN', 'tag': 'NNP', 'dependency': 'compound', 'sentiment': 0.0, 'shape': 'Xxxxx', 'is_alpha': True, 'is_stopwords': False}, {'token': 'repeat', 'lemma': 'repeat', 'pos': 'NOUN', 'tag': 'NN', 'dependency': 'nsubj', 'sentiment': 0.0, 'shape': 'xxxx', 'is_alpha': True, 'is_stopwords': False}, {'token': 'missed', 'lemma': 'miss', 'pos': 'VERB', 'tag': 'VBN', 'dependency': 'ccomp', 'sentiment': 0.0, 'shape': 'xxxx', 'is_alpha': True, 'is_stopwords': False}, {'token': ',', 'lemma': ',', 'pos': 'PUNCT', 'tag': ',', 'dependency': 'punct', 'sentiment': 0.0, 'shape': ',', 'is_alpha': False, 'is_stopwords': False}, {'token': 'Desperate', 'lemma': 'Desperate', 'pos': 'PROPN', 'tag': 'NNP', 'dependency': 'compound', 'sentiment': 0.0, 'shape': 'Xxxxx', 'is_alpha': True, 'is_stopwords': False}, {'token': 'Housewives', 'lemma': 'Housewives', 'pos': 'PROPN', 'tag': 'NNP', 'dependency': 'nmod', 'sentiment': 0.0, 'shape': 'Xxxxx', 'is_alpha': True, 'is_stopwords': False}, {'token': '(', 'lemma': '(', 'pos': 'PUNCT', 'tag': '-LRB-', 'dependency': 'punct', 'sentiment': 0.0, 'shape': '(', 'is_alpha': False, 'is_stopwords': False}, {'token': 'new', 'lemma': 'new', 'pos': 'ADJ', 'tag': 'JJ', 'dependency': 'amod', 'sentiment': 0.0, 'shape': 'xxx', 'is_alpha': True, 'is_stopwords': False}, {'token': ')', 'lemma': ')', 'pos': 'PUNCT', 'tag': '-RRB-', 'dependency': 'punct', 'sentiment': 0.0, 'shape': ')', 'is_alpha': False, 'is_stopwords': False}, {'token': 'watch', 'lemma': 'watch', 'pos': 'NOUN', 'tag': 'NN', 'dependency': 'ROOT', 'sentiment': 0.0, 'shape': 'xxxx', 'is_alpha': True, 'is_stopwords': False}, {'token': '.', 'lemma': '.', 'pos': 'PUNCT', 'tag': '.', 'dependency': 'punct', 'sentiment': 0.0, 'shape': '.', 'is_alpha': False, 'is_stopwords': False}], 2: [{'token': 'big', 'lemma': 'big', 'pos': 'ADJ', 'tag': 'JJ', 'dependency': 'amod', 'sentiment': 0.0, 'shape': 'xxx', 'is_alpha': True, 'is_stopwords': False}, {'token': 'fan', 'lemma': 'fan', 'pos': 'NOUN', 'tag': 'NN', 'dependency': 'compound', 'sentiment': 0.0, 'shape': 'xxx', 'is_alpha': True, 'is_stopwords': False}, {'token': 'Stephen', 'lemma': 'Stephen', 'pos': 'PROPN', 'tag': 'NNP', 'dependency': 'compound', 'sentiment': 0.0, 'shape': 'Xxxxx', 'is_alpha': True, 'is_stopwords': False}, {'token': 'Kingwork', 'lemma': 'Kingwork', 'pos': 'PROPN', 'tag': 'NNP', 'dependency': 'ROOT', 'sentiment': 0.0, 'shape': 'Xxxxx', 'is_alpha': True, 'is_stopwords': False}, {'token': ',', 'lemma': ',', 'pos': 'PUNCT', 'tag': ',', 'dependency': 'punct', 'sentiment': 0.0, 'shape': ',', 'is_alpha': False, 'is_stopwords': False}, {'token': 'film', 'lemma': 'film', 'pos': 'VERB', 'tag': 'VBP', 'dependency': 'compound', 'sentiment': 0.0, 'shape': 'xxxx', 'is_alpha': True, 'is_stopwords': False}, {'token': 'greater', 'lemma': 'great', 'pos': 'ADJ', 'tag': 'JJR', 'dependency': 'amod', 'sentiment': 0.0, 'shape': 'xxxx', 'is_alpha': True, 'is_stopwords': False}, {'token': 'fan', 'lemma': 'fan', 'pos': 'NOUN', 'tag': 'NN', 'dependency': 'compound', 'sentiment': 0.0, 'shape': 'xxx', 'is_alpha': True, 'is_stopwords': False}, {'token': 'King', 'lemma': 'king', 'pos': 'NOUN', 'tag': 'NN', 'dependency': 'appos', 'sentiment': 0.0, 'shape': 'Xxxx', 'is_alpha': True, 'is_stopwords': False}, {'token': '.', 'lemma': '.', 'pos': 'PUNCT', 'tag': '.', 'dependency': 'punct', 'sentiment': 0.0, 'shape': '.', 'is_alpha': False, 'is_stopwords': False}, {'token': 'Pet', 'lemma': 'Pet', 'pos': 'PROPN', 'tag': 'NNP', 'dependency': 'compound', 'sentiment': 0.0, 'shape': 'Xxx', 'is_alpha': True, 'is_stopwords': False}, {'token': 'Sematary', 'lemma': 'Sematary', 'pos': 'PROPN', 'tag': 'NNP', 'dependency': 'compound', 'sentiment': 0.0, 'shape': 'Xxxxx', 'is_alpha': True, 'is_stopwords': False}, {'token': 'Creed', 'lemma': 'Creed', 'pos': 'PROPN', 'tag': 'NNP', 'dependency': 'compound', 'sentiment': 0.0, 'shape': 'Xxxxx', 'is_alpha': True, 'is_stopwords': False}, {'token': 'family', 'lemma': 'family', 'pos': 'NOUN', 'tag': 'NN', 'dependency': 'ROOT', 'sentiment': 0.0, 'shape': 'xxxx', 'is_alpha': True, 'is_stopwords': False}, {'token': '.', 'lemma': '.', 'pos': 'PUNCT', 'tag': '.', 'dependency': 'punct', 'sentiment': 0.0, 'shape': '.', 'is_alpha': False, 'is_stopwords': False}]}}