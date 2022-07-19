import pytest

from src.transform.pandas_operator import *


@pytest.fixture(scope='module')
def dataset():
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "type": ['drama', 'comedy', 'thriller'],
            "useless": [0, 0, 0],
            "text": [
                "first think another Disney movie, might good, it's kids movie.",
                "Put aside Dr. House repeat missed, Desperate Housewives (new) watch one.",
                "big fan Stephen King's work, film made even greater fan King. Pet Sematary Creed family.",
            ],
            "polarity": [1, 0, 1]
        }
    )


def test_DataFrameColumnsSelection(dataset):
    pipe = DataFrameColumnsSelection(columns=["text", "polarity"])
    pipe.fit(dataset)
    output = pipe.transform(dataset)
    assert output.to_dict() == {'text': {0: "first think another Disney movie, might good, it's kids movie.", 1: 'Put aside Dr. House repeat missed, Desperate Housewives (new) watch one.', 2: "big fan Stephen King's work, film made even greater fan King. Pet Sematary Creed family."}, 'polarity': {0: 1, 1: 0, 2: 1}}


def test_DataFrameColumnsDrop(dataset):
    pipe = DataFrameColumnsDrop(columns=["useless"])
    pipe.fit(dataset)
    output = pipe.transform(dataset)
    assert output.to_dict() == {'id': {0: 1, 1: 2, 2: 3}, 'type': {0: 'drama', 1: 'comedy', 2: 'thriller'}, 'text': {0: "first think another Disney movie, might good, it's kids movie.", 1: 'Put aside Dr. House repeat missed, Desperate Housewives (new) watch one.', 2: "big fan Stephen King's work, film made even greater fan King. Pet Sematary Creed family."}, 'polarity': {0: 1, 1: 0, 2: 1}}


def test_DataFrameColumnsRename(dataset):
    pipe = DataFrameColumnsRename(columns_mapping={"useless": "new_useless"})
    pipe.fit(dataset)
    output = pipe.transform(dataset)
    assert output.to_dict() == {'id': {0: 1, 1: 2, 2: 3}, 'type': {0: 'drama', 1: 'comedy', 2: 'thriller'}, 'new_useless': {0: 0, 1: 0, 2: 0}, 'text': {0: "first think another Disney movie, might good, it's kids movie.", 1: 'Put aside Dr. House repeat missed, Desperate Housewives (new) watch one.', 2: "big fan Stephen King's work, film made even greater fan King. Pet Sematary Creed family."}, 'polarity': {0: 1, 1: 0, 2: 1}}


def test_DataFrameTextFormat(dataset):
    pipe = DataFrameTextFormat(text_column="text")
    pipe.fit(dataset)
    output = pipe.transform(dataset)
    assert output.to_dict() == {'id': {0: 1, 1: 2, 2: 3}, 'type': {0: 'drama', 1: 'comedy', 2: 'thriller'}, 'useless': {0: 0, 1: 0, 2: 0}, 'text': {0: "first think another disney movie, might good, it's kids movie.", 1: 'put aside dr. house repeat missed, desperate housewives (new) watch one.', 2: "big fan stephen king's work, film made even greater fan king. pet sematary creed family."}, 'polarity': {0: 1, 1: 0, 2: 1}}


def test_DataFrameDropEmptyRows(dataset):
    pipe = DataFrameDropEmptyRows(text_column="text")
    pipe.fit(dataset)
    output = pipe.transform(dataset)
    assert output.to_dict() == {'id': {0: 1, 1: 2, 2: 3}, 'type': {0: 'drama', 1: 'comedy', 2: 'thriller'}, 'useless': {0: 0, 1: 0, 2: 0}, 'text': {0: "first think another disney movie, might good, it's kids movie.", 1: 'put aside dr. house repeat missed, desperate housewives (new) watch one.', 2: "big fan stephen king's work, film made even greater fan king. pet sematary creed family."}, 'polarity': {0: 1, 1: 0, 2: 1}}


def test_DataFrameTextLength(dataset):
    pipe = DataFrameTextLength(text_column="text", new_column="length")
    pipe.fit(dataset)
    output = pipe.transform(dataset)
    assert output.to_dict() == {'id': {0: 1, 1: 2, 2: 3}, 'type': {0: 'drama', 1: 'comedy', 2: 'thriller'}, 'useless': {0: 0, 1: 0, 2: 0}, 'text': {0: "first think another disney movie, might good, it's kids movie.", 1: 'put aside dr. house repeat missed, desperate housewives (new) watch one.', 2: "big fan stephen king's work, film made even greater fan king. pet sematary creed family."}, 'polarity': {0: 1, 1: 0, 2: 1}, 'length': {0: 62, 1: 72, 2: 88}}


def test_DataFrameTextNumberWords(dataset):
    pipe = DataFrameTextNumberWords(text_column="text", new_column="words_number")
    pipe.fit(dataset)
    output = pipe.transform(dataset)
    assert output.to_dict() == {'id': {0: 1, 1: 2, 2: 3}, 'type': {0: 'drama', 1: 'comedy', 2: 'thriller'}, 'useless': {0: 0, 1: 0, 2: 0}, 'text': {0: "first think another disney movie, might good, it's kids movie.", 1: 'put aside dr. house repeat missed, desperate housewives (new) watch one.', 2: "big fan stephen king's work, film made even greater fan king. pet sematary creed family."}, 'polarity': {0: 1, 1: 0, 2: 1}, 'length': {0: 62, 1: 72, 2: 88}, 'words_number': {0: 10, 1: 11, 2: 15}}


def test_DataFrameValueFrequency(dataset):
    pipe = DataFrameValueFrequency(text_column="polarity", new_column="frequency")
    pipe.fit(dataset)
    output = pipe.transform(dataset)
    assert output.to_dict() == {'id': {0: 1, 1: 2, 2: 3}, 'type': {0: 'drama', 1: 'comedy', 2: 'thriller'}, 'useless': {0: 0, 1: 0, 2: 0}, 'text': {0: "first think another disney movie, might good, it's kids movie.", 1: 'put aside dr. house repeat missed, desperate housewives (new) watch one.', 2: "big fan stephen king's work, film made even greater fan king. pet sematary creed family."}, 'polarity': {0: 1, 1: 0, 2: 1}, 'length': {0: 62, 1: 72, 2: 88}, 'words_number': {0: 10, 1: 11, 2: 15}, 'frequency': {0: 2, 1: 1, 2: 2}}


def test_DataFrameExplodeColumn(dataset):
    pipe = DataFrameExplodeColumn(text_column="polarity")
    pipe.fit(dataset)
    output = pipe.transform(dataset)
    assert output.to_dict() == {'id': {0: 1, 1: 2, 2: 3}, 'type': {0: 'drama', 1: 'comedy', 2: 'thriller'}, 'useless': {0: 0, 1: 0, 2: 0}, 'text': {0: "first think another disney movie, might good, it's kids movie.", 1: 'put aside dr. house repeat missed, desperate housewives (new) watch one.', 2: "big fan stephen king's work, film made even greater fan king. pet sematary creed family."}, 'polarity': {0: 1, 1: 0, 2: 1}, 'length': {0: 62, 1: 72, 2: 88}, 'words_number': {0: 10, 1: 11, 2: 15}, 'frequency': {0: 2, 1: 1, 2: 2}}


def test_DataFrameInplodeColumn(dataset):
    pipe = DataFrameInplodeColumn(key_column="polarity", agg_column="type")
    pipe.fit(dataset)
    output = pipe.transform(dataset)
    assert output.to_dict() == {'polarity': {0: 0, 1: 1}, 'type': {0: ['comedy'], 1: ['drama', 'thriller']}}


def test_DataFrameQueryFilter(dataset):
    pipe = DataFrameQueryFilter(text_column="polarity", query="== 1")
    pipe.fit(dataset)
    output = pipe.transform(dataset)
    assert output.to_dict() == {'id': {0: 1, 2: 3}, 'type': {0: 'drama', 2: 'thriller'}, 'useless': {0: 0, 2: 0}, 'text': {0: "first think another disney movie, might good, it's kids movie.", 2: "big fan stephen king's work, film made even greater fan king. pet sematary creed family."}, 'polarity': {0: 1, 2: 1}, 'length': {0: 62, 2: 88}, 'words_number': {0: 10, 2: 15}, 'frequency': {0: 2, 2: 2}}
