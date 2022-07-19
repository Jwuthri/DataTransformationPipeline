import pytest

from src.transform.pipeline import *


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


def test_pipeline(dataset):
    pipe = Pipeline(
        [
            (
                "DataFrameColumnsSelection",
                DataFrameColumnsSelection(columns=["text", "polarity"]),
            ),
            (
                "DataFrameTextNumberWords",
                DataFrameTextNumberWords("text", "number_words"),
            ),
            ("DataFrameTextLength", DataFrameTextLength("text", "text_length")),
            ("DataFrameValueFrequency", DataFrameValueFrequency("polarity", "freq")),
            ("DataFrameQueryFilter", DataFrameQueryFilter("number_words", query=">10")),
            ("NlpDetectLanguage", NlpDetectLanguage("text", "lang")),
        ]
    )
    pp = PipelineTransform(pipe, njobs=1)
    output = pp.transform(dataset, None)
    assert output.to_dict() == {'text': {1: 'Put aside Dr. House repeat missed, Desperate Housewives (new) watch one.', 2: "big fan Stephen King's work, film made even greater fan King. Pet Sematary Creed family."}, 'polarity': {1: 0, 2: 1}, 'number_words': {1: 11, 2: 15}, 'text_length': {1: 72, 2: 88}, 'freq': {1: 1, 2: 2}, 'lang': {1: 'ENGLISH', 2: 'ENGLISH'}}


def test_pipeline_incorrect_result(dataset):
    pipe = Pipeline(
        [
            (
                "DataFrameColumnsSelection",
                DataFrameColumnsSelection(columns=["text", "polarity"]),
            ),
            (
                "DataFrameTextNumberWords",
                DataFrameTextNumberWords("text", "number_words"),
            ),
            ("DataFrameTextLength", DataFrameTextLength("text", "text_length")),
            ("DataFrameValueFrequency", DataFrameValueFrequency("polarity", "freq")),
            ("DataFrameQueryFilter", DataFrameQueryFilter("number_words", query=">10")),
            ("NlpDetectLanguage", NlpDetectLanguage("text", "lang")),
        ]
    )
    pp = PipelineTransform(pipe, njobs=1)
    output = pp.transform(dataset, None)
    assert output.to_dict() != {'text': {1: 'Put aside dr. House repeat missed, Desperate Housewives (new) watch one.', 2: "big fan Stephen King's work, film made even greater fan King. Pet Sematary Creed family."}, 'polarity': {1: 0, 2: 1}, 'number_words': {1: 11, 2: 15}, 'text_length': {1: 72, 2: 88}, 'freq': {1: 1, 2: 2}, 'lang': {1: 'ENGLISH', 2: 'ENGLISH'}}
