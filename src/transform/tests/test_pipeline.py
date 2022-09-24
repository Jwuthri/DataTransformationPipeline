import pytest

from src.fixtures.data import FIXTURE_DF
from src.transform.pipeline import *


@pytest.fixture(scope="module")
def dataset():
    return FIXTURE_DF


def test_pipeline(dataset):
    pipeline = Pipeline(
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
    transform = PipelineTransform(pipeline, njobs=1)
    output = transform.transform(dataset, None)
    assert output.to_dict() == {
        "text": {
            1: "Put aside Dr. House repeat missed, Desperate Housewives (new) watch one.",
            2: "big fan Stephen King's work, film made even greater fan King. Pet Sematary Creed family.",
        },
        "polarity": {1: 0, 2: 1},
        "number_words": {1: 11, 2: 15},
        "text_length": {1: 72, 2: 88},
        "freq": {1: 1, 2: 2},
        "lang": {1: "en", 2: "en"},
    }


def test_pipeline_incorrect_result(dataset):
    pipeline = Pipeline(
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
    transform = PipelineTransform(pipeline, njobs=1)
    output = transform.transform(dataset, None)
    assert output.to_dict() != {
        "text": {
            1: "Put aside dr. House repeat missed, Desperate Housewives (new) watch one.",
            2: "big fan Stephen King's work, film made even greater fan King. Pet Sematary Creed family.",
        },
        "polarity": {1: 0, 2: 1},
        "number_words": {1: 11, 2: 15},
        "text_length": {1: 72, 2: 88},
        "freq": {1: 1, 2: 2},
        "lang": {1: "ENGLISH", 2: "ENGLISH"},
    }
