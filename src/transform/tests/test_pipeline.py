import pytest

from src.transform.pipeline import *
from src.data.settings import IMDB_DATA_PATH


def test_pipeline():
    """Testing to load model in memory"""
    pipe = Pipeline(
        [
            (
                "DataFrameColumnsSelection",
                DataFrameColumnsSelection(columns=["text", "polarity"]),
            ),
            ("DataFrameTextLength", DataFrameTextLength("text", "text_length")),
            (
                "DataFrameTextNumberWords",
                DataFrameTextNumberWords("text", "number_words"),
            ),
            ("DataFrameValueFrequency", DataFrameValueFrequency("polarity", "freq")),
            ("DataFrameQueryFilter", DataFrameQueryFilter("number_words", query=">10")),
            ("NlpDetectLanguage", NlpDetectLanguage("text", "lang")),
            ("NlpSpeechTagging", NlpSpeechTagging("text", "pos")),
        ]
    )
    set_config(display="diagram")
    pp = PipelineTransform(pipe, njobs=1)
    res = pp.transform(IMDB_DATA_PATH, None)
    assert 1 == 1
