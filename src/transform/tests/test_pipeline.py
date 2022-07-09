from src.transform.pipeline import *
from src.fixtures.data import FIXTURE_DF, FIXTURE_PIPELINE_EXPECTED_OUTPUT


def test_pipeline():
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
    pp = PipelineTransform(pipe, njobs=1)
    res = pp.transform(FIXTURE_DF, None)
    assert res.to_dict() == FIXTURE_PIPELINE_EXPECTED_OUTPUT
