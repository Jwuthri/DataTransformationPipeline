from typing import Union
import multiprocessing as mp

import pandas as pd
import numpy as np

from sklearn import set_config
from sklearn.pipeline import Pipeline

from src.fixtures.data import FIXTURE_DF
from src.settings import DEBUG, LOGGER
from src.transform.pandas_operator import *
from src.transform.nlp_operator import *
from src.utils.decorator import timeit


class PipelineTransform:
    def __init__(self, pipeline: Pipeline, njobs: int = 1) -> None:
        """
        > This function takes a pipeline and a number of jobs as input and sets the number of jobs to the
        number of jobs inputted if the number of jobs is greater than 0, otherwise it sets the number of
        jobs to 2

        :param pipeline: The pipeline object that will be used to train the model
        :type pipeline: Pipeline
        :param njobs: number of jobs to run in parallel, defaults to 2
        :type njobs: int (optional)
        """
        self.pipeline = pipeline
        self.njobs = self.find_optimal_jobs(njobs)

    def find_optimal_jobs(self, njobs: int) -> int:
        """
        > If the number of jobs is -1, then set the number of jobs to the number of CPUs minus 1.
        Otherwise, set the number of jobs to the minimum of the number of jobs and the number of CPUs

        :param njobs: number of parallel jobs to run. If -1, then the number of jobs is set to the
        number of CPU cores on your system minus 1
        :type njobs: int
        :return: The number of jobs.
        """
        max_number_cpu = mp.cpu_count() - 1
        if njobs == -1:
            njobs = max_number_cpu
        else:
            njobs = max(1, min(njobs, max_number_cpu))

        return njobs

    def _pool(self) -> mp.Pool:
        """
        > The function returns a `Pool` object from the `multiprocessing` module, which is
        initialized with the number of jobs specified in the `njobs` attribute of the `self` object

        :return: A Pool object.
        """
        return mp.Pool(self.njobs)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        > The function takes a pipeline and a dataframe as input, and returns a dataframe as output

        :param pipeline: Pipeline
        :type pipeline: Pipeline
        :param df: The dataframe to be processed
        :type df: pd.DataFrame
        :return: A dataframe with the columns that were selected by the pipeline.
        """
        return self.pipeline.fit_transform(df)

    def mp_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        > It splits the dataframe into njobs parts, then it uses a pool of workers to process each part of
        the dataframe

        :param df: pd.DataFrame
        :type df: pd.DataFrame
        :return: A dataframe
        """
        df_splitted = np.array_split(df, self.njobs)
        pool = self._pool()
        datas = pool.map(self.process, df_splitted)
        pool.close()

        return pd.concat(datas)

    @staticmethod
    def read_data(input_file: str, chunksize: int) -> List[pd.DataFrame]:
        """
        > Reads a CSV file into a Pandas DataFrame, either as a single DataFrame or as a list of
        DataFrames, depending on the value of the chunksize parameter

        :param input_file: The path to the file you want to read
        :type input_file: str
        :param chunksize: The number of rows to read in at a time. If None, then all rows are read.
        :type chunksize: int
        :return: A list of dataframes.
        """
        if chunksize is None:
            return [pd.read_csv(input_file, encoding="latin1")]
        else:
            return pd.read_csv(input_file, encoding="latin1", chunksize=chunksize)

    @timeit
    def transform(self, input: Union[str, pd.DataFrame], chunksize: int = None) -> pd.DataFrame:
        """
        > It reads a file or dataframe in chunks, processes each chunk, and returns a list of dataframes

        :param input: input file to read or input pandas dataframe
        :type input: Union[str, pd.DataFrame]
        :param chunksize: how to split dataset into chunks
        :type chunksize: int
        :return: A dataframe
        """
        transformed_dfs: List[pd.DataFrame] = []
        if isinstance(input, str):
            chunks_df = self.read_data(input, chunksize)
        else:
            n = 1 if chunksize is None else len(input) // chunksize
            chunks_df = np.array_split(input, n)
        for chunk_df in chunks_df:
            if DEBUG:
                LOGGER.info(f"working on rows {chunk_df.index.min()} to {chunk_df.index.max()}")
                LOGGER.info(chunk_df.info(memory_usage="deep"))
            transformed_dfs.append(self.mp_process(chunk_df))

        return pd.concat(transformed_dfs)


if __name__ == "__main__": 
    pipeline = Pipeline(
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
    print(pipeline)
    transform = PipelineTransform(pipeline, njobs=1)
    res = transform.transform(FIXTURE_DF, None)
    print(res)
