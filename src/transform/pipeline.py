from typing import List
from functools import partial
import multiprocessing as mp

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from src.settings import DEBUG, LOGGER


class Selecter(BaseEstimator):

    def __init__(self, columns):
        self.columns = columns

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x[self.columns]


class PipelineTransform(object):

    def __init__(self, pipeline: Pipeline, njobs: int = 2):
        self.pipeline = pipeline
        self.njobs = self.set_njobs(njobs)

    def set_njobs(self, njobs: int):
        max_number_cpu = mp.cpu_count()
        if njobs == -1:
            njobs = max_number_cpu - 1
        else:
            njobs = min(njobs, max_number_cpu)

        return njobs

    def set_pool(self):
        return mp.Pool(self.njobs)

    @staticmethod
    def process_single(pipeline, df):
        return pipeline.fit_transform(df)

    def process_multiple(self, pipeline, df: pd.DataFrame):
        df_splitted = np.array_split(df, self.njobs)
        process_single_mp = partial(self.process_single, pipeline)
        pool = self.set_pool()
        res = pool.map(process_single_mp, df_splitted)
        pool.close()

    @staticmethod
    def read_data(input_file: str, chunksize: int):
        if chunksize is None:
            return [pd.read_csv(input_file)]
        else:
            return pd.read_csv(input_file, chunksize=chunksize)

    def main(self, input_file: str, chunksize: int = None):
        nrows: int = 0
        list_dfs: List[pd.DataFrame] = []
        for chunk_df in self.read_data(input_file, chunksize):
            chunk_df_size = len(chunk_df)
            if DEBUG:
                LOGGER.info(f"working on rows {nrows} to {nrows + chunk_df_size}")
                LOGGER.info(chunk_df.info(memory_usage='deep'))
            nrows += chunk_df_size
            list_dfs.append(self.process_multiple(self.pipeline, chunk_df))

        return list_dfs


if __name__ == '__main__':
    pipe = Pipeline([
        ('selector', Selecter(columns=['stripped_text', 'label']))
    ])
    pp = PipelineTransform(pipe, njobs=1)
    res = pp.main("../../data/raw/macro_suggestion_experiment.csv")
    print(res)
