"""This module is the entry-point for the run.py to handle spark session \
building and ETL."""

import contextlib
from pathlib import Path
from time import time
from typing import Generator

from pyspark.sql import SparkSession

from src.jobs.extract import extract_file
from src.jobs.transform import get_feature_set, get_normalized_matches, get_team_win_ratio, get_player_stats
from src.jobs.load import write_to_path
from src.jobs.utils.general import EnvEnum
from src.jobs.utils.log_utils import Logger


def jobs_main(spark: SparkSession, logger: Logger, input_dir: str) -> None:
    """
    High-level function to perform the ETL job.

    Args:
        spark (SparkSession) : spark session to perform ETL job
        logger (Logger) : logger class instance
        input_dir (str): path on which the job will be performed

    """
    file_names = ["match.csv", "player.csv", "player_attributes.csv"]
    match_df, player_df, player_attributes_df = [
        extract_file(spark, input_dir, item) for item in file_names
    ]

    logger.info(f"Processing {match_df.count()} matches.")
    start_time = time()
    features_df = get_feature_set(match_df, player_df, player_attributes_df).cache()
    write_to_path(features_df)
    end_time = time()
    logger.info(f"Wrote feature set with {features_df.count()} in {end_time - start_time}s")


@contextlib.contextmanager
def spark_build(env: EnvEnum) -> Generator[SparkSession, None, None]:
    """
    Build the spark object.

    Args:
        env (EnvEnum): environment of the spark-application

    Yields:
        SparkSession object

    """
    spark_builder = SparkSession.builder
    app_name = Path(__file__).parent.name

    if env == EnvEnum.dev:
        spark = spark_builder.appName(app_name).getOrCreate()
    else:
        raise NotImplementedError
    try:
        yield spark
    finally:
        spark.stop()
