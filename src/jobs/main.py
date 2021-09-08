"""This module is the entry-point for the run.py to handle spark session \
building and ETL."""

import contextlib
from pyspark.sql import SparkSession
from pathlib import Path
from typing import Generator

from src.jobs.extract import extract_file
from src.jobs.transform import transform_df, transform_match_df, get_win_ratio_for_team
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
    match_df, player_df, player_attribute_df = [extract_file(spark, input_dir, item) for item in file_names]

    logger.info(f"{file_names} extracted to DataFrame")

    logger.info(f"match_df has {match_df.count()} rows.")
    transformed_match_df = transform_match_df(match_df)

    count_df = get_win_ratio_for_team(transformed_match_df)
    logger.info("Counted words in the DataFrame")

    load.write_to_path(count_df)
    logger.info("Written counted words to path")


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
