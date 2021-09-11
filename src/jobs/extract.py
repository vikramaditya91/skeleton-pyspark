"""This module is responsible for transform (E) in ETL."""
from pyspark.sql import SparkSession, DataFrame


def extract_file(spark: SparkSession, input_dir: str, basename: str) -> DataFrame:
    """
    Extract the local file into a DF.

    Args:
        spark (SparkSession): spark session to read the file
        input_dir (str): file_path to extract
        basename (str): base name of the file

    Returns:
        DataFrame of single-column text file

    """
    return (
        spark.read.option("header", "true")
        .option("delimiter", ",")
        .csv(f"{input_dir}/{basename}")
    )
