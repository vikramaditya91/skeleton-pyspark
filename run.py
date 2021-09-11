"""Entry point to the pyspark job."""
import typer
from pathlib import Path

from src.jobs.main import jobs_main, spark_build
from src.jobs.utils.general import EnvEnum
from src.jobs.utils import log_utils


def main(
    env: EnvEnum = typer.Argument(..., help="Environment for the spark-job"),
    input_dir: str = typer.Argument(
        f"file://{Path(__file__).parent}/data", help="File which will be parsed"
    ),
) -> None:
    """Execute main function for the package."""
    with spark_build(env=env) as spark:
        logger = log_utils.Logger(env=env, spark=spark)
        logger.info("Spark and logger initialized")
        jobs_main(spark, logger, input_dir=input_dir)


if __name__ == "__main__":
    typer.run(main)
