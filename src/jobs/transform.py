"""This module is responsible for transform (T) in ETL."""
from functools import reduce
from operator import add

from pyspark.sql import DataFrame, functions as func, Window
from pyspark.sql.types import IntegerType, FloatType


def clean_df_nulls(df):
    return df.na.drop(
        subset=[
            "home_team_goal",
            "away_team_goal",
            "home_team_api_id",
            "away_team_api_id",
        ]
    )


def column_replace_name(df, orig: str, final: str):
    return df.toDF(*(column.replace(orig, final) for column in df.columns))


def transform_match_df(match_df_raw: DataFrame) -> DataFrame:
    cleaned_match_df = clean_df_nulls(match_df_raw)

    # TODO Determine which columns to keep and drop. Currently keeping all
    reduced_match_df = cleaned_match_df.select("*")

    reduced_match_df = reduced_match_df.withColumn(
        "result",
        func.when(
            reduced_match_df.home_team_goal.cast(IntegerType())
            > reduced_match_df.away_team_goal.cast(IntegerType()),
            func.lit("WIN"),
        )
        .when(
            reduced_match_df.home_team_goal.cast(IntegerType())
            == reduced_match_df.away_team_goal.cast(IntegerType()),
            func.lit("TIE"),
        )
        .otherwise(func.lit("LOSE")),
    )
    reduced_match_df = reduced_match_df.na.drop(subset=["result"])

    home_df = column_replace_name(reduced_match_df, "home", "this")
    home_df = column_replace_name(home_df, "away", "other")
    home_df = home_df.withColumn("is_playing_home_game", func.lit(True))

    away_df = column_replace_name(reduced_match_df, "home", "other")
    away_df = column_replace_name(away_df, "away", "this")
    away_df = away_df.withColumn("is_playing_home_game", func.lit(False))

    return home_df.union(away_df)


def get_team_win_ratio(transformed_match_df):
    return (
        transformed_match_df.groupBy("this_team_api_id")
        .agg(
            func.sum((func.col("result") == "WIN").cast(IntegerType())).alias(
                "won_count"
            ),
            func.count("result").alias("games_count"),
        )
        .withColumn("win_ratio", func.col("won_count") / func.col("games_count"))
        .select("this_team_api_id", "win_ratio")
    )


def get_match_players(match_df: DataFrame):
    def match_player_df(df, type_of_team: str, is_home: bool):
        return df.select(
            "match_api_id",
            func.explode(
                func.array(
                    *tuple(f"{type_of_team}_player_{item}" for item in range(1, 12))
                )
            ).alias("player_api_id"),
        ).withColumn("is_home_player", func.lit(is_home))

    home_match_player_df = match_player_df(match_df, "home", True)
    away_match_player_df = match_player_df(match_df, "away", False)
    return home_match_player_df.union(away_match_player_df)


def get_player_stats(
    match_player_df: DataFrame,
    player_attributes_df: DataFrame,
    high_potential_threshold: float = 0.8,
) -> DataFrame:
    average_potential_df = player_attributes_df.groupBy("player_api_id").agg(
        func.mean("potential").alias("potential")
    )
    return (
        match_player_df.join(average_potential_df, "player_api_id", "leftouter")
        .groupBy("match_api_id", "is_playing_home_game")
        .agg(
            func.max("potential").alias("max_potential"),
            func.avg("potential").alias("average_potential"),
        )
        .withColumn(
            "has_high_potential_player",
            func.col("max_potential") > high_potential_threshold,
        )
        .select(
            "match_api_id",
            "is_playing_home_game",
            "has_high_potential_player",
            "average_potential",
        )
    )


def get_average_player_potential(match_df, player_attribute_df):
    match_player_df = get_match_player_df(match_df)
    averaged_player_attributes = (
        player_attribute_df.select(
            "player_api_id", func.col("potential").cast(FloatType())
        )
        .groupBy("player_api_id")
        .agg(func.mean("potential").alias("player_average_potential"))
    )
    return match_player_df.join(
        averaged_player_attributes, "player_api_id", "leftouter"
    )
