"""Integration-test transform jobs."""
from datetime import date
from pyspark.sql import SparkSession, Row

from src.jobs.transform import get_player_stats, get_team_win_ratio, get_match_players, get_normalized_matches, get_feature_set
from src.jobs.extract import extract_file


def test_get_feature_set(spark_session_test: SparkSession) -> None:
    match_df = extract_file(spark_session_test, "./data", "match.csv")
    player_attributes_df = extract_file(spark_session_test, "./data", "player_attributes.csv")
    player_df = extract_file(spark_session_test, "./data", "player.csv")
    features_df = get_feature_set(match_df, player_df, player_attributes_df)
    assert features_df.count() == 2000
    assert sorted(features_df.columns) == sorted(["this_team_api_id", "other_team_api_id", "win_ratio", "average_potential", "has_high_potential_player", "is_playing_home_game", "target"])


def test_get_normalized_matches(spark_session_test: SparkSession) -> None:
    match_df = spark_session_test.createDataFrame([
        Row(match_api_id="a", home_team_api_id=1, away_team_api_id=2, home_team_goal=3, away_team_goal=2),
        Row(match_api_id="b", home_team_api_id=2, away_team_api_id=3, home_team_goal=2, away_team_goal=3),
        Row(match_api_id="c", home_team_api_id=3, away_team_api_id=4, home_team_goal=2, away_team_goal=2),
    ])
    out_df = get_normalized_matches(match_df)
    assert sorted(out_df.collect()) == sorted([
        Row(match_api_id="a", this_team_api_id=1, other_team_api_id=2, is_playing_home_game=True, result="WIN"),
        Row(match_api_id="a", this_team_api_id=2, other_team_api_id=1, is_playing_home_game=False, result="LOSE"),
        Row(match_api_id="b", this_team_api_id=2, other_team_api_id=3, is_playing_home_game=True, result="LOSE"),
        Row(match_api_id="b", this_team_api_id=3, other_team_api_id=2, is_playing_home_game=False, result="WIN"),
        Row(match_api_id="c", this_team_api_id=3, other_team_api_id=4, is_playing_home_game=True, result="TIE"),
        Row(match_api_id="c", this_team_api_id=4, other_team_api_id=3, is_playing_home_game=False, result="TIE"),
    ])

def test_get_match_players(spark_session_test: SparkSession) -> None:
    match_1_home_players = dict(
        zip([f"home_player_{i}" for i in range(1, 12)], range(1, 12))
    )
    match_1_away_players = dict(
        zip([f"away_player_{i}" for i in range(1, 12)], range(12, 23))
    )
    match_2_home_players = dict(
        zip([f"home_player_{i}" for i in range(1, 12)], range(10, 21))
    )
    match_2_away_players = dict(
        zip([f"away_player_{i}" for i in range(1, 12)], range(20, 31))
    )
    match_df = spark_session_test.createDataFrame(
        [
            Row(match_api_id=1, **match_1_home_players, **match_1_away_players),
            Row(match_api_id=2, **match_2_home_players, **match_2_away_players),
        ]
    )
    out_df = get_match_players(match_df)
    assert sorted(out_df.collect()) == sorted(
        [
            Row(match_api_id=1, player_api_id=player_id, is_playing_home_game=True)
            for player_id in match_1_home_players.values()
        ]
        + [
            Row(match_api_id=1, player_api_id=player_id, is_playing_home_game=False)
            for player_id in match_1_away_players.values()
        ]
        + [
            Row(match_api_id=2, player_api_id=player_id, is_playing_home_game=True)
            for player_id in match_2_home_players.values()
        ]
        + [
            Row(match_api_id=2, player_api_id=player_id, is_playing_home_game=False)
            for player_id in match_2_away_players.values()
        ]
    )


def test_get_team_win_ratio(spark_session_test: SparkSession) -> None:
    normalized_match_df = spark_session_test.createDataFrame(
        [
            Row(match_api_id=1, this_team_api_id=1, other_team_api_id=2, result="WIN"),
            Row(match_api_id=1, this_team_api_id=2, other_team_api_id=1, result="LOSE"),
            Row(match_api_id=2, this_team_api_id=1, other_team_api_id=3, result="WIN"),
            Row(match_api_id=2, this_team_api_id=3, other_team_api_id=1, result="LOSE"),
            Row(match_api_id=3, this_team_api_id=2, other_team_api_id=3, result="TIE"),
            Row(match_api_id=3, this_team_api_id=3, other_team_api_id=2, result="TIE"),
            Row(match_api_id=4, this_team_api_id=3, other_team_api_id=4, result="LOSE"),
            Row(match_api_id=4, this_team_api_id=4, other_team_api_id=3, result="WIN"),
            Row(match_api_id=5, this_team_api_id=3, other_team_api_id=4, result="WIN"),
            Row(match_api_id=5, this_team_api_id=4, other_team_api_id=3, result="LOSE"),
        ]
    )
    out_df = get_team_win_ratio(normalized_match_df)
    assert sorted(out_df.collect()) == sorted(
        [
            Row(team_api_id=1, win_ratio=1.0),
            Row(team_api_id=2, win_ratio=0.0),
            Row(team_api_id=3, win_ratio=0.25),
            Row(team_api_id=4, win_ratio=0.5),
        ]
    )


def test_get_player_stats(spark_session_test: SparkSession) -> None:
    match_player_df = spark_session_test.createDataFrame(
        [
            Row(match_api_id="match1", player_api_id=1, is_playing_home_game=True),
            Row(match_api_id="match1", player_api_id=2, is_playing_home_game=True),
            Row(match_api_id="match1", player_api_id=3, is_playing_home_game=False),
            Row(match_api_id="match1", player_api_id=4, is_playing_home_game=False),
            Row(match_api_id="match2", player_api_id=1, is_playing_home_game=True),
            Row(match_api_id="match2", player_api_id=2, is_playing_home_game=False),
        ]
    )
    player_attributes_df = spark_session_test.createDataFrame(
        [
            Row(player_api_id=1, date=date(2021, 1, 1), potential=0.9),
            Row(player_api_id=2, date=date(2021, 1, 1), potential=0.7),
            Row(player_api_id=2, date=date(2021, 1, 2), potential=0.8),
            Row(player_api_id=3, date=date(2021, 1, 1), potential=0.7),
            Row(player_api_id=4, date=date(2021, 1, 1), potential=0.8),
            Row(player_api_id=4, date=date(2021, 1, 2), potential=1.0),
        ]
    )
    out_df = get_player_stats(match_player_df, player_attributes_df)
    assert sorted(out_df.collect()) == sorted(
        [
            Row(
                match_api_id="match1",
                is_playing_home_game=True,
                has_high_potential_player=True,
                average_potential=(0.9 + 0.75) / 2,
            ),
            Row(
                match_api_id="match1",
                is_playing_home_game=False,
                has_high_potential_player=True,
                average_potential=(0.7 + 0.9) / 2,
            ),
            Row(
                match_api_id="match2",
                is_playing_home_game=True,
                has_high_potential_player=True,
                average_potential=0.9,
            ),
            Row(
                match_api_id="match2",
                is_playing_home_game=False,
                has_high_potential_player=False,
                average_potential=0.75,
            ),
        ]
    )
