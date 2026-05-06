"""
test_features.py

Pytest tests for the feature calculation functions.
"""

import pytest
import pandas as pd
import numpy as np

from features import calculate_momentum_score, create_feature_table, clean_feature_table


def test_calculate_momentum_score_valid_values():
    """
    Tests that the momentum score is calculated correctly for normal inputs.
    """
    mom_1w = 0.02
    mom_2w = 0.04
    mom_4w = 0.08
    vol_4w = 0.05

    expected_score = (0.2 * mom_1w + 0.3 * mom_2w + 0.5 * mom_4w) / (vol_4w + 0.0001)

    actual_score = calculate_momentum_score(mom_1w, mom_2w, mom_4w, vol_4w)

    assert actual_score == pytest.approx(expected_score)


def test_calculate_momentum_score_missing_value():
    """
    Tests that the function raises a ValueError when a feature value is missing.
    """
    with pytest.raises(ValueError):
        calculate_momentum_score(np.nan, 0.04, 0.08, 0.05)


def test_calculate_momentum_score_negative_volatility():
    """
    Tests that the function raises a ValueError when volatility is negative.
    """
    with pytest.raises(ValueError):
        calculate_momentum_score(0.02, 0.04, 0.08, -0.05)


def test_create_feature_table_has_required_columns():
    """
    Tests that the feature table contains the required columns.
    """
    dates = pd.date_range(start="2024-01-05", periods=8, freq="W-FRI")

    weekly_prices = pd.DataFrame({
        "AAPL": [100, 102, 104, 103, 106, 108, 110, 112],
        "MSFT": [200, 198, 202, 205, 207, 206, 210, 215]
    }, index=dates)

    feature_df = create_feature_table(weekly_prices)

    required_columns = {
        "date",
        "ticker",
        "price",
        "weekly_return",
        "next_week_return",
        "mom_1w",
        "mom_2w",
        "mom_4w",
        "vol_4w",
        "target",
        "score"
    }

    assert required_columns.issubset(set(feature_df.columns))


def test_clean_feature_table_removes_missing_values():
    """
    Tests that clean_feature_table removes rows with missing required values.
    """
    feature_df = pd.DataFrame({
        "mom_1w": [0.01, np.nan],
        "mom_2w": [0.02, 0.03],
        "mom_4w": [0.04, 0.05],
        "vol_4w": [0.01, 0.02],
        "score": [2.0, 3.0],
        "target": [1, 0],
        "next_week_return": [0.03, 0.01]
    })

    clean_df = clean_feature_table(feature_df)

    assert len(clean_df) == 1
