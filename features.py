"""
features.py

This module calculates weekly stock features including momentum, rolling volatility,
risk-adjusted momentum score, and the target variable for machine learning.
"""

import numpy as np
import pandas as pd


def calculate_momentum_score(mom_1w, mom_2w, mom_4w, vol_4w):
    """
    Calculates the risk-adjusted momentum score.

    Args:
        mom_1w (float): One-week momentum.
        mom_2w (float): Two-week momentum.
        mom_4w (float): Four-week momentum.
        vol_4w (float): Four-week rolling volatility.

    Returns:
        float: Risk-adjusted momentum score.
    """
    epsilon = 0.0001

    if pd.isna(mom_1w) or pd.isna(mom_2w) or pd.isna(mom_4w) or pd.isna(vol_4w):
        raise ValueError("Momentum and volatility values cannot be missing.")

    if vol_4w < 0:
        raise ValueError("Volatility cannot be negative.")

    score = (0.2 * mom_1w + 0.3 * mom_2w + 0.5 * mom_4w) / (vol_4w + epsilon)

    return score


def create_feature_table(weekly_prices):
    """
    Creates a long-format feature table for all stocks and weeks.

    Args:
        weekly_prices (pandas.DataFrame): Weekly adjusted close prices.

    Returns:
        pandas.DataFrame: Feature table with one row per stock per week.
    """
    if weekly_prices.empty:
        raise ValueError("Weekly prices cannot be empty.")

    weekly_returns = weekly_prices.pct_change()

    mom_1w = weekly_prices.pct_change(1)
    mom_2w = weekly_prices.pct_change(2)
    mom_4w = weekly_prices.pct_change(4)
    vol_4w = weekly_returns.rolling(4).std()

    # Target: 1 if next week's return is positive, else 0
    target = (weekly_returns.shift(-1) > 0).astype(int)
    next_week_return = weekly_returns.shift(-1)

    rows = []

    for date in weekly_prices.index:
        for ticker in weekly_prices.columns:
            row = {
                "date": date,
                "ticker": ticker,
                "price": weekly_prices.loc[date, ticker],
                "weekly_return": weekly_returns.loc[date, ticker],
                "next_week_return": next_week_return.loc[date, ticker],
                "mom_1w": mom_1w.loc[date, ticker],
                "mom_2w": mom_2w.loc[date, ticker],
                "mom_4w": mom_4w.loc[date, ticker],
                "vol_4w": vol_4w.loc[date, ticker],
                "target": target.loc[date, ticker],
            }

            try:
                row["score"] = calculate_momentum_score(
                    row["mom_1w"],
                    row["mom_2w"],
                    row["mom_4w"],
                    row["vol_4w"],
                )
            except ValueError:
                row["score"] = np.nan

            rows.append(row)

    feature_df = pd.DataFrame(rows)

    return feature_df


def clean_feature_table(feature_df):
    """
    Removes rows with missing feature values.

    Args:
        feature_df (pandas.DataFrame): Raw feature table.

    Returns:
        pandas.DataFrame: Cleaned feature table.
    """
    required_columns = ["mom_1w", "mom_2w", "mom_4w", "vol_4w", "score", "target", "next_week_return"]

    clean_df = feature_df.dropna(subset=required_columns).copy()

    return clean_df


def get_valid_tickers_for_week(feature_df, week, eligible_tickers):
    """
    Filters tickers for one week based on S&P 500 membership and valid feature data.

    Args:
        feature_df (pandas.DataFrame): Feature dataset.
        week: Weekly date.
        eligible_tickers (set): Stocks eligible for that week.

    Returns:
        list: Valid tickers for that week.
    """
    required_columns = ["mom_1w", "mom_2w", "mom_4w", "vol_4w", "score"]

    week_data = feature_df[feature_df["date"] == week]

    valid_tickers = []

    for _, row in week_data.iterrows():
        ticker = row["ticker"]

        if ticker not in eligible_tickers:
            continue

        has_all_features = True

        for column in required_columns:
            if pd.isna(row[column]):
                has_all_features = False

        if has_all_features:
            valid_tickers.append(ticker)

    return valid_tickers


def save_feature_table(feature_df, file_path="data/features.csv"):
    """
    Saves the feature table to a CSV file.

    Args:
        feature_df (pandas.DataFrame): Feature table.
        file_path (str): Output CSV path.
    """
    feature_df.to_csv(file_path, index=False)


if __name__ == "__main__":
    weekly_prices = pd.read_csv("data/weekly_prices.csv", index_col=0, parse_dates=True)

    feature_df = create_feature_table(weekly_prices)
    clean_df = clean_feature_table(feature_df)

    save_feature_table(clean_df, "data/features.csv")

    print("Feature table created successfully.")
    print("Raw feature table shape:", feature_df.shape)
    print("Clean feature table shape:", clean_df.shape)