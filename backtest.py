"""
backtest.py

This module runs the weekly stock selection strategy using the trained model.
It selects 5 stocks from the previous week's top performers and 5 from the
previous week's bottom performers.
"""

import numpy as np
import pandas as pd
from stock import Stock
from portfolio import Portfolio


def run_backtest(feature_df, model, scaler=None):
    """
    Runs weekly backtest of the strategy.

    Args:
        feature_df (pandas.DataFrame): Clean feature DataFrame.
        model: Trained neural network.
        scaler: Optional fitted StandardScaler.

    Returns:
        tuple: weekly_results DataFrame and selected_stocks DataFrame.
    """
    weekly_results = []
    selected_rows = []

    weeks = sorted(feature_df["date"].unique())

    for week in weeks[:-1]:
        week_data = feature_df[feature_df["date"] == week].copy()

        if len(week_data) < 30:
            continue

        sorted_data = week_data.sort_values("weekly_return")

        bottom_15 = sorted_data.head(15)
        top_15 = sorted_data.tail(15)

        candidates = []

        for group_name, group_data in [("top", top_15), ("bottom", bottom_15)]:
            for _, row in group_data.iterrows():
                features = [
                    row["mom_1w"],
                    row["mom_2w"],
                    row["mom_4w"],
                    row["vol_4w"],
                    row["score"],
                ]

                feature_array = np.array([features])

                if scaler is not None:
                    feature_array = scaler.transform(feature_array)

                prediction = model.predict(feature_array, verbose=0)[0][0]

                stock = Stock(
                    ticker=row["ticker"],
                    date=row["date"],
                    group=group_name,
                    features={
                        "mom_1w": row["mom_1w"],
                        "mom_2w": row["mom_2w"],
                        "mom_4w": row["mom_4w"],
                        "vol_4w": row["vol_4w"],
                        "score": row["score"],
                    },
                    prediction=prediction,
                    next_return=row["next_week_return"],
                )

                candidates.append(stock)

        top_group = [stock for stock in candidates if stock.group == "top"]
        bottom_group = [stock for stock in candidates if stock.group == "bottom"]

        top_group = sorted(top_group, key=lambda stock: stock.prediction, reverse=True)
        bottom_group = sorted(bottom_group, key=lambda stock: stock.prediction, reverse=True)

        selected = top_group[:5] + bottom_group[:5]

        portfolio = Portfolio()

        for stock in selected:
            portfolio.add_stock(stock)

            selected_rows.append(
                {
                    "date": stock.date,
                    "ticker": stock.ticker,
                    "group": stock.group,
                    "prediction": stock.prediction,
                    "next_week_return": stock.next_return,
                    "mom_1w": stock.features["mom_1w"],
                    "mom_2w": stock.features["mom_2w"],
                    "mom_4w": stock.features["mom_4w"],
                    "vol_4w": stock.features["vol_4w"],
                    "score": stock.features["score"],
                }
            )

        strategy_return = portfolio.get_realized_return()

        benchmark_return = week_data["next_week_return"].mean()

        weekly_results.append(
            {
                "date": week,
                "strategy_return": strategy_return,
                "benchmark_return": benchmark_return,
                "num_selected": len(portfolio),
                "avg_prediction": portfolio.get_average_prediction(),
            }
        )

    weekly_results_df = pd.DataFrame(weekly_results)
    selected_stocks_df = pd.DataFrame(selected_rows)

    return weekly_results_df, selected_stocks_df