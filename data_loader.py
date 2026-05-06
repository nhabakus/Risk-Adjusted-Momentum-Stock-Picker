"""
data_loader.py

This module loads S&P 500 stock price data, handles S&P 500 membership changes,
converts daily prices into weekly prices, calculates weekly returns, and saves/loads
CSV files for the project.
"""

from io import StringIO
import requests
import os
import pandas as pd
import yfinance as yf


def get_sp500_tickers(limit=500):
    """
    Gets current S&P 500 ticker symbols from Wikipedia.

    Args:
        limit (int): Number of tickers to return.

    Returns:
        list: List of ticker symbols.
    """
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        sp500_table = pd.read_html(StringIO(response.text))[0]

        tickers = sp500_table["Symbol"].tolist()
        tickers = [ticker.replace(".", "-") for ticker in tickers]

        return tickers[:limit]

    except Exception as error:
        raise RuntimeError(f"Could not load S&P 500 ticker list: {error}")

def get_sp500_changes():
    """
    Gets S&P 500 additions and removals from Wikipedia.

    Returns:
        pandas.DataFrame: DataFrame with date, added, and removed columns.
    """
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(url, headers=headers)
        response.raise_for_status()

        tables = pd.read_html(StringIO(response.text))

        changes_table = None

        for table in tables:
            column_names = [str(column).lower() for column in table.columns]

            if any("date" in column for column in column_names) and any(
                "added" in column for column in column_names
            ):
                changes_table = table
                break

        if changes_table is None:
            raise ValueError("Could not find S&P 500 changes table.")

        if isinstance(changes_table.columns, pd.MultiIndex):
            changes_table.columns = [
                "_".join([str(part) for part in column if str(part) != "nan"]).strip()
                for column in changes_table.columns
            ]

        cleaned_rows = []

        for _, row in changes_table.iterrows():
            row_dict = row.to_dict()

            date_value = None
            added_value = None
            removed_value = None

            for key, value in row_dict.items():
                key_lower = str(key).lower()

                if "date" in key_lower:
                    date_value = value

                if "added" in key_lower and "ticker" in key_lower:
                    added_value = value

                if "removed" in key_lower and "ticker" in key_lower:
                    removed_value = value

            if pd.notna(date_value):
                cleaned_rows.append(
                    {
                        "date": pd.to_datetime(date_value, errors="coerce"),
                        "added": added_value if pd.notna(added_value) else None,
                        "removed": removed_value if pd.notna(removed_value) else None,
                    }
                )

        changes = pd.DataFrame(cleaned_rows)
        changes = changes.dropna(subset=["date"])
        changes = changes.sort_values("date")

        changes["added"] = changes["added"].astype(str).str.replace(".", "-", regex=False)
        changes["removed"] = changes["removed"].astype(str).str.replace(".", "-", regex=False)

        changes.loc[changes["added"].isin(["None", "nan"]), "added"] = None
        changes.loc[changes["removed"].isin(["None", "nan"]), "removed"] = None

        return changes

    except Exception as error:
        raise RuntimeError(f"Could not load S&P 500 changes: {error}")


def download_price_data(tickers, start_date="2020-01-01", end_date="2025-01-01"):
    """
    Downloads daily adjusted close prices for selected tickers.

    Args:
        tickers (list): List of ticker symbols.
        start_date (str): First date of historical data.
        end_date (str): Last date of historical data.

    Returns:
        pandas.DataFrame: Daily adjusted close prices.
    """
    if not tickers:
        raise ValueError("Ticker list cannot be empty.")

    try:
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False,
            group_by="column",
        )

        if "Adj Close" not in data.columns:
            raise ValueError("Adjusted close prices were not found.")

        prices = data["Adj Close"]

        # If only one ticker is downloaded, yfinance may return a Series
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0])

        prices = prices.dropna(axis=1, how="all")

        if prices.empty:
            raise ValueError("Downloaded price data is empty.")

        return prices

    except Exception as error:
        raise RuntimeError(f"Could not download stock price data: {error}")


def convert_to_weekly_prices(daily_prices):
    """
    Converts daily adjusted close prices into weekly Friday prices.

    Args:
        daily_prices (pandas.DataFrame): Daily adjusted close prices.

    Returns:
        pandas.DataFrame: Weekly adjusted close prices.
    """
    if daily_prices.empty:
        raise ValueError("Daily price data cannot be empty.")

    weekly_prices = daily_prices.resample("W-FRI").last()

    # Remove stocks with too little weekly data
    weekly_prices = weekly_prices.dropna(axis=1, thresh=20)

    return weekly_prices


def calculate_weekly_returns(weekly_prices):
    """
    Calculates weekly percentage returns.

    Args:
        weekly_prices (pandas.DataFrame): Weekly adjusted close prices.

    Returns:
        pandas.DataFrame: Weekly percentage returns.
    """
    if weekly_prices.empty:
        raise ValueError("Weekly price data cannot be empty.")

    weekly_returns = weekly_prices.pct_change()

    return weekly_returns


def build_weekly_membership(current_tickers, changes, weekly_dates):
    """
    Builds an approximate S&P 500 membership set for each week.

    This starts from the current S&P 500 list and walks backward using the
    additions/removals table.

    Args:
        current_tickers (list): Current S&P 500 tickers.
        changes (pandas.DataFrame): DataFrame with date, added, and removed columns.
        weekly_dates (pandas.DatetimeIndex): Weekly dates from price data.

    Returns:
        dict: Dictionary mapping each week to a set of eligible tickers.
    """
    if not current_tickers:
        raise ValueError("Current ticker list cannot be empty.")

    if changes.empty:
        raise ValueError("Changes DataFrame cannot be empty.")

    membership_by_week = {}
    current_members = set(current_tickers)

    for week in sorted(weekly_dates, reverse=True):
        eligible = set(current_members)

        # These are changes that happened after the week we are reconstructing
        future_changes = changes[changes["date"] > week]

        for _, change in future_changes.iterrows():
            added = change["added"]
            removed = change["removed"]

            # Going backward: stocks added later should not exist before add date
            if added is not None and added in eligible:
                eligible.remove(added)

            # Going backward: stocks removed later should exist before removal date
            if removed is not None:
                eligible.add(removed)

        membership_by_week[week] = eligible

    return membership_by_week


def save_data(df, file_path):
    """
    Saves a DataFrame to a CSV file.

    Args:
        df (pandas.DataFrame): DataFrame to save.
        file_path (str): Path where the CSV file will be saved.
    """
    folder = os.path.dirname(file_path)

    if folder and not os.path.exists(folder):
        os.makedirs(folder)

    df.to_csv(file_path)


def load_saved_data(file_path):
    """
    Loads a saved CSV file into a DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pandas.DataFrame: Loaded DataFrame.
    """
    try:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)

        if df.empty:
            raise ValueError("Loaded CSV file is empty.")

        return df

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")

    except Exception as error:
        raise RuntimeError(f"Could not load saved CSV file: {error}")


if __name__ == "__main__":
    tickers = get_sp500_tickers(limit=100)

    daily_prices = download_price_data(
        tickers,
        start_date="2020-01-01",
        end_date="2025-01-01",
    )

    weekly_prices = convert_to_weekly_prices(daily_prices)
    weekly_returns = calculate_weekly_returns(weekly_prices)

    changes = get_sp500_changes()
    membership_by_week = build_weekly_membership(
        current_tickers=tickers,
        changes=changes,
        weekly_dates=weekly_prices.index,
    )

    save_data(daily_prices, "data/daily_prices.csv")
    save_data(weekly_prices, "data/weekly_prices.csv")
    save_data(weekly_returns, "data/weekly_returns.csv")
    save_data(changes, "data/sp500_changes.csv")

    print("Data downloaded and saved successfully.")
    print("Number of tickers:", len(tickers))
    print("Daily prices shape:", daily_prices.shape)
    print("Weekly prices shape:", weekly_prices.shape)
    print("Weekly returns shape:", weekly_returns.shape)
    print("Number of membership weeks:", len(membership_by_week))
