"""
stock.py

This module defines the Stock class used to represent an individual stock
candidate in the momentum stock selection project.
"""


class Stock:
    """
    Represents one stock candidate with its features and prediction score.
    """

    def __init__(self, ticker, date, group, features, prediction=0.0, next_return=0.0):
        """
        Initializes a Stock object.

        Args:
            ticker (str): Stock ticker symbol.
            date: Week date.
            group (str): Either 'top' or 'bottom'.
            features (dict): Dictionary of feature values.
            prediction (float): Neural network predicted probability.
            next_return (float): Actual next-week return.
        """
        self.ticker = ticker
        self.date = date
        self.group = group
        self.features = features
        self.prediction = prediction
        self.next_return = next_return

    def get_feature_vector(self):
        """
        Returns the stock's features as a list for model input.

        Returns:
            list: Feature values in fixed order.
        """
        return [
            self.features["mom_1w"],
            self.features["mom_2w"],
            self.features["mom_4w"],
            self.features["vol_4w"],
            self.features["score"],
        ]

    def is_top_group(self):
        """
        Checks whether the stock came from the top performer group.

        Returns:
            bool: True if stock is in top group.
        """
        return self.group == "top"

    def __str__(self):
        """
        Returns a readable string representation of the Stock object.

        Returns:
            str: Stock description.
        """
        return (
            f"{self.ticker} | Date: {self.date} | Group: {self.group} | "
            f"Prediction: {self.prediction:.4f}"
        )

    def __eq__(self, other):
        """
        Compares two Stock objects by ticker and date.

        Args:
            other (Stock): Another Stock object.

        Returns:
            bool: True if both stocks have the same ticker and date.
        """
        if not isinstance(other, Stock):
            return False

        return self.ticker == other.ticker and self.date == other.date