"""
portfolio.py

This module defines the Portfolio class, which manages a collection of Stock
objects and calculates portfolio-level results.
"""


class Portfolio:
    """
    Represents a portfolio of selected Stock objects.
    """

    def __init__(self):
        """
        Initializes an empty portfolio.
        """
        self.stocks = []

    def add_stock(self, stock):
        """
        Adds a Stock object to the portfolio.

        Args:
            stock: A Stock object.
        """
        self.stocks.append(stock)

    def __len__(self):
        """
        Returns the number of stocks in the portfolio.

        Returns:
            int: Number of stocks.
        """
        return len(self.stocks)

    def __str__(self):
        """
        Returns a readable portfolio summary.

        Returns:
            str: Portfolio description.
        """
        return f"Portfolio with {len(self.stocks)} stocks"

    def get_average_prediction(self):
        """
        Calculates the average prediction score of all stocks.

        Returns:
            float: Average predicted probability.
        """
        if len(self.stocks) == 0:
            return 0.0

        total = 0

        for stock in self.stocks:
            total += stock.prediction

        return total / len(self.stocks)

    def get_realized_return(self):
        """
        Calculates equal-weight realized portfolio return.

        Returns:
            float: Average next-week return of selected stocks.
        """
        if len(self.stocks) == 0:
            return 0.0

        total = 0

        for stock in self.stocks:
            total += stock.next_return

        return total / len(self.stocks)

    def get_tickers(self):
        """
        Returns all ticker symbols in the portfolio.

        Returns:
            list: List of ticker symbols.
        """
        return [stock.ticker for stock in self.stocks]