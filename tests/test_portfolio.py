"""
test_portfolio.py

Pytest tests for the Portfolio and Stock classes.
"""

from stock import Stock
from portfolio import Portfolio


def make_test_stock(ticker, group, prediction, next_return):
    """
    Creates a simple Stock object for testing.
    """
    features = {
        "mom_1w": 0.01,
        "mom_2w": 0.02,
        "mom_4w": 0.03,
        "vol_4w": 0.01,
        "score": 2.0,
    }

    return Stock(
        ticker=ticker,
        date="2024-01-05",
        group=group,
        features=features,
        prediction=prediction,
        next_return=next_return,
    )


def test_portfolio_add_stock_and_len():
    """
    Tests that stocks can be added and __len__ works.
    """
    portfolio = Portfolio()

    stock1 = make_test_stock("AAPL", "top", 0.70, 0.02)
    stock2 = make_test_stock("MSFT", "bottom", 0.60, -0.01)

    portfolio.add_stock(stock1)
    portfolio.add_stock(stock2)

    assert len(portfolio) == 2


def test_portfolio_average_prediction():
    """
    Tests that average prediction is calculated correctly.
    """
    portfolio = Portfolio()

    portfolio.add_stock(make_test_stock("AAPL", "top", 0.70, 0.02))
    portfolio.add_stock(make_test_stock("MSFT", "bottom", 0.50, 0.01))

    assert portfolio.get_average_prediction() == 0.60


def test_portfolio_realized_return():
    """
    Tests that equal-weight portfolio return is calculated correctly.
    """
    portfolio = Portfolio()

    portfolio.add_stock(make_test_stock("AAPL", "top", 0.70, 0.02))
    portfolio.add_stock(make_test_stock("MSFT", "bottom", 0.50, -0.01))

    assert portfolio.get_realized_return() == 0.005


def test_portfolio_get_tickers():
    """
    Tests that portfolio returns selected ticker symbols.
    """
    portfolio = Portfolio()

    portfolio.add_stock(make_test_stock("AAPL", "top", 0.70, 0.02))
    portfolio.add_stock(make_test_stock("MSFT", "bottom", 0.50, -0.01))

    assert portfolio.get_tickers() == ["AAPL", "MSFT"]


def test_stock_equality_operator():
    """
    Tests the Stock __eq__ operator overload.
    """
    stock1 = make_test_stock("AAPL", "top", 0.70, 0.02)
    stock2 = make_test_stock("AAPL", "bottom", 0.50, -0.01)
    stock3 = make_test_stock("MSFT", "top", 0.70, 0.02)

    assert stock1 == stock2
    assert stock1 != stock3