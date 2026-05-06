# Neural Network Weekly Stock Allocation Strategy

## Team Members
- Nick Habakus - 20015040 - nhabakus@stevens.edu
- Maxim Sergeyev - 20014078 - msergey1@stevens.edu

## Overview

This project develops a machine learning-based stock selection strategy that uses momentum and volatility features to predict short-term stock movements. A neural network is trained on historical S&P 500 data to estimate the probability that a stock will have a positive return in the following week.

Each week, the model selects a portfolio of 10 stocks:
- 5 from the top 15 performers of the previous week
- 5 from the bottom 15 performers of the previous week

The portfolio is rebalanced weekly and evaluated over an out-of-sample test period.

---

## Objective

The goal of this project is to:
- Build a predictive model using financial time series data
- Apply machine learning to a real-world portfolio allocation problem
- Evaluate performance through a systematic backtesting framework
- Compare strategy performance to a benchmark

---

## Data Source

- Data is obtained from Yahoo Finance using the `yfinance` library
- Universe: S&P 500 stocks
- Time period:
  - Training: 2020-01-01 to 2023-12-31
  - Testing: 2024-01-01 to 2025-01-01

To ensure reproducibility and avoid dependency on external APIs, precomputed features are included in:
data/features.csv


---

## Feature Engineering

For each stock and each week, the following features are computed:

- **mom_1w**: 1-week return  
- **mom_2w**: 2-week return  
- **mom_4w**: 4-week return  
- **vol_4w**: 4-week rolling volatility  
- **score**: risk-adjusted momentum  

### Score Formula
score = (0.2 * mom_1w + 0.3 * mom_2w + 0.5 * mom_4w) / (vol_4w + 0.0001)


### Target Variable

- **target = 1** if next week's return > 0  
- **target = 0** otherwise  

---

## Model

A feedforward neural network is used for binary classification.

### Architecture:
- Input layer: 5 features  
- Hidden layers:
  - Dense(32, ReLU)
  - Dense(16, ReLU)
  - Dense(8, ReLU)
- Output layer:
  - Dense(1, Sigmoid)

### Objective:
The model predicts:
P(stock return > 0 next week)


---

## Strategy

Each week in the test period:

1. Rank all stocks by last week's return  
2. Select:
   - Top 15 performers  
   - Bottom 15 performers  
3. Compute features for these 30 stocks  
4. Use the neural network to predict probabilities  
5. Select:
   - 5 highest-probability stocks from top group  
   - 5 highest-probability stocks from bottom group  
6. Form a 10-stock equal-weight portfolio  
7. Hold for one week  
8. Record realized return  

The portfolio is fully rebalanced every week.

---

## Backtesting

The model is trained on historical data (2020–2023) and evaluated on unseen data (2024–2025). Performance is measured using:

- Average weekly return  
- Benchmark comparison  
- Win rate  
- Cumulative growth  

---

## Results

Typical results:

- Test accuracy: ~53–54%  
- Strategy average weekly return: ~0.8–1.0%  
- Positive performance over test period  

Results may vary slightly due to stochastic model training.

---

## Project Structure

momentum_stock_picker/
│
├── data_loader.py
├── features.py
├── model.py
├── backtest.py
├── stock.py
├── portfolio.py
├── main.ipynb
│
├── data/
│ ├── features.csv
│ ├── daily_prices.csv
│ ├── weekly_prices.csv
│ ├── weekly_returns.csv
│ └── sp500_changes.csv
│
├── outputs/
│ ├── weekly_results.csv
│ ├── selected_stocks.csv
│ ├── full_weekly_stock_report.csv
│ ├── stock_selection_counts.csv
│ ├── selected_stocks_detailed.csv
│ └── selected_timeline.csv
│
├── tests/
│ ├── test_features.py
│ └── test_portfolio.py
│
└── README.md


---

## How to Run

### 1. Install dependencies
python -m pip install -r requirements.txt

---

### 2. Run the notebook

Open:
main.ipynb


Run all cells from top to bottom.

---

### Optional: Rebuild dataset

In the notebook, in the second cell, set:
```
REBUILD_DATA = True
```

This will download raw data and recompute features.

### Contributions

Maxim: Portfolio and stock building, creating csvs and loading data, neural network model, troubleshooting

Nick: Test files, feature calculation, backtests model, readME, troubleshooting

### Conclusion

This project demonstrates that combining momentum signals with machine learning can create a systematic stock selection strategy. While results are promising, further improvements could include additional features, more advanced models, and realistic trading constraints.
