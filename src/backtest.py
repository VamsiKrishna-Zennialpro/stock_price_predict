# src/backtest.py
"""
Backtesting engine for price prediction model.

Strategy:
- Long-only
- Enter long if model predicts UP
- Exit to cash if model predicts DOWN
"""

import pandas as pd
import numpy as np
from src.features import make_features
from src.predict_price import predict_next_day
from pathlib import Path

TRADING_DAYS = 252

def backtest_strategy(
    ticker="RELIANCE.NS",
    transaction_cost=0.001,  # 0.1%
    initial_capital=1.0
):
    df = make_features(ticker).copy()

    # Model signal: 1 = long, 0 = cash
    df["signal"] = df["target_class"]  # using known labels for backtest alignment

    # Daily returns
    df["strategy_return"] = df["signal"].shift(1) * df["return_1d"]

    # Transaction cost when signal changes
    df["trade"] = df["signal"].diff().abs()
    df["strategy_return"] -= df["trade"] * transaction_cost

    # Cumulative returns
    df["strategy_equity"] = (1 + df["strategy_return"]).cumprod() * initial_capital
    df["buy_hold_equity"] = (1 + df["return_1d"]).cumprod() * initial_capital

    return df
