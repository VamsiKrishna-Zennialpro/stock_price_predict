# src/backtest.py
"""
Leakage-free backtesting engine for price prediction model.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from src.features import make_features

MODEL_PATH = Path(__file__).parents[1] / "models" / "price_model.pkl"
TRADING_DAYS = 252


def run_backtest(
    ticker="RELIANCE.NS",
    transaction_cost=0.001,   # 0.1%
    initial_capital=1.0,
):
    """
    Proper backtest:
    - Uses model predictions (not ground truth)
    - Signal at t applied to return at t+1
    """

    # Load trained model
    model = joblib.load(MODEL_PATH)

    # Load feature data
    df = make_features(ticker).copy()
    df = df.sort_values("date").reset_index(drop=True)

    feature_cols = [
        "return_1d",
        "return_5d",
        "return_10d",
        "vol_change",
        "avg_compound",
        "pct_positive",
        "pct_negative",
        "article_count",
    ]

    # Predict probabilities
    df["pred_proba"] = model.predict_proba(df[feature_cols])[:, 1]
    df["signal"] = (df["pred_proba"] > 0.5).astype(int)

    # Shift signal so prediction at t is used for t+1 return
    df["signal"] = df["signal"].shift(1).fillna(0)

    # Strategy returns
    df["strategy_return"] = df["signal"] * df["return_1d"]

    # Transaction cost (only when position changes)
    df["trade"] = df["signal"].diff().abs()
    df["strategy_return"] -= df["trade"] * transaction_cost

    # Equity curves
    df["strategy_equity"] = (1 + df["strategy_return"]).cumprod() * initial_capital
    df["buy_hold_equity"] = (1 + df["return_1d"]).cumprod() * initial_capital

    return df
