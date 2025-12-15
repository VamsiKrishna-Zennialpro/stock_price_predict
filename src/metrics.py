# src/metrics.py
import numpy as np

TRADING_DAYS = 252

def sharpe_ratio(returns, risk_free_rate=0.0):
    excess = returns - risk_free_rate / TRADING_DAYS
    if excess.std() == 0:
        return 0.0
    return np.sqrt(TRADING_DAYS) * excess.mean() / excess.std()

def max_drawdown(equity_curve):
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    return drawdown.min()

def total_return(equity_curve):
    return equity_curve.iloc[-1] - 1.0
