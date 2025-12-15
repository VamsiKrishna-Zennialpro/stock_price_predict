# scripts/run_backtest.py
import argparse
import matplotlib.pyplot as plt
from src.backtest import backtest_strategy
from src.metrics import sharpe_ratio, max_drawdown, total_return

def run(ticker):
    df = backtest_strategy(ticker)

    strat_ret = df["strategy_return"].dropna()
    bh_ret = df["return_1d"].dropna()

    print("===== BACKTEST RESULTS =====")
    print(f"Ticker: {ticker}")
    print(f"Strategy Return: {total_return(df['strategy_equity']):.2%}")
    print(f"Buy & Hold Return: {total_return(df['buy_hold_equity']):.2%}")
    print(f"Strategy Sharpe: {sharpe_ratio(strat_ret):.2f}")
    print(f"Buy & Hold Sharpe: {sharpe_ratio(bh_ret):.2f}")
    print(f"Strategy Max Drawdown: {max_drawdown(df['strategy_equity']):.2%}")
    print(f"Buy & Hold Max Drawdown: {max_drawdown(df['buy_hold_equity']):.2%}")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df["strategy_equity"], label="Strategy")
    plt.plot(df["date"], df["buy_hold_equity"], label="Buy & Hold")
    plt.legend()
    plt.title(f"Backtest Equity Curve — {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="RELIANCE.NS")
    args = parser.parse_args()
    run(args.ticker)
