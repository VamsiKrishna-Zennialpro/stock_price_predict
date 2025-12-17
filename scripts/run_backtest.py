# scripts/run_backtest.py
import matplotlib.pyplot as plt
from src.backtest import run_backtest
from src.metrics import sharpe_ratio, max_drawdown, total_return


def main(ticker="RELIANCE.NS"):
    df = run_backtest(ticker)

    strat_ret = df["strategy_return"].dropna()
    bh_ret = df["return_1d"].dropna()

    print("\n===== BACKTEST RESULTS =====")
    print(f"Ticker: {ticker}")
    print(f"Strategy Return: {total_return(df['strategy_equity']):.2%}")
    print(f"Buy & Hold Return: {total_return(df['buy_hold_equity']):.2%}")
    print(f"Strategy Sharpe: {sharpe_ratio(strat_ret):.2f}")
    print(f"Buy & Hold Sharpe: {sharpe_ratio(bh_ret):.2f}")
    print(f"Strategy Max Drawdown: {max_drawdown(df['strategy_equity']):.2%}")
    print(f"Buy & Hold Max Drawdown: {max_drawdown(df['buy_hold_equity']):.2%}")

    # Plot equity curve
    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df["strategy_equity"], label="Strategy")
    plt.plot(df["date"], df["buy_hold_equity"], label="Buy & Hold")
    plt.legend()
    plt.title(f"Equity Curve — {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
