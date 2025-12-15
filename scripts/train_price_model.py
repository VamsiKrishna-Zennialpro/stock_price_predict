# scripts/train_price_model.py
from src.train_price_model import train_model
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="RELIANCE.NS")
    args = parser.parse_args()

    train_model(args.ticker)
