# src/train_price_model.py
"""
Train a price prediction model (direction: up/down) using sentiment + price features.
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
from pathlib import Path

from src.features import make_features

MODEL_PATH = Path(__file__).parents[1] / "models" / "price_model.pkl"

def train_model(ticker="RELIANCE.NS"):
    df = make_features(ticker)
    feature_cols = [
        "return_1d", "return_5d", "return_10d", "vol_change",
        "avg_compound", "pct_positive", "pct_negative", "article_count"
    ]

    X = df[feature_cols]
    y = df["target_class"]

    tscv = TimeSeriesSplit(n_splits=5)

    best_acc = 0
    best_model = None

    print("Training model with 5-fold time-series CV...")

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        print(f"Fold {fold+1} Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_model = model

    print("\nBest CV Accuracy:", best_acc)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"Model saved at {MODEL_PATH}")

    return best_acc

if __name__ == "__main__":
    train_model("RELIANCE.NS")
