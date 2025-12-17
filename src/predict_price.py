# src/predict_price.py
import joblib
from pathlib import Path
from src.features import make_features
from src.observability.langfuse_client import get_langfuse
import time

MODEL_PATH = Path(__file__).parents[1] / "models" / "price_model.pkl"

FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_10d",
    "vol_change",
    "avg_compound",
    "pct_positive",
    "pct_negative",
    "article_count",
]


def predict_next_day(ticker="RELIANCE.NS"):
    lf = get_langfuse()
    start = time.time()

    with lf.trace(
        name="price_prediction",
        input={"ticker": ticker},
        metadata={
            "model": "RandomForest",
            "model_version": "price-v1",
        },
    ) as trace:
        model = joblib.load(MODEL_PATH)
        df = make_features(ticker)

        row = df.iloc[-1]
        X = row[FEATURE_COLUMNS].values.reshape(1, -1)

        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0][1]

        result = {
            "ticker": ticker,
            "prediction": "UP" if pred == 1 else "DOWN",
            "confidence": float(proba),
            "date": str(row["date"]),
        }

        trace.output = result
        trace.score(name="confidence", value=proba)
        trace.metadata["latency_sec"] = time.time() - start

        return result
