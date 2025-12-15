# scripts/evaluate_sentiment_models.py
"""
Evaluate VADER (model_version='vader-v1') vs LLM (model_version='llm-v1') labels.

Outputs:
- overall agreement rate
- confusion counts (VADER x LLM)
- per-label precision-like stats (simple)
"""

from collections import Counter, defaultdict
from src.db import SessionLocal
from src.schema import SentimentScore
from sqlalchemy import and_
import argparse
import pandas as pd

def fetch_pairs(session, ticker=None, start=None, end=None):
    # fetch pairs joined by clean_id (same article)
    q = session.query(SentimentScore).filter(SentimentScore.model_version.in_(["vader-v1", "llm-v1"]))
    if ticker:
        q = q.filter(SentimentScore.ticker == ticker)
    if start:
        q = q.filter(SentimentScore.published_at >= start)
    if end:
        q = q.filter(SentimentScore.published_at < end)
    rows = q.all()
    # group by clean_id
    grouped = defaultdict(dict)
    for r in rows:
        grouped[r.clean_id][r.model_version] = r
    # only keep where both present
    pairs = []
    for cid, d in grouped.items():
        if "vader-v1" in d and "llm-v1" in d:
            pairs.append((d["vader-v1"], d["llm-v1"]))
    return pairs

def evaluate(pairs):
    total = len(pairs)
    if total == 0:
        return None
    agree = 0
    conf_mat = Counter()
    label_set = set()
    for vader, llm in pairs:
        vlab = (vader.label or "neutral").lower()
        llab = (llm.label or "neutral").lower()
        label_set.add(vlab); label_set.add(llab)
        if vlab == llab:
            agree += 1
        conf_mat[(vlab, llab)] += 1
    agreement = agree / total
    # create confusion dataframe
    labels = sorted(label_set)
    matrix = pd.DataFrame(0, index=labels, columns=labels)
    for (v, l), cnt in conf_mat.items():
        matrix.loc[v, l] = cnt
    return {"total_pairs": total, "agreement": agreement, "confusion_matrix": matrix}

def main(ticker=None, start=None, end=None):
    session = SessionLocal()
    pairs = fetch_pairs(session, ticker=ticker, start=start, end=end)
    print(f"Found {len(pairs)} pairs with both vader-v1 and llm-v1 labels.")
    res = evaluate(pairs)
    if res is None:
        print("No pairs to evaluate.")
        return
    print(f"Agreement: {res['agreement']*100:.2f}% ({res['total_pairs']} pairs)")
    print("Confusion matrix (rows = vader, cols = llm):")
    print(res["confusion_matrix"].to_markdown())
    session.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()
    main(ticker=args.ticker, start=args.start, end=args.end)
