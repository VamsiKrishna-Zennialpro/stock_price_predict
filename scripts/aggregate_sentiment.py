# scripts/aggregate_sentiment.py
from sqlalchemy.orm import Session
from sqlalchemy import func
from src.db import SessionLocal
from src.schema import SentimentScore, DailySentiment
import argparse


def aggregate_sentiment(model_version: str):
    session: Session = SessionLocal()

    q = (
        session.query(
            SentimentScore.ticker,
            func.date(SentimentScore.created_at).label("date"),
            func.avg(SentimentScore.compound).label("avg_compound"),
            func.count(SentimentScore.id).label("article_count"),
            func.avg(
                func.case((SentimentScore.label == "positive", 1), else_=0)
            ).label("pct_positive"),
            func.avg(
                func.case((SentimentScore.label == "negative", 1), else_=0)
            ).label("pct_negative"),
        )
        .filter(SentimentScore.model_version == model_version)
        .group_by(
            SentimentScore.ticker,
            func.date(SentimentScore.created_at),
        )
    )

    rows = q.all()

    for r in rows:
        record = (
            session.query(DailySentiment)
            .filter(
                DailySentiment.ticker == r.ticker,
                DailySentiment.date == r.date,
                DailySentiment.model_version == model_version,
            )
            .first()
        )

        if not record:
            record = DailySentiment(
                ticker=r.ticker,
                date=r.date,
                model_version=model_version,
            )

        record.avg_compound = r.avg_compound
        record.article_count = r.article_count
        record.pct_positive = r.pct_positive
        record.pct_negative = r.pct_negative

        session.add(record)

    session.commit()
    session.close()
    print(f"Aggregated sentiment for model_version={model_version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-version", required=True)
    args = parser.parse_args()

    aggregate_sentiment(args.model_version)
