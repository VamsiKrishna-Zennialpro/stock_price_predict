# scripts/scheduler.py
from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime
import subprocess


def run_pipeline():
    print("Starting daily pipeline:", datetime.utcnow())

    subprocess.run(
        ["python", "scripts/clean_and_label.py", "--from-db"],
        check=True,
    )
    subprocess.run(
        ["python", "scripts/aggregate_sentiment.py", "--model-version", "llm-gemini-v1"],
        check=True,
    )
    subprocess.run(
        ["python", "scripts/build_vectorstore.py"],
        check=True,
    )
    subprocess.run(
        ["python", "scripts/train_price_model.py"],
        check=True,
    )

    print("Pipeline completed successfully")


if __name__ == "__main__":
    scheduler = BlockingScheduler(timezone="Asia/Kolkata")

    # Run every day at 6 AM IST
    scheduler.add_job(run_pipeline, "cron", hour=6, minute=0)

    print("Scheduler started (daily at 6 AM IST)")
    scheduler.start()
