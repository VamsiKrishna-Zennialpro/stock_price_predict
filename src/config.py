import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parents[1] / ".env")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres_ai:postgres@localhost:5432/postgres_ai")
DATA_DIR = Path(__file__).parents[1] / "data"
