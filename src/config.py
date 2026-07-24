from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]

load_dotenv(ROOT_DIR / ".env")


# ------------------------------------------------------------------
# Project folders
# ------------------------------------------------------------------

DATA_DIR = ROOT_DIR / "data"

BIRD_DATA_DIR = DATA_DIR / "bird"

SPIDER_DATA_DIR = DATA_DIR / "spider"

OUTPUT_DIR = ROOT_DIR / "outputs"

MODEL_DIR = OUTPUT_DIR / "models"

METRICS_DIR = OUTPUT_DIR / "metrics"

PREDICTIONS_DIR = OUTPUT_DIR / "predictions"


for folder in (
    DATA_DIR,
    BIRD_DATA_DIR,
    SPIDER_DATA_DIR,
    OUTPUT_DIR,
    MODEL_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
):
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )


# ------------------------------------------------------------------
# OpenAI configuration
# ------------------------------------------------------------------

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY",
    "",
)

OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-4o-mini",
)

DEFAULT_PROVIDER = os.getenv(
    "DEFAULT_PROVIDER",
    "OpenAI",
)


# ------------------------------------------------------------------
# PostgreSQL configuration
# ------------------------------------------------------------------

DB_HOST = os.getenv(
    "DB_HOST",
    "localhost",
)

DB_PORT = int(
    os.getenv(
        "DB_PORT",
        "5432",
    )
)

DB_NAME = os.getenv(
    "DB_NAME",
    "text2sql_rag",
)

DB_USER = os.getenv(
    "DB_USER",
    "postgres",
)

DB_PASSWORD = os.getenv(
    "DB_PASSWORD",
    "",
)


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    (
        f"postgresql+psycopg2://"
        f"{DB_USER}:"
        f"{DB_PASSWORD}@"
        f"{DB_HOST}:"
        f"{DB_PORT}/"
        f"{DB_NAME}"
    ),
)


# ------------------------------------------------------------------
# Application settings
# ------------------------------------------------------------------

MAX_RESULT_ROWS = int(
    os.getenv(
        "MAX_RESULT_ROWS",
        "200",
    )
)


DATASETS = {
    "bird": {
        "name": "BIRD Mini-Dev",
        "root": BIRD_DATA_DIR,
        "questions_file": (
            BIRD_DATA_DIR
            / "dev.json"
        ),
        "tables_file": (
            BIRD_DATA_DIR
            / "dev_tables.json"
        ),
        "database_directory": (
            BIRD_DATA_DIR
            / "dev_databases"
        ),
        "postgres_prefix": "bird",
    },
    "spider": {
        "name": "Spider",
        "root": SPIDER_DATA_DIR,
        "questions_file": (
            SPIDER_DATA_DIR
            / "dev.json"
        ),
        "tables_file": (
            SPIDER_DATA_DIR
            / "tables.json"
        ),
        "database_directory": (
            SPIDER_DATA_DIR
            / "database"
        ),
        "postgres_prefix": "spider",
    },
}


def postgres_schema_name(
    dataset: str,
    db_id: str,
) -> str:
    """
    Create the PostgreSQL schema name used by the application.

    Examples:
        bird_california_schools
        spider_concert_singer
    """

    dataset_name = (
        str(dataset)
        .strip()
        .lower()
    )

    database_name = (
        str(db_id)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    return (
        f"{dataset_name}_"
        f"{database_name}"
    )