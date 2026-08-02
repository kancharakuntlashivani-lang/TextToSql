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

OPENAI_API_KEY = (
    os.getenv("OPENAI_API_KEY") or ""
).strip()

OPENAI_MODEL = (
    os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
).strip()

LLM_MODEL = OPENAI_MODEL

DEFAULT_PROVIDER = (
    os.getenv("DEFAULT_PROVIDER")
    or os.getenv("LLM_PROVIDER")
    or "OpenAI"
).strip()


# ------------------------------------------------------------------
# PostgreSQL configuration
# ------------------------------------------------------------------

DB_HOST = (
    os.getenv("DB_HOST") or "localhost"
).strip()

DB_NAME = (
    os.getenv("DB_NAME") or "text2sql_rag"
).strip()

DB_USER = (
    os.getenv("DB_USER") or "postgres"
).strip()

DB_PASSWORD = os.getenv("DB_PASSWORD") or ""

db_port_value = (
    os.getenv("DB_PORT") or "5432"
).strip()

try:
    DB_PORT = int(db_port_value)
except ValueError:
    DB_PORT = 5432


def normalise_database_url(value: str) -> str:
    """
    Convert a PostgreSQL URL into a SQLAlchemy-compatible URL.
    """

    url = str(value or "").strip()

    url = url.strip('"').strip("'")

    if url.startswith("postgres://"):
        url = (
            "postgresql://"
            + url[len("postgres://"):]
        )

    if url.startswith("postgresql://"):
        url = (
            "postgresql+psycopg2://"
            + url[len("postgresql://"):]
        )

    return url


render_database_url = (
    os.getenv("SQLALCHEMY_URL")
    or os.getenv("DATABASE_URL")
    or ""
).strip()


if render_database_url:
    DATABASE_URL = normalise_database_url(
        render_database_url
    )
else:
    DATABASE_URL = (
        f"postgresql+psycopg2://"
        f"{DB_USER}:"
        f"{DB_PASSWORD}@"
        f"{DB_HOST}:"
        f"{DB_PORT}/"
        f"{DB_NAME}"
    )


# core.py checks SQLALCHEMY_URL first.
SQLALCHEMY_URL = DATABASE_URL


# ------------------------------------------------------------------
# Application settings
# ------------------------------------------------------------------

max_rows_value = (
    os.getenv("MAX_RESULT_ROWS") or "200"
).strip()

try:
    MAX_RESULT_ROWS = int(max_rows_value)
except ValueError:
    MAX_RESULT_ROWS = 200


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
        .replace("-", "_")
        .replace(" ", "_")
    )

    database_name = (
        str(db_id)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    if dataset_name not in {
        "bird",
        "spider",
    }:
        raise ValueError(
            "Dataset must be either "
            "'bird' or 'spider'."
        )

    if not database_name:
        raise ValueError(
            "Database ID cannot be empty."
        )

    return (
        f"{dataset_name}_"
        f"{database_name}"
    )
