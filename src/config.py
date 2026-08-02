from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv


# ------------------------------------------------------------------
# Project root and environment loading
# ------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

load_dotenv(ROOT_DIR / ".env")


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

def get_env(
    name: str,
    default: str = "",
) -> str:
    """
    Read an environment variable safely.

    Empty Render environment variables are treated as missing values.
    """

    value = os.getenv(name)

    if value is None:
        return default

    value = str(value).strip()

    if not value:
        return default

    return value


def get_int_env(
    name: str,
    default: int,
) -> int:
    """
    Read an integer environment variable safely.

    Prevents errors such as:
    ValueError: invalid literal for int() with base 10: ''
    """

    value = get_env(
        name,
        str(default),
    )

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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

OPENAI_API_KEY = get_env(
    "OPENAI_API_KEY",
    "",
)

OPENAI_MODEL = get_env(
    "OPENAI_MODEL",
    get_env(
        "LLM_MODEL",
        "gpt-4o-mini",
    ),
)

# core.py may read LLM_MODEL.
LLM_MODEL = OPENAI_MODEL

LLM_PROVIDER = get_env(
    "LLM_PROVIDER",
    get_env(
        "DEFAULT_PROVIDER",
        "openai",
    ),
).lower()

DEFAULT_PROVIDER = LLM_PROVIDER


# ------------------------------------------------------------------
# PostgreSQL configuration
# ------------------------------------------------------------------

DB_HOST = get_env(
    "DB_HOST",
    "localhost",
)

DB_PORT = get_int_env(
    "DB_PORT",
    5432,
)

DB_NAME = get_env(
    "DB_NAME",
    "text2sql_rag",
)

DB_USER = get_env(
    "DB_USER",
    "postgres",
)

DB_PASSWORD = get_env(
    "DB_PASSWORD",
    "",
)


def normalise_database_url(
    value: str,
) -> str:
    """
    Convert a Render PostgreSQL URL into a SQLAlchemy-compatible URL.

    Supported inputs:
        postgres://user:password@host/database
        postgresql://user:password@host/database
        postgresql://user:password@host:5432/database
        postgresql+psycopg2://user:password@host/database

    The function also repairs an accidental empty port such as:
        host:/database
    """

    url = str(value or "").strip()

    # Remove accidental quotes copied from environment settings.
    url = url.strip('"').strip("'")

    # Remove spaces and line breaks accidentally copied into Render.
    url = re.sub(
        r"\s+",
        "",
        url,
    )

    if not url:
        return ""

    # Render or other providers may return postgres://.
    if url.startswith("postgres://"):
        url = (
            "postgresql://"
            + url[len("postgres://"):]
        )

    # Repair a malformed empty port:
    # hostname:/database -> hostname/database
    url = re.sub(
        r"(@[^/?#:]+):/([^/])",
        r"\1/\2",
        url,
    )

    # The project installs psycopg2-binary.
    if url.startswith("postgresql://"):
        url = (
            "postgresql+psycopg2://"
            + url[len("postgresql://"):]
        )

    return url


# DATABASE_URL is the main Render variable.
# SQLALCHEMY_URL is accepted only as a fallback.
raw_database_url = get_env(
    "DATABASE_URL",
    get_env(
        "SQLALCHEMY_URL",
        "",
    ),
)


if raw_database_url:
    DATABASE_URL = normalise_database_url(
        raw_database_url
    )
else:
    # Local development fallback using separate DB variables.
    encoded_user = quote_plus(
        DB_USER
    )

    encoded_password = quote_plus(
        DB_PASSWORD
    )

    DATABASE_URL = (
        "postgresql+psycopg2://"
        f"{encoded_user}:"
        f"{encoded_password}@"
        f"{DB_HOST}:"
        f"{DB_PORT}/"
        f"{DB_NAME}"
    )


if not DATABASE_URL:
    raise RuntimeError(
        "PostgreSQL configuration is missing. "
        "Set DATABASE_URL in Render or configure "
        "DB_HOST, DB_PORT, DB_NAME, DB_USER and DB_PASSWORD."
    )


# core.py checks SQLALCHEMY_URL first.
SQLALCHEMY_URL = DATABASE_URL


# ------------------------------------------------------------------
# Application settings
# ------------------------------------------------------------------

MAX_RESULT_ROWS = get_int_env(
    "MAX_RESULT_ROWS",
    200,
)

TOP_K_TABLES = get_int_env(
    "TOP_K_TABLES",
    5,
)

OUTPUT_DIR_NAME = get_env(
    "OUTPUT_DIR",
    "outputs",
)


# ------------------------------------------------------------------
# Dataset configuration
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# PostgreSQL schema-name helper
# ------------------------------------------------------------------

def clean_identifier(
    value: str,
) -> str:
    """
    Convert a dataset or database name into a safe PostgreSQL identifier.
    """

    cleaned = (
        str(value or "")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    cleaned = re.sub(
        r"[^a-z0-9_]",
        "_",
        cleaned,
    )

    cleaned = re.sub(
        r"_+",
        "_",
        cleaned,
    ).strip("_")

    if not cleaned:
        raise ValueError(
            "PostgreSQL identifier cannot be empty."
        )

    if cleaned[0].isdigit():
        cleaned = f"db_{cleaned}"

    return cleaned[:63]


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

    dataset_name = clean_identifier(
        dataset
    )

    database_name = clean_identifier(
        db_id
    )

    if dataset_name not in {
        "bird",
        "spider",
    }:
        raise ValueError(
            "Dataset must be either "
            "'bird' or 'spider'."
        )

    # Avoid adding the same dataset prefix twice.
    prefix = f"{dataset_name}_"

    if database_name.startswith(prefix):
        return database_name

    return (
        f"{dataset_name}_"
        f"{database_name}"
    )


# ------------------------------------------------------------------
# Safe diagnostic information
# ------------------------------------------------------------------

def database_configuration_summary() -> dict[str, object]:
    """
    Return database configuration details without exposing credentials.
    """

    return {
        "database_url_configured": bool(
            raw_database_url
        ),
        "database_driver": (
            "postgresql+psycopg2"
        ),
        "local_fallback_host": DB_HOST,
        "local_fallback_port": DB_PORT,
        "local_fallback_database": DB_NAME,
        "max_result_rows": MAX_RESULT_ROWS,
    }
