from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd

from src import config


# ============================================================
# Built-in datasets
# ============================================================

DATASETS = {
    "BIRD Mini-Dev": {
        "repo": "birdsql/bird_mini_dev",
        "split": "mini_dev_sqlite",
        "sql_field": "SQL",
        "description": (
            "Realistic database-grounded Text-to-SQL benchmark."
        ),
        "type": "builtin",
    },

    "Spider": {
        "repo": "xlangai/spider",
        "split": "validation",
        "sql_field": "query",
        "description": (
            "Complex cross-domain Text-to-SQL benchmark."
        ),
        "type": "builtin",
    },
}


# ============================================================
# Uploaded dataset folders
# ============================================================

UPLOAD_ROOT = (
    config.DATA_DIR
    / "uploaded_datasets"
)

UPLOAD_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Helper functions
# ============================================================

def safe_name(
    value: str,
) -> str:
    """
    Convert a dataset name into a filesystem-safe name.
    """

    cleaned = (
        str(value or "")
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    cleaned = "".join(
        character
        if (
            character.isalnum()
            or character == "_"
        )
        else "_"
        for character in cleaned
    )

    while "__" in cleaned:
        cleaned = cleaned.replace(
            "__",
            "_",
        )

    cleaned = cleaned.strip("_")

    if not cleaned:
        cleaned = "uploaded_dataset"

    return cleaned


def uploaded_dataset_dir(
    dataset_name: str,
) -> Path:
    """
    Return the directory for one uploaded dataset.
    """

    return (
        UPLOAD_ROOT
        / safe_name(dataset_name)
    )


def uploaded_metadata_path(
    dataset_name: str,
) -> Path:
    """
    Return metadata path for an uploaded dataset.
    """

    return (
        uploaded_dataset_dir(
            dataset_name
        )
        / "metadata.json"
    )


# ============================================================
# Dataset normalization
# ============================================================

def _normalise(
    dataset_name: str,
    rows: Iterable[dict],
    sql_field: str | None = None,
) -> pd.DataFrame:
    """
    Convert different Text-to-SQL dataset formats into one
    consistent internal DataFrame.
    """

    if dataset_name in DATASETS:
        spec = DATASETS[
            dataset_name
        ]

        default_sql_field = (
            spec.get(
                "sql_field",
                "query",
            )
        )

    else:
        default_sql_field = (
            "query"
        )

    selected_sql_field = (
        sql_field
        or default_sql_field
    )

    normalised: list[dict] = []

    for idx, row in enumerate(
        rows
    ):
        if not isinstance(
            row,
            dict,
        ):
            continue

        question = str(
            row.get("question")
            or row.get("utterance")
            or row.get("text")
            or ""
        ).strip()

        db_id = str(
            row.get("db_id")
            or row.get("database_id")
            or row.get("database")
            or ""
        ).strip()

        gold_sql = str(
            row.get(
                selected_sql_field
            )
            or row.get("SQL")
            or row.get("sql")
            or row.get("query")
            or row.get("gold_sql")
            or ""
        ).strip()

        evidence = str(
            row.get(
                "evidence",
                "",
            )
            or ""
        ).strip()

        difficulty = str(
            row.get(
                "difficulty",
                row.get(
                    "question_type",
                    "unknown",
                ),
            )
            or "unknown"
        ).strip()

        normalised.append(
            {
                "dataset": (
                    dataset_name
                ),

                "question_id": (
                    row.get(
                        "question_id",
                        row.get(
                            "id",
                            idx,
                        ),
                    )
                ),

                "db_id": db_id,

                "question": (
                    question
                ),

                "gold_sql": (
                    gold_sql
                ),

                "evidence": (
                    evidence
                ),

                "difficulty": (
                    difficulty
                ),
            }
        )

    frame = pd.DataFrame(
        normalised
    )

    if frame.empty:
        return frame

    # Keep benchmark rows that contain a question.
    #
    # Gold SQL is optional for uploaded custom datasets because
    # users may want to ask questions even without benchmark SQL.
    frame = frame[
        frame["question"].ne("")
    ].copy()

    frame[
        "question_words"
    ] = (
        frame["question"]
        .str.split()
        .str.len()
    )

    frame[
        "sql_words"
    ] = (
        frame["gold_sql"]
        .fillna("")
        .astype(str)
        .str.split()
        .str.len()
    )

    return frame.reset_index(
        drop=True
    )


# ============================================================
# Cache paths
# ============================================================

def cache_path(
    dataset_name: str,
) -> Path:
    """
    Return cache location for built-in and uploaded datasets.
    """

    if dataset_name in DATASETS:
        safe = safe_name(
            dataset_name
        )

        return (
            config.DATA_DIR
            / f"{safe}.jsonl"
        )

    return (
        uploaded_dataset_dir(
            dataset_name
        )
        / "questions.jsonl"
    )


# ============================================================
# Built-in dataset download
# ============================================================

def download_dataset(
    dataset_name: str,
    force: bool = False,
) -> pd.DataFrame:
    """
    Download and cache a built-in benchmark dataset.
    """

    if dataset_name not in DATASETS:
        raise ValueError(
            f"Unsupported built-in dataset: "
            f"{dataset_name}"
        )

    path = cache_path(
        dataset_name
    )

    if (
        path.exists()
        and not force
    ):
        return pd.read_json(
            path,
            lines=True,
        )

    spec = DATASETS[
        dataset_name
    ]

    try:
        from datasets import (
            load_dataset,
        )

        dataset = load_dataset(
            spec["repo"],
            split=spec["split"],
        )

    except Exception as exc:
        raise RuntimeError(
            f"Could not download "
            f"{dataset_name}: {exc}"
        ) from exc

    frame = _normalise(
        dataset_name,
        dataset,
        sql_field=spec[
            "sql_field"
        ],
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_json(
        path,
        orient="records",
        lines=True,
    )

    return frame


# ============================================================
# Uploaded dataset registration
# ============================================================

def register_uploaded_dataset(
    dataset_name: str,
    questions_file: str | Path | None = None,
    sqlite_file: str | Path | None = None,
    description: str = "",
) -> dict:
    """
    Register a user-provided dataset.

    Supported:
        questions:
            .json
            .jsonl
            .csv

        database:
            .sqlite
            .db
            .sqlite3
    """

    clean_name = str(
        dataset_name or ""
    ).strip()

    if not clean_name:
        raise ValueError(
            "Dataset name cannot be empty."
        )

    root = uploaded_dataset_dir(
        clean_name
    )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    questions_destination = None
    sqlite_destination = None

    # --------------------------------------------------------
    # Questions file
    # --------------------------------------------------------

    if questions_file:
        source = Path(
            questions_file
        )

        if not source.exists():
            raise FileNotFoundError(
                f"Questions file not found: "
                f"{source}"
            )

        extension = (
            source.suffix
            .lower()
        )

        if extension not in {
            ".json",
            ".jsonl",
            ".csv",
        }:
            raise ValueError(
                "Questions file must be "
                "JSON, JSONL or CSV."
            )

        questions_destination = (
            root
            / f"questions{extension}"
        )

        shutil.copy2(
            source,
            questions_destination,
        )

        frame = (
            load_uploaded_questions_file(
                dataset_name=clean_name,
                path=questions_destination,
            )
        )

        normalized_path = (
            root
            / "questions.jsonl"
        )

        frame.to_json(
            normalized_path,
            orient="records",
            lines=True,
        )

    # --------------------------------------------------------
    # SQLite database
    # --------------------------------------------------------

    if sqlite_file:
        source = Path(
            sqlite_file
        )

        if not source.exists():
            raise FileNotFoundError(
                f"SQLite file not found: "
                f"{source}"
            )

        extension = (
            source.suffix
            .lower()
        )

        if extension not in {
            ".sqlite",
            ".sqlite3",
            ".db",
        }:
            raise ValueError(
                "Database file must be "
                ".sqlite, .sqlite3 or .db."
            )

        sqlite_destination = (
            root
            / f"database{extension}"
        )

        shutil.copy2(
            source,
            sqlite_destination,
        )

    metadata = {
        "dataset_name": (
            clean_name
        ),

        "safe_name": (
            safe_name(
                clean_name
            )
        ),

        "description": (
            description
            or "User uploaded Text-to-SQL dataset."
        ),

        "questions_file": (
            str(
                questions_destination
            )
            if questions_destination
            else None
        ),

        "sqlite_file": (
            str(
                sqlite_destination
            )
            if sqlite_destination
            else None
        ),

        "type": "uploaded",
    }

    uploaded_metadata_path(
        clean_name
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    return metadata


# ============================================================
# Uploaded questions loader
# ============================================================

def load_uploaded_questions_file(
    dataset_name: str,
    path: str | Path,
) -> pd.DataFrame:
    """
    Load an uploaded JSON, JSONL or CSV question file.
    """

    path = Path(
        path
    )

    extension = (
        path.suffix
        .lower()
    )

    if extension == ".csv":
        raw_frame = pd.read_csv(
            path
        )

        rows = raw_frame.to_dict(
            orient="records"
        )

    elif extension == ".jsonl":
        raw_frame = pd.read_json(
            path,
            lines=True,
        )

        rows = raw_frame.to_dict(
            orient="records"
        )

    elif extension == ".json":
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if isinstance(
            payload,
            list,
        ):
            rows = payload

        elif isinstance(
            payload,
            dict,
        ):
            rows = (
                payload.get("data")
                or payload.get(
                    "questions"
                )
                or payload.get(
                    "records"
                )
                or []
            )

        else:
            rows = []

    else:
        raise ValueError(
            "Unsupported questions file."
        )

    return _normalise(
        dataset_name,
        rows,
    )


# ============================================================
# Uploaded dataset discovery
# ============================================================

def list_uploaded_datasets() -> list[str]:
    """
    Return all uploaded dataset names.
    """

    names: list[str] = []

    if not UPLOAD_ROOT.exists():
        return names

    for folder in (
        UPLOAD_ROOT.iterdir()
    ):
        if not folder.is_dir():
            continue

        metadata_file = (
            folder
            / "metadata.json"
        )

        if not metadata_file.exists():
            continue

        try:
            metadata = json.loads(
                metadata_file.read_text(
                    encoding="utf-8"
                )
            )

            name = str(
                metadata.get(
                    "dataset_name",
                    "",
                )
            ).strip()

            if name:
                names.append(
                    name
                )

        except Exception:
            continue

    return sorted(
        names
    )


def get_uploaded_metadata(
    dataset_name: str,
) -> dict:
    """
    Return metadata for an uploaded dataset.
    """

    path = uploaded_metadata_path(
        dataset_name
    )

    if not path.exists():
        return {}

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    except Exception:
        return {}


# ============================================================
# SQLite inspection
# ============================================================

def inspect_sqlite_database(
    sqlite_path: str | Path,
) -> dict[str, list[dict]]:
    """
    Inspect tables and columns in an uploaded SQLite database.
    """

    path = Path(
        sqlite_path
    )

    if not path.exists():
        raise FileNotFoundError(
            str(path)
        )

    connection = sqlite3.connect(
        str(path)
    )

    output: dict[
        str,
        list[dict],
    ] = {}

    try:
        cursor = (
            connection.cursor()
        )

        cursor.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        )

        tables = [
            row[0]
            for row
            in cursor.fetchall()
        ]

        for table in tables:
            escaped = (
                str(table)
                .replace(
                    '"',
                    '""',
                )
            )

            cursor.execute(
                f'PRAGMA table_info('
                f'"{escaped}")'
            )

            columns = []

            for row in (
                cursor.fetchall()
            ):
                columns.append(
                    {
                        "name": row[1],
                        "type": (
                            row[2]
                            or "TEXT"
                        ),
                        "not_null": bool(
                            row[3]
                        ),
                        "primary_key": bool(
                            row[5]
                        ),
                    }
                )

            output[
                str(table)
            ] = columns

    finally:
        connection.close()

    return output


# ============================================================
# Dataset loading
# ============================================================

def load_dataset_frame(
    dataset_name: str,
    auto_download: bool = True,
) -> pd.DataFrame:
    """
    Load either a built-in or uploaded dataset.
    """

    path = cache_path(
        dataset_name
    )

    if path.exists():
        return pd.read_json(
            path,
            lines=True,
        )

    if dataset_name in DATASETS:
        if auto_download:
            return download_dataset(
                dataset_name
            )

        return pd.DataFrame()

    if dataset_name in (
        list_uploaded_datasets()
    ):
        metadata = (
            get_uploaded_metadata(
                dataset_name
            )
        )

        questions_file = (
            metadata.get(
                "questions_file"
            )
        )

        if questions_file:
            return (
                load_uploaded_questions_file(
                    dataset_name,
                    questions_file,
                )
            )

    return pd.DataFrame()


# ============================================================
# All datasets
# ============================================================

def all_dataset_names() -> list[str]:
    """
    Return built-in and uploaded dataset names.
    """

    return (
        list(
            DATASETS.keys()
        )
        + list_uploaded_datasets()
    )


def load_all(
    auto_download: bool = True,
) -> pd.DataFrame:
    """
    Load every available dataset.
    """

    frames: list[
        pd.DataFrame
    ] = []

    for name in all_dataset_names():
        try:
            frame = (
                load_dataset_frame(
                    name,
                    auto_download=(
                        auto_download
                        if name in DATASETS
                        else False
                    ),
                )
            )

            if not frame.empty:
                frames.append(
                    frame
                )

        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
    )


# ============================================================
# Dataset summary
# ============================================================

def dataset_summary(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return dataset-level summary statistics.
    """

    if frame.empty:
        return pd.DataFrame()

    working = frame.copy()

    if (
        "question_words"
        not in working.columns
    ):
        working[
            "question_words"
        ] = (
            working[
                "question"
            ]
            .astype(str)
            .str.split()
            .str.len()
        )

    if (
        "sql_words"
        not in working.columns
    ):
        working[
            "sql_words"
        ] = (
            working[
                "gold_sql"
            ]
            .fillna("")
            .astype(str)
            .str.split()
            .str.len()
        )

    return (
        working
        .groupby(
            "dataset",
            as_index=False,
        )
        .agg(
            questions=(
                "question_id",
                "count",
            ),

            databases=(
                "db_id",
                "nunique",
            ),

            average_question_words=(
                "question_words",
                "mean",
            ),

            average_sql_words=(
                "sql_words",
                "mean",
            ),
        )
    )


# ============================================================
# Uploaded database utilities
# ============================================================

def uploaded_sqlite_path(
    dataset_name: str,
) -> Path | None:
    """
    Return the SQLite database associated with an uploaded dataset.
    """

    metadata = (
        get_uploaded_metadata(
            dataset_name
        )
    )

    value = metadata.get(
        "sqlite_file"
    )

    if not value:
        return None

    path = Path(
        value
    )

    return (
        path
        if path.exists()
        else None
    )


def uploaded_dataset_has_database(
    dataset_name: str,
) -> bool:
    """
    Check whether an uploaded dataset contains a SQLite database.
    """

    return (
        uploaded_sqlite_path(
            dataset_name
        )
        is not None
    )


def uploaded_dataset_has_questions(
    dataset_name: str,
) -> bool:
    """
    Check whether an uploaded dataset contains questions.
    """

    frame = (
        load_dataset_frame(
            dataset_name,
            auto_download=False,
        )
    )

    return not frame.empty
