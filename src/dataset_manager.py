from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import psycopg2
from psycopg2 import sql

from src import config


def get_connection():
    return psycopg2.connect(
        host=config.DB_HOST,
        port=config.DB_PORT,
        dbname=config.DB_NAME,
        user=config.DB_USER,
        password=config.DB_PASSWORD,
    )


def get_dataset_config(dataset: str) -> dict[str, Any]:
    dataset_key = dataset.strip().lower()

    if dataset_key not in config.DATASETS:
        raise ValueError(
            f"Unsupported dataset: {dataset}. "
            f"Available datasets: {list(config.DATASETS)}"
        )

    return config.DATASETS[dataset_key]


def load_questions(dataset: str) -> list[dict[str, Any]]:
    dataset_config = get_dataset_config(dataset)
    question_file: Path = dataset_config["questions_file"]

    if not question_file.exists():
        raise FileNotFoundError(
            f"Question file not found: {question_file}"
        )

    with question_file.open(
        "r",
        encoding="utf-8",
    ) as file:
        rows = json.load(file)

    questions: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        db_id = (
            row.get("db_id")
            or row.get("database_id")
            or row.get("database")
        )

        question = (
            row.get("question")
            or row.get("utterance")
            or row.get("text")
        )

        gold_sql = (
            row.get("SQL")
            or row.get("query")
            or row.get("sql")
            or ""
        )

        if not db_id or not question:
            continue

        questions.append(
            {
                "question_id": index,
                "dataset": dataset.lower(),
                "db_id": str(db_id),
                "question": str(question),
                "gold_sql": str(gold_sql),
            }
        )

    return questions


def list_postgres_databases(
    dataset: str,
) -> list[str]:
    prefix = f"{dataset.strip().lower()}_%"

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT schema_name
                FROM information_schema.schemata
                WHERE schema_name LIKE %s
                ORDER BY schema_name
                """,
                (prefix,),
            )

            schemas = [
                row[0]
                for row in cursor.fetchall()
            ]
    finally:
        connection.close()

    dataset_prefix = f"{dataset.strip().lower()}_"

    return [
        schema[len(dataset_prefix):]
        for schema in schemas
        if schema.startswith(dataset_prefix)
    ]


def schema_exists(
    dataset: str,
    db_id: str,
) -> bool:
    schema_name = config.postgres_schema_name(
        dataset,
        db_id,
    )

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.schemata
                    WHERE schema_name = %s
                )
                """,
                (schema_name,),
            )

            return bool(cursor.fetchone()[0])
    finally:
        connection.close()


def get_schema_metadata(
    dataset: str,
    db_id: str,
) -> dict[str, list[dict[str, str]]]:
    schema_name = config.postgres_schema_name(
        dataset,
        db_id,
    )

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    table_name,
                    column_name,
                    data_type
                FROM information_schema.columns
                WHERE table_schema = %s
                ORDER BY table_name, ordinal_position
                """,
                (schema_name,),
            )

            rows = cursor.fetchall()
    finally:
        connection.close()

    metadata: dict[
        str,
        list[dict[str, str]]
    ] = {}

    for table_name, column_name, data_type in rows:
        metadata.setdefault(
            table_name,
            [],
        ).append(
            {
                "column": column_name,
                "type": data_type,
            }
        )

    return metadata


def format_schema_for_prompt(
    dataset: str,
    db_id: str,
) -> str:
    metadata = get_schema_metadata(
        dataset,
        db_id,
    )

    if not metadata:
        return "No schema metadata found."

    sections: list[str] = []

    for table_name, columns in metadata.items():
        column_text = ", ".join(
            f"{column['column']} {column['type']}"
            for column in columns
        )

        sections.append(
            f"Table {table_name}({column_text})"
        )

    return "\n".join(sections)


def execute_query(
    dataset: str,
    db_id: str,
    query: str,
    max_rows: int | None = None,
) -> tuple[list[str], list[tuple]]:
    max_rows = (
        max_rows
        if max_rows is not None
        else config.MAX_RESULT_ROWS
    )

    schema_name = config.postgres_schema_name(
        dataset,
        db_id,
    )

    cleaned_query = query.strip().rstrip(";")

    if not cleaned_query.lower().startswith(
        ("select", "with")
    ):
        raise ValueError(
            "Only SELECT and WITH queries are allowed."
        )

    connection = get_connection()
    connection.autocommit = False

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL(
                    "SET LOCAL search_path TO {}, public"
                ).format(
                    sql.Identifier(schema_name)
                )
            )

            limited_query = (
                f"SELECT * FROM ({cleaned_query}) "
                f"AS generated_query LIMIT {int(max_rows)}"
            )

            cursor.execute(limited_query)

            columns = [
                description.name
                for description in cursor.description
            ]

            rows = cursor.fetchall()

        connection.rollback()

        return columns, rows

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()