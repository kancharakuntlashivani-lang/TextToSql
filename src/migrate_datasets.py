from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sqlite3
import sys
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
from tqdm import tqdm


# Allow imports from the project root when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config


@dataclass
class MigrationResult:
    dataset: str
    database_id: str
    source_file: str
    postgres_schema: str
    table_name: str
    sqlite_rows: int
    postgres_rows: int
    status: str
    message: str = ""


def safe_identifier(value: str) -> str:
    """
    Convert dataset, database, schema, and table names into safe
    PostgreSQL-compatible identifiers.
    """

    cleaned = re.sub(
        r"[^a-zA-Z0-9_]+",
        "_",
        str(value),
    )

    cleaned = cleaned.strip("_").lower()

    if not cleaned:
        cleaned = "unnamed"

    if cleaned[0].isdigit():
        cleaned = f"db_{cleaned}"

    return cleaned[:63]


def get_postgres_connection():
    """
    Create a PostgreSQL connection using values from src/config.py.
    """

    return psycopg2.connect(
        host=config.DB_HOST,
        port=config.DB_PORT,
        dbname=config.DB_NAME,
        user=config.DB_USER,
        password=config.DB_PASSWORD,
    )


def discover_sqlite_file(database_folder: Path) -> Path:
    """
    Find the SQLite database file inside a dataset database folder.
    """

    candidates: list[Path] = []

    for pattern in (
        "*.sqlite",
        "*.sqlite3",
        "*.db",
    ):
        candidates.extend(
            database_folder.rglob(pattern)
        )

    candidates = [
        path
        for path in candidates
        if path.is_file()
    ]

    if not candidates:
        raise FileNotFoundError(
            f"No SQLite database file found in: "
            f"{database_folder}"
        )

    return sorted(
        candidates,
        key=lambda path: (
            len(path.parts),
            len(path.name),
        ),
    )[0]


def sqlite_to_postgres_type(
    sqlite_type: str,
) -> str:
    """
    Map common SQLite column types to PostgreSQL column types.
    """

    value = (
        sqlite_type
        or ""
    ).upper()

    if "INT" in value:
        return "BIGINT"

    if any(
        token in value
        for token in (
            "CHAR",
            "CLOB",
            "TEXT",
            "VARCHAR",
        )
    ):
        return "TEXT"

    if "BLOB" in value:
        return "BYTEA"

    if any(
        token in value
        for token in (
            "REAL",
            "FLOA",
            "DOUB",
        )
    ):
        return "DOUBLE PRECISION"

    if any(
        token in value
        for token in (
            "NUMERIC",
            "DECIMAL",
        )
    ):
        return "NUMERIC"

    if "BOOL" in value:
        return "BOOLEAN"

    if (
        "DATE" in value
        or "TIME" in value
    ):
        # Text is safer because benchmark datasets often contain
        # inconsistent date formats.
        return "TEXT"

    return "TEXT"


def clean_value(
    value: Any,
    target_type: str,
) -> Any:
    """
    Clean SQLite values before inserting them into PostgreSQL.
    """

    if value is None:
        return None

    if isinstance(value, float):
        if (
            math.isnan(value)
            or math.isinf(value)
        ):
            return None

    if target_type == "TEXT":
        if isinstance(value, bytes):
            value = value.decode(
                "utf-8",
                errors="replace",
            )

        return str(value).replace(
            "\x00",
            "",
        )

    if target_type == "BYTEA":
        if isinstance(value, bytes):
            return psycopg2.Binary(value)

        return psycopg2.Binary(
            str(value).encode(
                "utf-8",
                errors="replace",
            )
        )

    if target_type == "BIGINT":
        try:
            return int(value)
        except (
            TypeError,
            ValueError,
            OverflowError,
        ):
            return None

    if target_type == "DOUBLE PRECISION":
        try:
            return float(value)
        except (
            TypeError,
            ValueError,
            OverflowError,
        ):
            return None

    if target_type == "NUMERIC":
        try:
            return Decimal(str(value))
        except Exception:
            return None

    if target_type == "BOOLEAN":
        if isinstance(value, bool):
            return value

        text = str(value).strip().lower()

        if text in {
            "1",
            "true",
            "yes",
            "y",
        }:
            return True

        if text in {
            "0",
            "false",
            "no",
            "n",
        }:
            return False

        return None

    return value


def get_database_root(
    dataset: str,
) -> Path:
    """
    Return the folder that contains all SQLite database directories.
    """

    dataset = dataset.lower()

    if dataset == "bird":
        root = (
            config.BIRD_DATA_DIR
            / "dev_databases"
        )

    elif dataset == "spider":
        preferred = (
            config.SPIDER_DATA_DIR
            / "database"
        )

        if preferred.exists():
            root = preferred
        else:
            # Some Spider downloads place the database folder
            # inside another nested directory.
            matches = [
                path
                for path in config.SPIDER_DATA_DIR.rglob(
                    "database"
                )
                if path.is_dir()
            ]

            if not matches:
                raise FileNotFoundError(
                    "Spider database directory was not found. "
                    "Expected data/spider/database."
                )

            root = matches[0]

    else:
        raise ValueError(
            f"Unsupported dataset: {dataset}"
        )

    if not root.exists():
        raise FileNotFoundError(
            f"Database directory does not exist: {root}"
        )

    return root


def discover_database_folders(
    dataset: str,
) -> list[Path]:
    """
    Discover database folders that contain SQLite files.
    """

    root = get_database_root(dataset)

    folders: list[Path] = []

    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue

        has_sqlite = any(
            child.rglob("*.sqlite")
        ) or any(
            child.rglob("*.sqlite3")
        ) or any(
            child.rglob("*.db")
        )

        if has_sqlite:
            folders.append(child)

    if not folders:
        # Fallback for an uncommon flat database layout.
        sqlite_files = list(
            root.glob("*.sqlite")
        )

        if sqlite_files:
            folders = [root]

    if not folders:
        raise FileNotFoundError(
            f"No SQLite database folders found in: {root}"
        )

    return folders


def get_sqlite_tables(
    sqlite_connection: sqlite3.Connection,
) -> list[str]:
    """
    Return user-created tables from an SQLite database.
    """

    rows = sqlite_connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    ).fetchall()

    return [
        str(row[0])
        for row in rows
    ]


def get_table_columns(
    sqlite_connection: sqlite3.Connection,
    table_name: str,
) -> list[tuple]:
    """
    Read table column metadata from SQLite.
    """

    escaped_name = table_name.replace(
        '"',
        '""',
    )

    return sqlite_connection.execute(
        f'PRAGMA table_info("{escaped_name}")'
    ).fetchall()


def create_schema(
    postgres_connection,
    schema_name: str,
    replace: bool,
) -> None:
    """
    Create or recreate a PostgreSQL schema.
    """

    with postgres_connection.cursor() as cursor:
        if replace:
            cursor.execute(
                sql.SQL(
                    "DROP SCHEMA IF EXISTS {} CASCADE"
                ).format(
                    sql.Identifier(schema_name)
                )
            )

        cursor.execute(
            sql.SQL(
                "CREATE SCHEMA IF NOT EXISTS {}"
            ).format(
                sql.Identifier(schema_name)
            )
        )

    postgres_connection.commit()


def create_postgres_table(
    postgres_connection,
    schema_name: str,
    table_name: str,
    columns: list[tuple],
) -> None:
    """
    Create a PostgreSQL table matching the SQLite table structure.
    """

    definitions = []

    for column in columns:
        original_column_name = str(
            column[1]
        )

        column_type = sqlite_to_postgres_type(
            str(column[2])
        )

        definitions.append(
            sql.SQL("{} {}").format(
                sql.Identifier(
                    original_column_name
                ),
                sql.SQL(column_type),
            )
        )

    with postgres_connection.cursor() as cursor:
        cursor.execute(
            sql.SQL(
                "CREATE TABLE {}.{} ({})"
            ).format(
                sql.Identifier(schema_name),
                sql.Identifier(table_name),
                sql.SQL(", ").join(
                    definitions
                ),
            )
        )

    postgres_connection.commit()


def migrate_table(
    sqlite_connection: sqlite3.Connection,
    postgres_connection,
    schema_name: str,
    table_name: str,
    columns: list[tuple],
    batch_size: int,
) -> tuple[int, int]:
    """
    Copy one SQLite table into PostgreSQL and return row counts.
    """

    escaped_table_name = table_name.replace(
        '"',
        '""',
    )

    column_names = [
        str(column[1])
        for column in columns
    ]

    column_types = [
        sqlite_to_postgres_type(
            str(column[2])
        )
        for column in columns
    ]

    sqlite_row_count = int(
        sqlite_connection.execute(
            f'SELECT COUNT(*) '
            f'FROM "{escaped_table_name}"'
        ).fetchone()[0]
    )

    if sqlite_row_count == 0:
        return 0, 0

    source_cursor = sqlite_connection.execute(
        f'SELECT * FROM "{escaped_table_name}"'
    )

    insert_query = sql.SQL(
        "INSERT INTO {}.{} ({}) VALUES %s"
    ).format(
        sql.Identifier(schema_name),
        sql.Identifier(table_name),
        sql.SQL(", ").join(
            sql.Identifier(name)
            for name in column_names
        ),
    )

    insert_query_text = insert_query.as_string(
        postgres_connection
    )

    with postgres_connection.cursor() as cursor:
        while True:
            batch = source_cursor.fetchmany(
                batch_size
            )

            if not batch:
                break

            cleaned_rows = []

            for row in batch:
                cleaned_row = tuple(
                    clean_value(
                        value,
                        column_types[index],
                    )
                    for index, value
                    in enumerate(row)
                )

                cleaned_rows.append(
                    cleaned_row
                )

            execute_values(
                cursor,
                insert_query_text,
                cleaned_rows,
                page_size=batch_size,
            )

    postgres_connection.commit()

    with postgres_connection.cursor() as cursor:
        cursor.execute(
            sql.SQL(
                "SELECT COUNT(*) FROM {}.{}"
            ).format(
                sql.Identifier(schema_name),
                sql.Identifier(table_name),
            )
        )

        postgres_row_count = int(
            cursor.fetchone()[0]
        )

    return (
        sqlite_row_count,
        postgres_row_count,
    )


def migrate_database(
    dataset: str,
    database_folder: Path,
    replace: bool,
    batch_size: int,
) -> list[MigrationResult]:
    """
    Migrate one BIRD or Spider SQLite database into a PostgreSQL schema.
    """

    database_id = database_folder.name

    schema_name = safe_identifier(
        f"{dataset}_{database_id}"
    )

    results: list[MigrationResult] = []

    try:
        sqlite_file = discover_sqlite_file(
            database_folder
        )

    except Exception as exc:
        return [
            MigrationResult(
                dataset=dataset,
                database_id=database_id,
                source_file="",
                postgres_schema=schema_name,
                table_name="",
                sqlite_rows=0,
                postgres_rows=0,
                status="FAILED",
                message=str(exc),
            )
        ]

    sqlite_connection = sqlite3.connect(
        str(sqlite_file)
    )

    postgres_connection = None

    try:
        postgres_connection = (
            get_postgres_connection()
        )

        create_schema(
            postgres_connection,
            schema_name,
            replace,
        )

        tables = get_sqlite_tables(
            sqlite_connection
        )

        if not tables:
            raise RuntimeError(
                f"No tables found in SQLite database: "
                f"{sqlite_file}"
            )

        for table_name in tqdm(
            tables,
            desc=(
                f"{dataset.upper()} "
                f"{database_id}"
            ),
            leave=False,
        ):
            try:
                columns = get_table_columns(
                    sqlite_connection,
                    table_name,
                )

                if not columns:
                    raise RuntimeError(
                        f"No columns found for table "
                        f"{table_name}"
                    )

                create_postgres_table(
                    postgres_connection,
                    schema_name,
                    table_name,
                    columns,
                )

                (
                    sqlite_rows,
                    postgres_rows,
                ) = migrate_table(
                    sqlite_connection,
                    postgres_connection,
                    schema_name,
                    table_name,
                    columns,
                    batch_size,
                )

                status = (
                    "SUCCESS"
                    if sqlite_rows
                    == postgres_rows
                    else "MISMATCH"
                )

                results.append(
                    MigrationResult(
                        dataset=dataset,
                        database_id=database_id,
                        source_file=str(
                            sqlite_file
                        ),
                        postgres_schema=schema_name,
                        table_name=table_name,
                        sqlite_rows=sqlite_rows,
                        postgres_rows=postgres_rows,
                        status=status,
                    )
                )

            except Exception as table_error:
                postgres_connection.rollback()

                results.append(
                    MigrationResult(
                        dataset=dataset,
                        database_id=database_id,
                        source_file=str(
                            sqlite_file
                        ),
                        postgres_schema=schema_name,
                        table_name=table_name,
                        sqlite_rows=0,
                        postgres_rows=0,
                        status="FAILED",
                        message=str(
                            table_error
                        ),
                    )
                )

    except Exception as database_error:
        if postgres_connection:
            postgres_connection.rollback()

        results.append(
            MigrationResult(
                dataset=dataset,
                database_id=database_id,
                source_file=str(
                    sqlite_file
                ),
                postgres_schema=schema_name,
                table_name="",
                sqlite_rows=0,
                postgres_rows=0,
                status="FAILED",
                message=str(
                    database_error
                ),
            )
        )

    finally:
        sqlite_connection.close()

        if postgres_connection:
            postgres_connection.close()

    return results


def save_migration_report(
    results: list[MigrationResult],
) -> tuple[Path, Path]:
    """
    Save migration results as CSV and JSON.
    """

    config.OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        config.OUTPUT_DIR
        / "dataset_migration_report.csv"
    )

    json_path = (
        config.OUTPUT_DIR
        / "dataset_migration_report.json"
    )

    rows = [
        asdict(result)
        for result in results
    ]

    fieldnames = list(
        MigrationResult
        .__annotations__
        .keys()
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(
        json.dumps(
            rows,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return (
        csv_path,
        json_path,
    )


def print_summary(
    results: list[MigrationResult],
    csv_path: Path,
    json_path: Path,
) -> None:
    """
    Print a migration summary.
    """

    successful = sum(
        result.status == "SUCCESS"
        for result in results
    )

    mismatched = sum(
        result.status == "MISMATCH"
        for result in results
    )

    failed = sum(
        result.status == "FAILED"
        for result in results
    )

    total_rows = sum(
        result.postgres_rows
        for result in results
    )

    database_count = len(
        {
            (
                result.dataset,
                result.database_id,
            )
            for result in results
        }
    )

    summary = {
        "databases_processed":
            database_count,
        "successful_tables":
            successful,
        "mismatched_tables":
            mismatched,
        "failed_tables":
            failed,
        "total_postgres_rows":
            total_rows,
        "csv_report":
            str(csv_path),
        "json_report":
            str(json_path),
    }

    print("\nMigration completed.")

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    failed_results = [
        result
        for result in results
        if result.status == "FAILED"
    ]

    if failed_results:
        print(
            "\nSome tables failed. "
            "Check the CSV report for details."
        )


def test_postgres_connection() -> None:
    """
    Test PostgreSQL before starting a long migration.
    """

    connection = get_postgres_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT current_database(), "
                "current_user"
            )

            database_name, username = (
                cursor.fetchone()
            )

            print(
                "PostgreSQL connection successful."
            )

            print(
                f"Database: {database_name}"
            )

            print(
                f"User: {username}"
            )

    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Migrate BIRD and Spider SQLite "
            "databases into PostgreSQL."
        )
    )

    parser.add_argument(
        "--dataset",
        choices=(
            "bird",
            "spider",
            "all",
        ),
        default="all",
        help=(
            "Dataset to migrate. "
            "Default: all"
        ),
    )

    parser.add_argument(
        "--database",
        type=str,
        default=None,
        help=(
            "Optional single database ID, "
            "for example concert_singer."
        ),
    )

    parser.add_argument(
        "--replace",
        action="store_true",
        help=(
            "Drop and recreate existing "
            "PostgreSQL schemas."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help=(
            "Number of rows inserted per batch."
        ),
    )

    parser.add_argument(
        "--test-connection",
        action="store_true",
        help=(
            "Only test the PostgreSQL connection."
        ),
    )

    arguments = parser.parse_args()

    if arguments.test_connection:
        test_postgres_connection()
        return

    test_postgres_connection()

    datasets = (
        ["bird", "spider"]
        if arguments.dataset == "all"
        else [arguments.dataset]
    )

    all_results: list[
        MigrationResult
    ] = []

    for dataset in datasets:
        database_folders = (
            discover_database_folders(
                dataset
            )
        )

        if arguments.database:
            requested_database = (
                arguments.database
                .strip()
                .lower()
            )

            database_folders = [
                folder
                for folder in database_folders
                if folder.name.lower()
                == requested_database
            ]

            if not database_folders:
                raise FileNotFoundError(
                    f"Database "
                    f"'{arguments.database}' "
                    f"was not found in "
                    f"{dataset.upper()}."
                )

        print(
            f"\n{dataset.upper()}: "
            f"{len(database_folders)} "
            f"database(s) found."
        )

        for index, folder in enumerate(
            database_folders,
            start=1,
        ):
            print(
                f"[{index}/"
                f"{len(database_folders)}] "
                f"Migrating "
                f"{dataset}/"
                f"{folder.name}"
            )

            database_results = (
                migrate_database(
                    dataset=dataset,
                    database_folder=folder,
                    replace=arguments.replace,
                    batch_size=(
                        arguments.batch_size
                    ),
                )
            )

            all_results.extend(
                database_results
            )

    csv_path, json_path = (
        save_migration_report(
            all_results
        )
    )

    print_summary(
        all_results,
        csv_path,
        json_path,
    )


if __name__ == "__main__":
    main()