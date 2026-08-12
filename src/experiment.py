from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src import config
from src.core import run_strategy


# ============================================================
# Experiment configuration
# ============================================================

STRATEGIES = {
    "Full schema": "full",
    "Top-1": "top_1",
    "Top-3": "top_3",
    "Top-5": "top_5",
}


DATASET_KEYS = {
    "BIRD Mini-Dev": "bird",
    "Spider": "spider",
}


# ============================================================
# Helpers
# ============================================================

def dataset_key_from_name(
    dataset_name: str,
) -> str:
    """
    Convert the display dataset name used by the UI into the
    dataset key expected by core.py.
    """

    value = str(
        dataset_name or ""
    ).strip()

    if value in DATASET_KEYS:
        return DATASET_KEYS[value]

    return (
        value
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def safe_float(
    value,
    default: float = np.nan,
) -> float:
    """
    Safely convert a value into float.
    """

    try:
        if value is None:
            return default

        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return default


def safe_int(
    value,
    default: int = 0,
) -> int:
    """
    Safely convert a value into int.
    """

    try:
        if value is None:
            return default

        return int(value)

    except (
        TypeError,
        ValueError,
    ):
        return default


# ============================================================
# Controlled Text-to-SQL experiment
# ============================================================

def run_experiment(
    frame: pd.DataFrame,
    limit: int,
    provider: str = "OpenAI",
    progress_callback: (
        Callable[
            [int, int, str],
            None,
        ]
        | None
    ) = None,
) -> dict:
    """
    Run the same benchmark questions using:

        Full schema
        Top-1
        Top-3
        Top-5

    For every question and strategy the experiment:

        1. Reads the real PostgreSQL schema.
        2. Retrieves either the complete schema or Top-K tables.
        3. Generates PostgreSQL from the natural-language question.
        4. Validates the generated SQL.
        5. Executes the generated SQL against Render PostgreSQL.
        6. Executes the benchmark gold SQL.
        7. Compares execution results.
        8. Records latency, token usage and repair attempts.

    The gold SQL is used only for evaluation. It is never used to
    create the schema context for SQL generation.
    """

    if frame.empty:
        raise ValueError(
            "No dataset records are available."
        )

    if str(provider).strip().lower() != "openai":
        raise ValueError(
            "This experiment currently supports the OpenAI provider."
        )

    working = frame.copy()

    required_columns = {
        "dataset",
        "question_id",
        "db_id",
        "question",
        "gold_sql",
    }

    missing = (
        required_columns
        - set(
            working.columns
        )
    )

    if missing:
        raise ValueError(
            "Missing experiment columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    # --------------------------------------------------------
    # Clean benchmark records
    # --------------------------------------------------------

    working = working[
        working["question"]
        .astype(str)
        .str.strip()
        .ne("")
    ].copy()

    working = working[
        working["gold_sql"]
        .astype(str)
        .str.strip()
        .ne("")
    ].copy()

    if working.empty:
        raise ValueError(
            "No benchmark rows with both a question and gold SQL are available."
        )

    if limit:
        working = working.head(
            int(limit)
        ).copy()

    # Ensure optional fields always exist.
    for column, default in {
        "difficulty": "unknown",
        "evidence": "",
        "question_words": np.nan,
        "sql_words": np.nan,
    }.items():
        if column not in working.columns:
            working[column] = default

    # Derive word counts if not already available.
    if working[
        "question_words"
    ].isna().any():
        working[
            "question_words"
        ] = (
            working["question"]
            .astype(str)
            .str.split()
            .str.len()
        )

    if working[
        "sql_words"
    ].isna().any():
        working[
            "sql_words"
        ] = (
            working["gold_sql"]
            .astype(str)
            .str.split()
            .str.len()
        )

    # --------------------------------------------------------
    # Execute experiment
    # --------------------------------------------------------

    rows: list[dict] = []

    total = (
        len(working)
        * len(STRATEGIES)
    )

    done = 0

    for record in working.to_dict(
        orient="records"
    ):
        dataset_name = str(
            record.get(
                "dataset",
                "",
            )
        ).strip()

        dataset_key = (
            dataset_key_from_name(
                dataset_name
            )
        )

        db_id = str(
            record.get(
                "db_id",
                "",
            )
        ).strip()

        question = str(
            record.get(
                "question",
                "",
            )
        ).strip()

        gold_sql = str(
            record.get(
                "gold_sql",
                "",
            )
        ).strip()

        evidence = str(
            record.get(
                "evidence",
                "",
            )
            or ""
        ).strip()

        for (
            strategy_label,
            strategy_key,
        ) in STRATEGIES.items():

            done += 1

            if progress_callback:
                progress_callback(
                    done,
                    total,
                    (
                        f"{dataset_name} · "
                        f"{db_id} · "
                        f"{strategy_label}"
                    ),
                )

            # ----------------------------------------------
            # Run real PostgreSQL Text-to-SQL pipeline
            # ----------------------------------------------

            try:
                result = run_strategy(
                    question=question,
                    db_id=db_id,
                    strategy=strategy_key,
                    gold_sql=gold_sql,
                    dataset=dataset_key,
                    evidence=(
                        evidence
                        or None
                    ),
                    max_repair_attempts=1,
                )

                generated_sql = str(
                    result.get(
                        "generated_sql",
                        "",
                    )
                    or ""
                )

                success = safe_int(
                    result.get(
                        "success"
                    )
                )

                exact_match = (
                    result.get(
                        "exact_match"
                    )
                )

                execution_accuracy = (
                    result.get(
                        "execution_accuracy"
                    )
                )

                generation_latency = (
                    safe_float(
                        result.get(
                            "generation_latency"
                        )
                    )
                )

                execution_time = (
                    safe_float(
                        result.get(
                            "execution_time"
                        )
                    )
                )

                prompt_tokens = (
                    safe_int(
                        result.get(
                            "prompt_tokens"
                        )
                    )
                )

                completion_tokens = (
                    safe_int(
                        result.get(
                            "completion_tokens"
                        )
                    )
                )

                retrieved_tables = (
                    result.get(
                        "retrieved_tables"
                    )
                    or []
                )

                schema_tables = len(
                    retrieved_tables
                )

                repair_attempts = (
                    safe_int(
                        result.get(
                            "repair_attempts"
                        )
                    )
                )

                error = str(
                    result.get(
                        "error",
                        "",
                    )
                    or ""
                )

                postgres_schema = str(
                    result.get(
                        "postgres_schema",
                        "",
                    )
                    or ""
                )

            except Exception as exc:
                generated_sql = ""
                success = 0
                exact_match = 0
                execution_accuracy = 0
                generation_latency = np.nan
                execution_time = np.nan
                prompt_tokens = 0
                completion_tokens = 0
                schema_tables = 0
                repair_attempts = 0
                error = str(exc)
                postgres_schema = ""

            # ----------------------------------------------
            # Store one experimental observation
            # ----------------------------------------------

            rows.append(
                {
                    "dataset": (
                        dataset_name
                    ),

                    "dataset_key": (
                        dataset_key
                    ),

                    "question_id": (
                        record.get(
                            "question_id"
                        )
                    ),

                    "db_id": (
                        db_id
                    ),

                    "postgres_schema": (
                        postgres_schema
                    ),

                    "difficulty": str(
                        record.get(
                            "difficulty",
                            "unknown",
                        )
                        or "unknown"
                    ),

                    "question": (
                        question
                    ),

                    "gold_sql": (
                        gold_sql
                    ),

                    "evidence": (
                        evidence
                    ),

                    "strategy": (
                        strategy_label
                    ),

                    "strategy_key": (
                        strategy_key
                    ),

                    "generated_sql": (
                        generated_sql
                    ),

                    "exact_match": (
                        safe_int(
                            exact_match
                        )
                    ),

                    "execution_accuracy": (
                        safe_int(
                            execution_accuracy
                        )
                    ),

                    "success": (
                        success
                    ),

                    "generation_latency": (
                        generation_latency
                    ),

                    "execution_time": (
                        execution_time
                    ),

                    "prompt_tokens": (
                        prompt_tokens
                    ),

                    "completion_tokens": (
                        completion_tokens
                    ),

                    "total_tokens": (
                        prompt_tokens
                        + completion_tokens
                    ),

                    "question_words": (
                        safe_int(
                            record.get(
                                "question_words"
                            )
                        )
                    ),

                    "sql_words": (
                        safe_int(
                            record.get(
                                "sql_words"
                            )
                        )
                    ),

                    "schema_tables": (
                        schema_tables
                    ),

                    "repair_attempts": (
                        repair_attempts
                    ),

                    "error": (
                        error
                    ),
                }
            )

    # --------------------------------------------------------
    # Raw results
    # --------------------------------------------------------

    results = pd.DataFrame(
        rows
    )

    if results.empty:
        raise RuntimeError(
            "The experiment produced no results."
        )

    config.OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = (
        config.OUTPUT_DIR
        / "experiment_results.csv"
    )

    results.to_csv(
        results_path,
        index=False,
    )

    # --------------------------------------------------------
    # Strategy summary
    # --------------------------------------------------------

    summary = (
        results
        .groupby(
            [
                "dataset",
                "strategy",
            ],
            as_index=False,
        )
        .agg(
            questions=(
                "question_id",
                "count",
            ),

            sql_execution_rate=(
                "success",
                "mean",
            ),

            execution_accuracy=(
                "execution_accuracy",
                "mean",
            ),

            exact_match=(
                "exact_match",
                "mean",
            ),

            generation_latency=(
                "generation_latency",
                "mean",
            ),

            execution_time=(
                "execution_time",
                "mean",
            ),

            prompt_tokens=(
                "prompt_tokens",
                "mean",
            ),

            completion_tokens=(
                "completion_tokens",
                "mean",
            ),

            total_tokens=(
                "total_tokens",
                "mean",
            ),

            schema_tables=(
                "schema_tables",
                "mean",
            ),

            repair_rate=(
                "repair_attempts",
                lambda values: float(
                    (
                        pd.Series(
                            values
                        )
                        > 0
                    ).mean()
                ),
            ),
        )
    )

    # --------------------------------------------------------
    # Add percentage columns for easier dissertation reporting
    # --------------------------------------------------------

    summary[
        "sql_execution_percent"
    ] = (
        summary[
            "sql_execution_rate"
        ]
        * 100
    )

    summary[
        "execution_accuracy_percent"
    ] = (
        summary[
            "execution_accuracy"
        ]
        * 100
    )

    summary[
        "exact_match_percent"
    ] = (
        summary[
            "exact_match"
        ]
        * 100
    )

    summary_path = (
        config.OUTPUT_DIR
        / "dataset_strategy_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # Overall strategy comparison
    # --------------------------------------------------------

    overall = (
        results
        .groupby(
            "strategy",
            as_index=False,
        )
        .agg(
            questions=(
                "question_id",
                "count",
            ),

            execution_accuracy=(
                "execution_accuracy",
                "mean",
            ),

            sql_execution_rate=(
                "success",
                "mean",
            ),

            exact_match=(
                "exact_match",
                "mean",
            ),

            generation_latency=(
                "generation_latency",
                "mean",
            ),

            prompt_tokens=(
                "prompt_tokens",
                "mean",
            ),

            total_tokens=(
                "total_tokens",
                "mean",
            ),

            schema_tables=(
                "schema_tables",
                "mean",
            ),
        )
    )

    overall[
        "execution_accuracy_percent"
    ] = (
        overall[
            "execution_accuracy"
        ]
        * 100
    )

    overall[
        "sql_execution_percent"
    ] = (
        overall[
            "sql_execution_rate"
        ]
        * 100
    )

    overall[
        "exact_match_percent"
    ] = (
        overall[
            "exact_match"
        ]
        * 100
    )

    overall_path = (
        config.OUTPUT_DIR
        / "strategy_overall_summary.csv"
    )

    overall.to_csv(
        overall_path,
        index=False,
    )

    # --------------------------------------------------------
    # Best strategy
    # --------------------------------------------------------

    ranked = summary.sort_values(
        by=[
            "execution_accuracy",
            "prompt_tokens",
            "generation_latency",
        ],
        ascending=[
            False,
            True,
            True,
        ],
    )

    best_strategy = (
        ranked.iloc[0].to_dict()
        if not ranked.empty
        else {}
    )

    return {
        "results": (
            results
        ),

        "summary": (
            summary
        ),

        "overall": (
            overall
        ),

        "best_strategy": (
            best_strategy
        ),

        "path": (
            results_path
        ),

        "summary_path": (
            summary_path
        ),

        "overall_path": (
            overall_path
        ),
    }
