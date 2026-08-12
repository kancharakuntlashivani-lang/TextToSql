from __future__ import annotations

import numpy as np
import pandas as pd

from scipy.stats import (
    chi2_contingency,
    mannwhitneyu,
    wilcoxon,
)


# ============================================================
# Helpers
# ============================================================

def _empty_result(
    message: str,
) -> dict:
    return {
        "status": "not_available",
        "message": message,
    }


def _target_column(
    results: pd.DataFrame,
) -> str | None:
    """
    Prefer execution accuracy because two different SQL queries
    can be semantically equivalent even when exact SQL strings differ.
    """

    if "execution_accuracy" in results.columns:
        return "execution_accuracy"

    if "exact_match" in results.columns:
        return "exact_match"

    return None


# ============================================================
# Dataset comparison
# ============================================================

def dataset_comparison_test(
    results: pd.DataFrame,
) -> dict:
    """
    Compare correctness rates between datasets.

    Example:
        BIRD vs Spider

    Chi-square is appropriate here because correctness is binary.
    """

    if results.empty:
        return _empty_result(
            "No experiment results are available."
        )

    target = _target_column(
        results
    )

    if target is None:
        return _empty_result(
            "No correctness column was found."
        )

    data = results[
        [
            "dataset",
            target,
        ]
    ].dropna().copy()

    if data["dataset"].nunique() < 2:
        return _empty_result(
            "Two datasets are required."
        )

    data[target] = (
        data[target]
        .astype(int)
    )

    table = pd.crosstab(
        data["dataset"],
        data[target],
    )

    if table.shape[1] < 2:
        return _empty_result(
            "Both successful and unsuccessful outcomes "
            "are required for a Chi-square test."
        )

    statistic, p_value, dof, expected = (
        chi2_contingency(
            table
        )
    )

    significant = (
        p_value < 0.05
    )

    dataset_rates = (
        data
        .groupby("dataset")[target]
        .mean()
        .mul(100)
        .round(2)
        .to_dict()
    )

    return {
        "status": "ok",
        "test": "Chi-square",
        "target": target,
        "statistic": float(
            statistic
        ),
        "p_value": float(
            p_value
        ),
        "degrees_of_freedom": int(
            dof
        ),
        "significant": bool(
            significant
        ),
        "alpha": 0.05,
        "dataset_accuracy_percent": (
            dataset_rates
        ),
        "interpretation": (
            "There is a statistically significant "
            "difference in correctness between the datasets."
            if significant
            else
            "No statistically significant difference "
            "in correctness was detected between the datasets."
        ),
    }


# ============================================================
# Paired strategy comparison
# ============================================================

def paired_strategy_test(
    results: pd.DataFrame,
    baseline: str = "Full schema",
    comparison: str = "Top-5",
) -> dict:
    """
    Compare two schema strategies using the SAME questions.

    Because each benchmark question is evaluated under both
    strategies, this is a paired comparison.

    Wilcoxon signed-rank is used for the paired binary/ordinal
    correctness observations.
    """

    if results.empty:
        return _empty_result(
            "No experiment results are available."
        )

    target = _target_column(
        results
    )

    if target is None:
        return _empty_result(
            "No correctness column was found."
        )

    required = {
        "dataset",
        "question_id",
        "strategy",
        target,
    }

    missing = (
        required
        - set(results.columns)
    )

    if missing:
        return _empty_result(
            "Missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    subset = results[
        results["strategy"].isin(
            [
                baseline,
                comparison,
            ]
        )
    ].copy()

    if subset.empty:
        return _empty_result(
            f"No rows were found for "
            f"{baseline} and {comparison}."
        )

    pivot = subset.pivot_table(
        index=[
            "dataset",
            "question_id",
        ],
        columns="strategy",
        values=target,
        aggfunc="first",
    )

    if (
        baseline not in pivot.columns
        or comparison not in pivot.columns
    ):
        return _empty_result(
            "Both strategies must contain results "
            "for the same benchmark questions."
        )

    paired = pivot[
        [
            baseline,
            comparison,
        ]
    ].dropna()

    if paired.empty:
        return _empty_result(
            "No paired benchmark observations were found."
        )

    baseline_values = (
        paired[baseline]
        .astype(float)
        .to_numpy()
    )

    comparison_values = (
        paired[comparison]
        .astype(float)
        .to_numpy()
    )

    baseline_accuracy = float(
        np.mean(
            baseline_values
        )
    )

    comparison_accuracy = float(
        np.mean(
            comparison_values
        )
    )

    difference = (
        comparison_values
        - baseline_values
    )

    improved = int(
        np.sum(
            difference > 0
        )
    )

    worsened = int(
        np.sum(
            difference < 0
        )
    )

    unchanged = int(
        np.sum(
            difference == 0
        )
    )

    # Wilcoxon cannot be calculated when every paired
    # observation is identical.
    if np.all(
        difference == 0
    ):
        return {
            "status": "ok",
            "test": (
                "Wilcoxon signed-rank"
            ),
            "target": target,
            "baseline": baseline,
            "comparison": comparison,
            "paired_questions": int(
                len(paired)
            ),
            "baseline_accuracy": (
                baseline_accuracy
            ),
            "comparison_accuracy": (
                comparison_accuracy
            ),
            "baseline_accuracy_percent": (
                baseline_accuracy
                * 100
            ),
            "comparison_accuracy_percent": (
                comparison_accuracy
                * 100
            ),
            "absolute_improvement_percent": (
                (
                    comparison_accuracy
                    - baseline_accuracy
                )
                * 100
            ),
            "improved_questions": (
                improved
            ),
            "worsened_questions": (
                worsened
            ),
            "unchanged_questions": (
                unchanged
            ),
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "alpha": 0.05,
            "interpretation": (
                "The two strategies produced identical "
                "correctness outcomes for every paired question."
            ),
        }

    try:
        statistic, p_value = (
            wilcoxon(
                comparison_values,
                baseline_values,
                zero_method="wilcox",
                alternative="two-sided",
            )
        )

    except ValueError as exc:
        return _empty_result(
            f"Wilcoxon test could not be calculated: {exc}"
        )

    significant = (
        p_value < 0.05
    )

    improvement_percentage = (
        (
            comparison_accuracy
            - baseline_accuracy
        )
        * 100
    )

    return {
        "status": "ok",
        "test": (
            "Wilcoxon signed-rank"
        ),
        "target": target,
        "baseline": baseline,
        "comparison": comparison,
        "paired_questions": int(
            len(paired)
        ),
        "baseline_accuracy": float(
            baseline_accuracy
        ),
        "comparison_accuracy": float(
            comparison_accuracy
        ),
        "baseline_accuracy_percent": float(
            baseline_accuracy
            * 100
        ),
        "comparison_accuracy_percent": float(
            comparison_accuracy
            * 100
        ),
        "absolute_improvement_percent": float(
            improvement_percentage
        ),
        "improved_questions": (
            improved
        ),
        "worsened_questions": (
            worsened
        ),
        "unchanged_questions": (
            unchanged
        ),
        "statistic": float(
            statistic
        ),
        "p_value": float(
            p_value
        ),
        "significant": bool(
            significant
        ),
        "alpha": 0.05,
        "interpretation": (
            (
                f"{comparison} produced a statistically "
                f"significant difference compared with "
                f"{baseline}."
            )
            if significant
            else
            (
                f"No statistically significant difference "
                f"was detected between {baseline} and "
                f"{comparison}."
            )
        ),
    }


# ============================================================
# Compare every retrieval strategy with Full Schema
# ============================================================

def all_strategy_tests(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare:

        Full schema vs Top-1
        Full schema vs Top-3
        Full schema vs Top-5
    """

    comparisons = [
        "Top-1",
        "Top-3",
        "Top-5",
    ]

    rows = []

    for strategy in comparisons:
        result = paired_strategy_test(
            results=results,
            baseline="Full schema",
            comparison=strategy,
        )

        if (
            result.get("status")
            != "ok"
        ):
            rows.append(
                {
                    "baseline": (
                        "Full schema"
                    ),
                    "comparison": (
                        strategy
                    ),
                    "status": (
                        result.get(
                            "status"
                        )
                    ),
                    "message": (
                        result.get(
                            "message"
                        )
                    ),
                }
            )

            continue

        rows.append(
            {
                "baseline": (
                    "Full schema"
                ),

                "comparison": (
                    strategy
                ),

                "paired_questions": (
                    result[
                        "paired_questions"
                    ]
                ),

                "baseline_accuracy_percent": (
                    result[
                        "baseline_accuracy_percent"
                    ]
                ),

                "comparison_accuracy_percent": (
                    result[
                        "comparison_accuracy_percent"
                    ]
                ),

                "absolute_improvement_percent": (
                    result[
                        "absolute_improvement_percent"
                    ]
                ),

                "improved_questions": (
                    result[
                        "improved_questions"
                    ]
                ),

                "worsened_questions": (
                    result[
                        "worsened_questions"
                    ]
                ),

                "unchanged_questions": (
                    result[
                        "unchanged_questions"
                    ]
                ),

                "statistic": (
                    result[
                        "statistic"
                    ]
                ),

                "p_value": (
                    result[
                        "p_value"
                    ]
                ),

                "significant": (
                    result[
                        "significant"
                    ]
                ),

                "status": "ok",
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Efficiency comparison
# ============================================================

def strategy_efficiency_test(
    results: pd.DataFrame,
    metric: str = "prompt_tokens",
    baseline: str = "Full schema",
    comparison: str = "Top-5",
) -> dict:
    """
    Compare continuous efficiency metrics such as:

        prompt_tokens
        generation_latency
        execution_time
        schema_tables

    Uses a paired Wilcoxon test when question IDs are available.
    """

    if results.empty:
        return _empty_result(
            "No experiment results are available."
        )

    required = {
        "dataset",
        "question_id",
        "strategy",
        metric,
    }

    missing = (
        required
        - set(results.columns)
    )

    if missing:
        return _empty_result(
            "Missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    subset = results[
        results["strategy"].isin(
            [
                baseline,
                comparison,
            ]
        )
    ].copy()

    pivot = subset.pivot_table(
        index=[
            "dataset",
            "question_id",
        ],
        columns="strategy",
        values=metric,
        aggfunc="first",
    )

    if (
        baseline not in pivot.columns
        or comparison not in pivot.columns
    ):
        return _empty_result(
            "Both strategies are required."
        )

    paired = pivot[
        [
            baseline,
            comparison,
        ]
    ].dropna()

    if paired.empty:
        return _empty_result(
            "No paired observations are available."
        )

    baseline_values = (
        paired[baseline]
        .astype(float)
        .to_numpy()
    )

    comparison_values = (
        paired[comparison]
        .astype(float)
        .to_numpy()
    )

    difference = (
        comparison_values
        - baseline_values
    )

    baseline_mean = float(
        np.mean(
            baseline_values
        )
    )

    comparison_mean = float(
        np.mean(
            comparison_values
        )
    )

    if np.all(
        difference == 0
    ):
        statistic = 0.0
        p_value = 1.0

    else:
        statistic, p_value = (
            wilcoxon(
                comparison_values,
                baseline_values,
                zero_method="wilcox",
                alternative="two-sided",
            )
        )

    significant = (
        p_value < 0.05
    )

    if baseline_mean != 0:
        relative_change = (
            (
                comparison_mean
                - baseline_mean
            )
            / baseline_mean
        ) * 100

    else:
        relative_change = np.nan

    return {
        "status": "ok",
        "test": (
            "Wilcoxon signed-rank"
        ),
        "metric": metric,
        "baseline": baseline,
        "comparison": comparison,
        "paired_questions": int(
            len(paired)
        ),
        "baseline_mean": (
            baseline_mean
        ),
        "comparison_mean": (
            comparison_mean
        ),
        "absolute_change": float(
            comparison_mean
            - baseline_mean
        ),
        "relative_change_percent": (
            float(relative_change)
            if not np.isnan(
                relative_change
            )
            else None
        ),
        "statistic": float(
            statistic
        ),
        "p_value": float(
            p_value
        ),
        "significant": bool(
            significant
        ),
        "alpha": 0.05,
    }


# ============================================================
# Complete statistical report
# ============================================================

def complete_statistical_analysis(
    results: pd.DataFrame,
) -> dict:
    """
    Produce the statistical outputs required by the research app.
    """

    dataset_test = (
        dataset_comparison_test(
            results
        )
    )

    strategy_tests = (
        all_strategy_tests(
            results
        )
    )

    efficiency = {}

    for metric in [
        "prompt_tokens",
        "generation_latency",
        "schema_tables",
    ]:
        if metric in results.columns:
            efficiency[
                metric
            ] = strategy_efficiency_test(
                results=results,
                metric=metric,
                baseline="Full schema",
                comparison="Top-5",
            )

    return {
        "dataset_comparison": (
            dataset_test
        ),
        "strategy_comparisons": (
            strategy_tests
        ),
        "efficiency_comparisons": (
            efficiency
        ),
    }
