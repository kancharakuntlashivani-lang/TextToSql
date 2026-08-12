from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    cross_val_score,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)

from src import config


# ============================================================
# Features and target
# ============================================================

FEATURES_NUMERIC = [
    "question_words",
    "sql_words",
    "schema_tables",
    "prompt_tokens",
    "generation_latency",
]

FEATURES_CATEGORY = [
    "dataset",
    "strategy",
    "difficulty",
]

TARGET = "exact_match"

GROUP_COLUMN = "question"


# ============================================================
# Helpers
# ============================================================

def safe_roc_auc(
    y_true: pd.Series,
    probability: np.ndarray,
) -> float:
    """Calculate ROC-AUC safely."""

    if pd.Series(y_true).nunique() < 2:
        return float("nan")

    try:
        return float(
            roc_auc_score(
                y_true,
                probability,
            )
        )
    except Exception:
        return float("nan")


def build_preprocessor() -> ColumnTransformer:
    """Create numeric and categorical preprocessing."""

    numeric = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scale",
                StandardScaler(),
            ),
        ]
    )

    categorical = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric,
                FEATURES_NUMERIC,
            ),
            (
                "categorical",
                categorical,
                FEATURES_CATEGORY,
            ),
        ]
    )


def build_models() -> dict:
    """Create the three classifiers used in the comparison."""

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),

        "Gradient Boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=150,
            max_leaf_nodes=15,
            min_samples_leaf=10,
            l2_regularization=1.0,
            random_state=42,
        ),
    }


# ============================================================
# Main training function
# ============================================================

def train_models(
    results: pd.DataFrame,
) -> dict:
    """
    Train and compare classifiers that predict whether generated
    SQL will exactly match the benchmark SQL.

    The split is grouped by natural-language question to prevent
    the same question appearing in both training and testing data.
    """

    required = set(
        FEATURES_NUMERIC
        + FEATURES_CATEGORY
        + [
            TARGET,
            "generated_sql",
            GROUP_COLUMN,
        ]
    )

    missing = required - set(
        results.columns
    )

    if missing:
        raise ValueError(
            "Missing experiment columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    # --------------------------------------------------------
    # Clean data
    # --------------------------------------------------------

    data = results.copy()

    data = data[
        data["generated_sql"]
        .astype(str)
        .str.strip()
        .str.len()
        > 0
    ].copy()

    data = data[
        data[TARGET].notna()
    ].copy()

    data[TARGET] = (
        data[TARGET]
        .astype(int)
    )

    data[GROUP_COLUMN] = (
        data[GROUP_COLUMN]
        .astype(str)
        .str.strip()
    )

    if len(data) < 30:
        raise ValueError(
            "At least 30 completed experiment rows "
            "are required."
        )

    if data[TARGET].nunique() < 2:
        raise ValueError(
            "Both correct and incorrect SQL outcomes "
            "are required."
        )

    unique_questions = (
        data[GROUP_COLUMN]
        .nunique()
    )

    if unique_questions < 5:
        raise ValueError(
            "At least five unique questions are required "
            "for grouped ML evaluation."
        )

    # --------------------------------------------------------
    # X, y and groups
    # --------------------------------------------------------

    X = data[
        FEATURES_NUMERIC
        + FEATURES_CATEGORY
    ].copy()

    y = data[TARGET].copy()

    groups = (
        data[GROUP_COLUMN]
        .astype(str)
    )

    # --------------------------------------------------------
    # Grouped train / test split
    # --------------------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=42,
    )

    train_index, test_index = next(
        splitter.split(
            X,
            y,
            groups=groups,
        )
    )

    X_train = X.iloc[
        train_index
    ].copy()

    X_test = X.iloc[
        test_index
    ].copy()

    y_train = y.iloc[
        train_index
    ].copy()

    y_test = y.iloc[
        test_index
    ].copy()

    train_groups = groups.iloc[
        train_index
    ]

    test_groups = groups.iloc[
        test_index
    ]

    # --------------------------------------------------------
    # Leakage check
    # --------------------------------------------------------

    train_questions = set(
        train_groups
    )

    test_questions = set(
        test_groups
    )

    overlap = (
        train_questions
        & test_questions
    )

    if overlap:
        raise RuntimeError(
            "Question leakage detected between "
            "training and testing datasets."
        )

    # --------------------------------------------------------
    # Cross-validation
    # --------------------------------------------------------

    cv_folds = min(
        5,
        len(train_questions),
    )

    if cv_folds < 2:
        raise ValueError(
            "Not enough unique questions for "
            "cross-validation."
        )

    cv = StratifiedGroupKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=42,
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    models = build_models()

    metric_rows = []
    prediction_rows = []

    for name, estimator in models.items():

        pipeline = Pipeline(
            steps=[
                (
                    "prep",
                    build_preprocessor(),
                ),
                (
                    "model",
                    estimator,
                ),
            ]
        )

        # ----------------------------------------------------
        # Cross-validation accuracy
        # ----------------------------------------------------

        try:
            cv_scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                groups=train_groups,
                cv=cv,
                scoring="accuracy",
            )

            cv_accuracy_mean = float(
                np.mean(cv_scores)
            )

            cv_accuracy_std = float(
                np.std(cv_scores)
            )

        except Exception:
            cv_accuracy_mean = float("nan")
            cv_accuracy_std = float("nan")

        # ----------------------------------------------------
        # Train final model
        # ----------------------------------------------------

        pipeline.fit(
            X_train,
            y_train,
        )

        train_pred = pipeline.predict(
            X_train
        )

        test_pred = pipeline.predict(
            X_test
        )

        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        if hasattr(
            pipeline,
            "predict_proba",
        ):
            probability = (
                pipeline
                .predict_proba(
                    X_test
                )[:, 1]
            )

        else:
            probability = (
                test_pred.astype(float)
            )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        train_accuracy = accuracy_score(
            y_train,
            train_pred,
        )

        test_accuracy = accuracy_score(
            y_test,
            test_pred,
        )

        precision = precision_score(
            y_test,
            test_pred,
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            test_pred,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            test_pred,
            zero_division=0,
        )

        roc_auc = safe_roc_auc(
            y_test,
            probability,
        )

        overfitting_gap = (
            train_accuracy
            - test_accuracy
        )

        metric_rows.append(
            {
                "model": name,
                "train_accuracy": float(
                    train_accuracy
                ),
                "test_accuracy": float(
                    test_accuracy
                ),
                "precision": float(
                    precision
                ),
                "recall": float(
                    recall
                ),
                "f1": float(
                    f1
                ),
                "roc_auc": float(
                    roc_auc
                ),
                "cv_accuracy_mean": float(
                    cv_accuracy_mean
                ),
                "cv_accuracy_std": float(
                    cv_accuracy_std
                ),
                "overfitting_gap": float(
                    overfitting_gap
                ),
                "train_rows": int(
                    len(X_train)
                ),
                "test_rows": int(
                    len(X_test)
                ),
                "unique_train_questions": int(
                    len(train_questions)
                ),
                "unique_test_questions": int(
                    len(test_questions)
                ),
            }
        )

        # ----------------------------------------------------
        # Prediction details
        # ----------------------------------------------------

        for (
            idx,
            actual,
            predicted,
            prob,
        ) in zip(
            X_test.index,
            y_test,
            test_pred,
            probability,
        ):

            prediction_rows.append(
                {
                    "row_id": int(idx),
                    "model": name,
                    "actual": int(actual),
                    "prediction": int(
                        predicted
                    ),
                    "success_probability": float(
                        prob
                    ),
                }
            )

    # --------------------------------------------------------
    # Final DataFrames
    # --------------------------------------------------------

    metrics = pd.DataFrame(
        metric_rows
    )

    metrics = metrics.sort_values(
        [
            "cv_accuracy_mean",
            "f1",
        ],
        ascending=False,
    ).reset_index(
        drop=True
    )

    predictions = pd.DataFrame(
        prediction_rows
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    config.OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics.to_csv(
        config.OUTPUT_DIR
        / "ml_model_metrics.csv",
        index=False,
    )

    predictions.to_csv(
        config.OUTPUT_DIR
        / "ml_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Evaluation summary
    # --------------------------------------------------------

    evaluation_info = {
        "total_rows": int(
            len(data)
        ),
        "training_rows": int(
            len(X_train)
        ),
        "testing_rows": int(
            len(X_test)
        ),
        "unique_questions": int(
            unique_questions
        ),
        "training_questions": int(
            len(train_questions)
        ),
        "testing_questions": int(
            len(test_questions)
        ),
        "question_overlap": int(
            len(overlap)
        ),
        "cross_validation_folds": int(
            cv_folds
        ),
    }

    return {
        "metrics": metrics,
        "predictions": predictions,
        "evaluation_info": evaluation_info,
    }
