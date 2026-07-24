from __future__ import annotations
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from src import config

FEATURES_NUMERIC = ['question_words', 'sql_words', 'schema_tables', 'prompt_tokens', 'generation_latency']
FEATURES_CATEGORY = ['dataset', 'strategy', 'difficulty']
TARGET = 'exact_match'


def train_models(results: pd.DataFrame) -> dict:
    required = set(FEATURES_NUMERIC + FEATURES_CATEGORY + [TARGET])
    missing = required - set(results.columns)
    if missing:
        raise ValueError('Missing experiment columns: ' + ', '.join(sorted(missing)))
    data = results.copy()
    data = data[data['generated_sql'].astype(str).str.len() > 0]
    if len(data) < 30 or data[TARGET].nunique() < 2:
        raise ValueError('At least 30 completed rows with both successful and unsuccessful outcomes are required.')

    X = data[FEATURES_NUMERIC + FEATURES_CATEGORY]
    y = data[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.25, random_state=42, stratify=y
    )

    numeric = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
    categorical = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer([('num', numeric, FEATURES_NUMERIC), ('cat', categorical, FEATURES_CATEGORY)])

    models = {
        'Logistic Regression': LogisticRegression(max_iter=2000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=250, random_state=42, class_weight='balanced'),
        'Gradient Boosting': HistGradientBoostingClassifier(random_state=42),
    }

    metric_rows, prediction_rows = [], []
    for name, estimator in models.items():
        pipeline = Pipeline([('prep', preprocessor), ('model', estimator)])
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        probability = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, 'predict_proba') else pred.astype(float)
        metric_rows.append({
            'model': name,
            'accuracy': accuracy_score(y_test, pred),
            'precision': precision_score(y_test, pred, zero_division=0),
            'recall': recall_score(y_test, pred, zero_division=0),
            'f1': f1_score(y_test, pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, probability),
        })
        for idx, actual, predicted, prob in zip(X_test.index, y_test, pred, probability):
            prediction_rows.append({'row_id': int(idx), 'model': name, 'actual': int(actual),
                                    'prediction': int(predicted), 'success_probability': float(prob)})

    metrics = pd.DataFrame(metric_rows).sort_values('f1', ascending=False)
    predictions = pd.DataFrame(prediction_rows)
    metrics.to_csv(config.OUTPUT_DIR / 'ml_model_metrics.csv', index=False)
    predictions.to_csv(config.OUTPUT_DIR / 'ml_predictions.csv', index=False)
    return {'metrics': metrics, 'predictions': predictions}
