from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable
import pandas as pd
from src import config

DATASETS = {
    'BIRD Mini-Dev': {
        'repo': 'birdsql/bird_mini_dev',
        'split': 'mini_dev_sqlite',
        'sql_field': 'SQL',
        'description': 'Realistic database-grounded Text-to-SQL benchmark.',
    },
    'Spider': {
        'repo': 'xlangai/spider',
        'split': 'validation',
        'sql_field': 'query',
        'description': 'Complex cross-domain Text-to-SQL benchmark.',
    },
}


def _normalise(dataset_name: str, rows: Iterable[dict]) -> pd.DataFrame:
    spec = DATASETS[dataset_name]
    normalised = []
    for idx, row in enumerate(rows):
        normalised.append({
            'dataset': dataset_name,
            'question_id': row.get('question_id', idx),
            'db_id': str(row.get('db_id', '')).strip(),
            'question': str(row.get('question', '')).strip(),
            'gold_sql': str(row.get(spec['sql_field'], '')).strip(),
            'evidence': str(row.get('evidence', '') or '').strip(),
            'difficulty': str(row.get('difficulty', 'unknown') or 'unknown').strip(),
        })
    frame = pd.DataFrame(normalised)
    if not frame.empty:
        frame = frame[frame['question'].ne('') & frame['gold_sql'].ne('')].copy()
        frame['question_words'] = frame['question'].str.split().str.len()
        frame['sql_words'] = frame['gold_sql'].str.split().str.len()
    return frame.reset_index(drop=True)


def cache_path(dataset_name: str) -> Path:
    safe = dataset_name.lower().replace(' ', '_').replace('-', '_')
    return config.DATA_DIR / f'{safe}.jsonl'


def download_dataset(dataset_name: str, force: bool = False) -> pd.DataFrame:
    if dataset_name not in DATASETS:
        raise ValueError(f'Unsupported dataset: {dataset_name}')
    path = cache_path(dataset_name)
    if path.exists() and not force:
        return pd.read_json(path, lines=True)
    spec = DATASETS[dataset_name]
    from datasets import load_dataset
    dataset = load_dataset(spec['repo'], split=spec['split'])
    frame = _normalise(dataset_name, dataset)
    frame.to_json(path, orient='records', lines=True)
    return frame


def load_dataset_frame(dataset_name: str, auto_download: bool = True) -> pd.DataFrame:
    path = cache_path(dataset_name)
    if path.exists():
        return pd.read_json(path, lines=True)
    if auto_download:
        return download_dataset(dataset_name)
    return pd.DataFrame()


def load_all(auto_download: bool = True) -> pd.DataFrame:
    frames = []
    for name in DATASETS:
        try:
            frame = load_dataset_frame(name, auto_download=auto_download)
            if not frame.empty:
                frames.append(frame)
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def dataset_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    return (frame.groupby('dataset', as_index=False)
            .agg(questions=('question_id', 'count'),
                 databases=('db_id', 'nunique'),
                 average_question_words=('question_words', 'mean'),
                 average_sql_words=('sql_words', 'mean')))
