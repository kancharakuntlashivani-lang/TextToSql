from __future__ import annotations
import re
import time
from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd
from src import config
from src.provider import generate_sql, ProviderError

STRATEGIES = ['Full schema', 'Top-1', 'Top-3', 'Top-5']


def normalise_sql(sql: str) -> str:
    return re.sub(r'\s+', ' ', str(sql or '').strip().lower().rstrip(';'))


def simple_schema_context(gold_sql: str, strategy: str) -> str:
    # Portable fallback for the hosted comparison app. Real database schema extraction
    # can replace this function when local DB files are supplied.
    tables = re.findall(r'\b(?:from|join)\s+[`"\[]?([\w.]+)', gold_sql, flags=re.I)
    tables = list(dict.fromkeys(tables)) or ['unknown_table']
    if strategy == 'Top-1': tables = tables[:1]
    elif strategy == 'Top-3': tables = tables[:3]
    elif strategy == 'Top-5': tables = tables[:5]
    return '\n'.join(f'TABLE {t}' for t in tables)


def run_experiment(frame: pd.DataFrame, limit: int, provider: str = 'OpenAI',
                   progress_callback: Callable[[int, int, str], None] | None = None) -> dict:
    if frame.empty:
        raise ValueError('No dataset records are available.')
    sample = frame.head(limit).copy()
    rows = []
    total = len(sample) * len(STRATEGIES)
    done = 0
    for record in sample.to_dict('records'):
        for strategy in STRATEGIES:
            done += 1
            if progress_callback:
                progress_callback(done, total, f"{record['dataset']} · {strategy}")
            context = simple_schema_context(record['gold_sql'], strategy)
            try:
                generated = generate_sql(record['question'], context, provider)
                error = ''
            except ProviderError as exc:
                generated = {'generated_sql': '', 'generation_latency': np.nan,
                             'prompt_tokens': 0, 'completion_tokens': 0}
                error = str(exc)
            generated_sql = generated['generated_sql']
            rows.append({
                'dataset': record['dataset'],
                'question_id': record['question_id'],
                'db_id': record['db_id'],
                'difficulty': record['difficulty'],
                'question': record['question'],
                'gold_sql': record['gold_sql'],
                'strategy': strategy,
                'generated_sql': generated_sql,
                'exact_match': int(bool(generated_sql) and normalise_sql(generated_sql) == normalise_sql(record['gold_sql'])),
                'success': int(bool(generated_sql)),
                'generation_latency': generated['generation_latency'],
                'prompt_tokens': generated['prompt_tokens'],
                'question_words': record['question_words'],
                'sql_words': record['sql_words'],
                'schema_tables': context.count('TABLE '),
                'error': error,
            })
    results = pd.DataFrame(rows)
    path = config.OUTPUT_DIR / 'experiment_results.csv'
    results.to_csv(path, index=False)
    summary = (results.groupby(['dataset', 'strategy'], as_index=False)
               .agg(questions=('question_id', 'count'),
                    exact_match=('exact_match', 'mean'),
                    success_rate=('success', 'mean'),
                    generation_latency=('generation_latency', 'mean'),
                    prompt_tokens=('prompt_tokens', 'mean')))
    summary.to_csv(config.OUTPUT_DIR / 'dataset_strategy_summary.csv', index=False)
    return {'results': results, 'summary': summary, 'path': path}
