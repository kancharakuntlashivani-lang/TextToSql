from __future__ import annotations
import pandas as pd
from scipy.stats import chi2_contingency


def dataset_comparison_test(results: pd.DataFrame) -> dict:
    if results.empty or results['dataset'].nunique() < 2:
        return {'status': 'not_available', 'message': 'Two datasets are required.'}
    table = pd.crosstab(results['dataset'], results['exact_match'])
    if table.shape[1] < 2:
        return {'status': 'not_available', 'message': 'Both outcome classes are required.'}
    statistic, p_value, dof, _ = chi2_contingency(table)
    return {'status': 'ok', 'test': 'Chi-square', 'statistic': float(statistic),
            'p_value': float(p_value), 'degrees_of_freedom': int(dof)}
