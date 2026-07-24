from __future__ import annotations
import sys
from src import config
from src.datasets_manager import DATASETS, cache_path

print('Python:', sys.version.split()[0])
print('Project:', config.ROOT_DIR)
print('OpenAI key:', 'configured' if config.OPENAI_API_KEY else 'not configured')
for name in DATASETS:
    path = cache_path(name)
    print(f'{name}:', 'cached' if path.exists() else 'will download automatically')
print('Output directory:', config.OUTPUT_DIR)
