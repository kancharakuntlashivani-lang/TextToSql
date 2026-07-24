from __future__ import annotations
import time
from src import config

class ProviderError(RuntimeError):
    pass


def generate_sql(question: str, schema_context: str, provider: str = 'OpenAI') -> dict:
    if provider != 'OpenAI':
        raise ProviderError('Live SQL generation requires the OpenAI provider.')
    if not config.OPENAI_API_KEY:
        raise ProviderError('OpenAI API key is not configured. Add OPENAI_API_KEY in Render or .env.')

    prompt = f'''You are a Text-to-SQL system. Return one read-only SQL query only.
Use only the schema supplied below. Do not include markdown or explanation.

SCHEMA:
{schema_context}

QUESTION:
{question}

SQL:'''
    started = time.perf_counter()
    try:
        from openai import OpenAI, RateLimitError, AuthenticationError, APIConnectionError
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            temperature=0,
            messages=[{'role': 'user', 'content': prompt}],
        )
        text = response.choices[0].message.content or ''
        sql = text.replace('```sql', '').replace('```', '').strip().rstrip(';')
        usage = response.usage
        return {
            'generated_sql': sql,
            'generation_latency': time.perf_counter() - started,
            'prompt_tokens': getattr(usage, 'prompt_tokens', 0) or 0,
            'completion_tokens': getattr(usage, 'completion_tokens', 0) or 0,
        }
    except RateLimitError as exc:
        raise ProviderError('OpenAI quota is unavailable. Add API billing or use saved experiment results.') from exc
    except AuthenticationError as exc:
        raise ProviderError('The OpenAI API key is invalid.') from exc
    except APIConnectionError as exc:
        raise ProviderError('The application could not reach the OpenAI API.') from exc
