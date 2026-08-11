
I’ll prepare a clean professor-ready research proposal report for Pravallika, with Overleaf LaTeX source and a compiled PDF. I’ll include the project aim, research gaps, methodology, architecture, tools, timeline, expected outcomes, and references.

I found recent support for this topic from current Text-to-SQL/LLM research and the emerging “agentic analytics” direction. I’m turning that into a simple, direct report rather than overcomplicating it.

Thought for 2m 12s
Done ✅

Download PDF

Download Overleaf ZIP


Pravallika_Data_Analyst_Research_Proposal.pdf
PDF


I’ll convert the full 20 questions and answers into a clean PDF with simple formatting, ready to submit or use for viva preparation.

Thought for 1m 1s
Done ✅

Download the 20 Questions & Answers PDF


Shivani_Data_Analyst_Project_QA.pdf
PDF


I’ll make the PDF as a focused research-gap papers document: table first, then short notes for each paper, direct links, and how each supports Shivani’s project.

Thought for 1m 34s
Done ✅

Download the 10 Research Gap Papers PDF

I included recent papers on Text-to-SQL, RAG, agentic analytics, XAI, and AI data analyst systems. 


Shivani_Research_Gap_Papers_Data_Analyst_Project.pdf
PDF


Wed, Jul 8 at 7:02 PM
3EEC8C40-6F63-478C-AA36-D3225FF61388.jpeg
4CEDB8BC-ED80-4030-96DF-E6895912798F.jpeg

Thought for 2m 22s
Done ✅

Download PDF

Download Overleaf ZIP


Shivani_Updated_Dissertation_Report.pdf
PDF


Thought for 3m 30s
Done nicely ✅ No email content included.

Download final updated PDF

Download final Overleaf ZIP


Shivani_Final_Updated_Dissertation_Report.pdf
PDF


Thought for 1m 33s
Done ✅ Only the Table of Contents plan.

Download PDF

Download Overleaf ZIP


Shivani_Table_of_Contents_Report_Plan.pdf
PDF






Shivani_Text2SQL_RAG_Implementation(1).zip
Zip Archive
extract and read all this files and give me code to do above bird dataset




Shivani_BIRD_Streamlit_Text2SQL_Full_Implementation(1).zip
Zip Archive
Will you please unzip this and check all files Its worked i dont have any output or UI to check if possible let add nice UI to this project by replace the whole zip


Pasted text(3).txt
Document
Give me terminal commands to do thus - 

Fri, Jul 24 at 1:17 PM

Shivani_BIRD_Text2SQL_Research_Studio_Enhanced(1).zip
Zip Archive
please ---- I asked you simple and nice UI i dont want more graphis just simple UI like cloud i dont want deploy button etc etc ---- I want nice and simple userfriendly UI please - please take your own time  dont give me errors again pls

I dont want all this - I want simple and nice please feel like real application pls -----------Please take your own time and think like me - 


Shivani_BIRD_Streamlit_Text2SQL.zip
Zip Archive
One more thing -- i want to use one more dataset as well and same pattern like pur description and one more thing  I want both datasets comaparion and also apply 3 ml models and predictions of results --- How was this i want to impress my professor and also i need to deploy in renders please take your own time and  do it for me pleasee

lets feed the datasets please - 


Pasted text(7).txt
Document



Pasted code.py
Python


Pasted code(1).py
Python

944df476-f9c7-433e-b999-7f8659e3db3a.png
5f751dee-0864-4508-b636-cb286784ece4.png


Pasted text(8).txt
Document
next 

what are env required tell me pls- DB_HOST=localhost
DB_PORT=5432
DB_NAME=text2sql_rag
DB_USER=postgres
DB_PASSWORD=Shivani@11
OPENAI_API_KEY=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
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OUTPUT_DIR=outputs


dissertation_template (1)(1)(1).zip
Zip Archive
Now, please take your own time and do  this report with all tables , images png and all as per the format i want 80 pages of report and give me full overleaf .zip back to and replace Girffith with DBS please student name as well please take your own time give me this report with any issues pls .

Worked for 54s
Completed the DBS-formatted dissertation package.

Files
Download the complete Overleaf ZIP

Download the compiled 91-page PDF preview

Included
91 compiled pages

Student name: Shivani Kancharakuntla

Institution changed from Griffith College to Dublin Business School

Dissertation title: Improving Natural Language-to-SQL Generation Using Relevant Schema Retrieval

DBS-style title page and original DBS wordmark

Abstract, declaration, acknowledgements and research objectives

Eight full chapters

BIRD and Spider benchmark discussion

Full-schema baseline and TF–IDF schema-retrieval methodology

PostgreSQL, Streamlit, OpenAI and Render implementation

Statistical evaluation using McNemar, paired tests and effect sizes

Tables, algorithms, code samples and original PNG diagrams

Installation, deployment, experiment-record and submission appendices

Complete BibLaTeX reference file

Successfully compiled with references, contents, figures and tables

The student ID is left as a clearly marked placeholder because it was not provided. Final numerical result fields are also marked for insertion from the verified experiment output, so no research results have been fabricated. 


main.pdf
PDF



Pasted text(9).txt
Document

Sun, Aug 2 at 5:05 PM
128488AB-6C8F-48BA-915F-76D5B11BC73B.jpeg
B27F5345-10D1-45B2-B374-6E97E69DE00A.jpeg

IMG_4E716AE1-BF7B-48DF-B955-D6C67D3EC13B.jpeg
IMG_EF3A4C4C-9409-43C0-9808-EE877F25685C.jpeg


TextToSql-main.zip
Zip Archive

075eff7b-b1ba-48df-8392-4f547bd9eca1.png
e2106e6b-5021-4528-9262-2f9a1212b28e.png
I need to ask user to enter his text and get sql query we need to show it true query are not as well in query please


Pasted text(12).txt
Document


Pasted text(13).txt
Document
update fully and give me - 


Pasted text(14).txt
Document



Pasted text(21).txt
Document


Pasted text(22).txt
Document



TextToSql-main (2).zip
Zip Archive
unzip this and check 

Now we found the real Text-to-SQL issue. Your pipeline is working, but the generated SQL is wrong for two reasons.

The model generated:

SELECT COUNT(*)
FROM schools s
JOIN satscores sa ON s.CDSCode = sa.cds
WHERE sa.AvgScrMath > 400 AND s.Virtual = 'Y';
But PostgreSQL says:

column s.cdscode does not exist
That happens because PostgreSQL folds unquoted identifiers to lowercase. Your migrated column appears to be case-sensitive, so CDSCode must be quoted:

s."CDSCode"
Also, the benchmark answer expects:

T2.Virtual = 'F'
not:

s.Virtual = 'Y'
So even after fixing the column error, that generated SQL would still be logically incorrect.

What we need to update
The prompt currently gives OpenAI only table/column names. We should make it much stricter:

Use column names exactly as provided.
If a table or column contains uppercase letters, quote it using PostgreSQL double quotes.
Do not change coded values or infer their meaning.
For example, the model should produce:

SELECT COUNT(DISTINCT s."School")
FROM satscores sa
JOIN schools s
    ON sa.cds = s."CDSCode"
WHERE s."Virtual" = 'F'
  AND sa."AvgScrMath" > 400;
Important research point
This result is actually useful for your dissertation:

Generated SQL executed?       FALSE
Execution accuracy?           FALSE
Exact match?                  FALSE
That is a legitimate failure case. We should not force every benchmark answer to become TRUE. Your research needs correct and incorrect cases so you can measure accuracy.

What we should improve is the schema representation/prompt, so failures happen because of actual reasoning mistakes rather than avoidable PostgreSQL case-sensitivity errors.

The next file to update should be src/core.py, specifically full_schema_context(), Retriever.retrieve(), and generate_sql(). I can give you the exact replacement code for those three sections so PostgreSQL identifiers are handled correctly.



Pasted text(20260811-104212).txt
Document

I checked the core.py you sent. The problem is exactly in the schema context and SQL-generation prompt: the schema currently sends raw column names, while the prompt only says to use supplied tables/columns. 


Replace these parts.

1. Add this helper before full_schema_context()
def quote_identifier(name: str) -> str:
    """
    Return an exact PostgreSQL quoted identifier.

    This is important because PostgreSQL converts unquoted identifiers
    to lowercase. BIRD/Spider can contain mixed-case column names such
    as CDSCode and AvgScrMath.
    """
    return '"' + str(name).replace('"', '""') + '"'


def format_table_for_prompt(
    table_name: str,
    columns: list[dict[str, str]],
) -> str:
    """Format a table for the LLM using exact PostgreSQL identifiers."""

    lines = [
        f"Table: {quote_identifier(table_name)}",
        "Columns:",
    ]

    for column in columns:
        lines.append(
            f"- {quote_identifier(column['name'])} "
            f"({column['type']})"
        )

    return "\n".join(lines)
2. Replace full_schema_context()
def full_schema_context(
    db_id: str,
    dataset: str | None = None,
) -> tuple[str, list[str]]:
    """Format the complete PostgreSQL schema for the LLM."""

    tables = schema_metadata(
        db_id=db_id,
        dataset=dataset,
    )

    sections: list[str] = []

    for table_name, columns in tables.items():
        sections.append(
            format_table_for_prompt(
                table_name,
                columns,
            )
        )

    return (
        "\n\n".join(sections).strip(),
        list(tables.keys()),
    )
3. Inside Retriever.retrieve(), replace lines 428–445
Your current retriever is also sending raw identifiers. 


Replace:

        lines: list[str] = []

        for table_name in selected_tables:
            columns = tables[table_name]

            lines.append(f"Table: {table_name}")
            lines.append(
                "Columns: "
                + ", ".join(
                    f"{column['name']} {column['type']}"
                    for column in columns
                )
            )
            lines.append("")

        return (
            "\n".join(lines).strip(),
            selected_tables,
        )
with:

        sections: list[str] = []

        for table_name in selected_tables:
            sections.append(
                format_table_for_prompt(
                    table_name,
                    tables[table_name],
                )
            )

        return (
            "\n\n".join(sections).strip(),
            selected_tables,
        )
4. Replace generate_sql() completely
def generate_sql(
    question: str,
    schema_context: str,
) -> tuple[str, float, int, int, bool]:
    """Generate exactly one read-only PostgreSQL query."""

    prompt = f"""
You are a PostgreSQL Text-to-SQL system.

DATABASE SCHEMA
---------------
{schema_context}

USER QUESTION
-------------
{question}

STRICT RULES
------------
1. Return exactly one PostgreSQL SELECT query.
2. WITH is allowed only when the final statement is SELECT.
3. Use ONLY tables and columns shown in DATABASE SCHEMA.
4. Preserve every table and column name exactly as shown.
5. PostgreSQL identifiers shown inside double quotes MUST remain
   double-quoted in the generated SQL.
6. Never change the spelling or capitalization of an identifier.
7. Never invent a table or column.
8. Use appropriate JOIN conditions based only on available schema.
9. Do not guess database values unless they can be determined from
   the question or supplied context.
10. Do not translate natural-language concepts into arbitrary coded
    values unless that mapping is explicitly known.
11. Return SQL only.
12. Do not return markdown.
13. Do not provide explanations or comments.

Generate the PostgreSQL query now.
""".strip()

    started_at = time.perf_counter()

    if not OPENAI_API_KEY:
        demo_sql = "SELECT 1 AS demo_mode;"

        return (
            demo_sql,
            time.perf_counter() - started_at,
            count_tokens(prompt),
            count_tokens(demo_sql),
            True,
        )

    client = OpenAI(
        api_key=OPENAI_API_KEY
    )

    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You generate one read-only PostgreSQL query. "
                    "Use only the supplied schema. "
                    "Preserve and quote PostgreSQL identifiers exactly "
                    "as provided. Return SQL only."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    generated_sql = clean_sql(
        response.choices[0].message.content or ""
    )

    usage = response.usage

    prompt_tokens = (
        int(usage.prompt_tokens)
        if usage is not None
        else count_tokens(prompt)
    )

    completion_tokens = (
        int(usage.completion_tokens)
        if usage is not None
        else count_tokens(generated_sql)
    )

    return (
        generated_sql,
        time.perf_counter() - started_at,
        prompt_tokens,
        completion_tokens,
        False,
    )
One more important issue
For the BIRD example, fixing quoting solves:

column s.cdscode does not exist
but it doesn't necessarily solve the semantic issue:

exclusively virtual → Virtual = 'F'
The benchmark's gold SQL uses Virtual = 'F'. Your current load_questions() already loads BIRD's evidence field, 

 but run_strategy() never passes that evidence to generate_sql().

That is important for BIRD. We should update the pipeline next so:

Question
   ↓
BIRD evidence/hint
   ↓
Relevant schema Top-5
   ↓
LLM
   ↓
Generated SQL
   ↓
Execute generated SQL
   ↓
Execute ground-truth SQL
   ↓
Compare results
   ↓
TRUE / FALSE + accuracy
That will give you a much more legitimate BIRD evaluation rather than trying to teach the model benchmark answers manually.


give me full code please

Yes — I updated the full src/core.py for you.

It now includes:

exact PostgreSQL identifier quoting, so columns like "CDSCode" and "AvgScrMath" work correctly;

improved schema formatting for both Full Schema and Top-K retrieval;

a stricter SQL-generation prompt;

optional BIRD benchmark evidence support;

the existing execution, exact-match, and execution-accuracy logic remains intact. Your original file already had the core evaluation flow; this update strengthens the schema/prompt handling around it. 


Download the full updated core.py

Replace:

src/core.py
with this file, then commit:

git add src/core.py
git commit -m "Improve PostgreSQL schema quoting and Text-to-SQL generation"
git push origin main
One important next change: your app.py should pass the benchmark evidence value into run_strategy() so BIRD hints are actually used. We should update that file next. 


core_full_updated.py
Code





core_full_updated.py


from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import tiktoken
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src import config


# ============================================================
# Configuration
# ============================================================

SQLALCHEMY_URL = getattr(
    config,
    "SQLALCHEMY_URL",
    getattr(config, "DATABASE_URL", ""),
)

if not SQLALCHEMY_URL:
    raise RuntimeError(
        "PostgreSQL connection URL is missing. Define SQLALCHEMY_URL "
        "or DATABASE_URL in src/config.py."
    )

OPENAI_API_KEY = getattr(config, "OPENAI_API_KEY", "")
LLM_MODEL = getattr(
    config,
    "LLM_MODEL",
    getattr(config, "OPENAI_MODEL", "gpt-4o-mini"),
)
MAX_RESULT_ROWS = int(getattr(config, "MAX_RESULT_ROWS", 200))

engine: Engine = create_engine(
    SQLALCHEMY_URL,
    pool_pre_ping=True,
    future=True,
)

BLOCKED_SQL = re.compile(
    r"\b("
    r"insert|update|delete|drop|alter|create|truncate|grant|revoke|"
    r"copy|call|do|merge|vacuum|analyze|refresh|reindex|cluster"
    r")\b",
    re.IGNORECASE,
)


# ============================================================
# Dataset and PostgreSQL schema helpers
# ============================================================

def safe_identifier(name: str) -> str:
    """Convert a value into a safe PostgreSQL-style identifier."""

    value = re.sub(
        r"[^a-zA-Z0-9_]+",
        "_",
        str(name),
    ).strip("_").lower()

    if not value:
        value = "database"

    if value[0].isdigit():
        value = f"db_{value}"

    return value[:63]


def normalise_dataset(dataset: str | None) -> str:
    """Validate and return a supported dataset name."""

    value = str(dataset or "bird").strip().lower()

    if value not in {"bird", "spider"}:
        raise ValueError(
            f"Unsupported dataset '{dataset}'. Use 'bird' or 'spider'."
        )

    return value


def split_dataset_and_db_id(
    dataset: str | None,
    db_id: str,
) -> tuple[str, str]:
    """
    Resolve the dataset and clean database ID.

    Supported inputs:
        dataset='bird', db_id='financial'
        dataset='bird', db_id='bird_financial'
        dataset=None, db_id='bird_financial'
        dataset=None, db_id='spider_concert_singer'
    """

    raw_db_id = str(db_id or "").strip()

    if not raw_db_id:
        raise ValueError("db_id cannot be empty.")

    lowered = raw_db_id.lower()

    if dataset is None:
        if lowered.startswith("bird_"):
            return "bird", raw_db_id[5:]

        if lowered.startswith("spider_"):
            return "spider", raw_db_id[7:]

        # Backward-compatible default for the original BIRD-only project.
        return "bird", raw_db_id

    dataset_key = normalise_dataset(dataset)
    prefix = f"{dataset_key}_"

    if lowered.startswith(prefix):
        raw_db_id = raw_db_id[len(prefix):]

    return dataset_key, raw_db_id


def postgres_schema_name(
    dataset: str | None,
    db_id: str,
) -> str:
    """Return the migrated PostgreSQL schema name."""

    dataset_key, clean_db_id = split_dataset_and_db_id(
        dataset,
        db_id,
    )

    if hasattr(config, "postgres_schema_name"):
        return config.postgres_schema_name(
            dataset_key,
            clean_db_id,
        )

    return safe_identifier(
        f"{dataset_key}_{clean_db_id}"
    )


DATABASE_HEALTH_ERROR = ""


def database_health() -> bool:
    """Return True when PostgreSQL is reachable and expose the real error."""

    global DATABASE_HEALTH_ERROR

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        DATABASE_HEALTH_ERROR = ""
        print("PostgreSQL connection successful.", flush=True)
        return True

    except Exception as exc:
        DATABASE_HEALTH_ERROR = f"{type(exc).__name__}: {exc}"
        print("POSTGRESQL CONNECTION ERROR:", DATABASE_HEALTH_ERROR, flush=True)
        return False


def list_schemas(
    dataset: str | None = None,
) -> list[str]:
    """List migrated PostgreSQL schemas."""

    query = """
        SELECT schema_name
        FROM information_schema.schemata
        WHERE schema_name NOT IN (
            'public',
            'pg_catalog',
            'information_schema'
        )
          AND schema_name NOT LIKE 'pg_%'
    """

    parameters: dict[str, Any] = {}

    if dataset is not None:
        dataset_key = normalise_dataset(dataset)
        query += " AND schema_name LIKE :prefix"
        parameters["prefix"] = f"{dataset_key}_%"

    query += " ORDER BY schema_name"

    with engine.connect() as connection:
        rows = connection.execute(
            text(query),
            parameters,
        ).fetchall()

    return [str(row[0]) for row in rows]


def list_databases(dataset: str) -> list[str]:
    """Return clean database IDs for a selected dataset."""

    dataset_key = normalise_dataset(dataset)
    prefix = f"{dataset_key}_"

    return [
        schema[len(prefix):]
        for schema in list_schemas(dataset_key)
        if schema.startswith(prefix)
    ]


def schema_exists(
    db_id: str,
    dataset: str | None = None,
) -> bool:
    """Check whether a migrated PostgreSQL schema exists."""

    schema = postgres_schema_name(dataset, db_id)

    query = text(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.schemata
            WHERE schema_name = :schema
        )
        """
    )

    with engine.connect() as connection:
        return bool(
            connection.execute(
                query,
                {"schema": schema},
            ).scalar_one()
        )


def schema_metadata(
    db_id: str,
    dataset: str | None = None,
) -> dict[str, list[dict[str, str]]]:
    """Read table and column metadata from PostgreSQL."""

    schema = postgres_schema_name(
        dataset,
        db_id,
    )

    query = text(
        """
        SELECT
            table_name,
            column_name,
            data_type
        FROM information_schema.columns
        WHERE table_schema = :schema
        ORDER BY table_name, ordinal_position
        """
    )

    tables: dict[str, list[dict[str, str]]] = {}

    with engine.connect() as connection:
        rows = connection.execute(
            query,
            {"schema": schema},
        ).fetchall()

    for table_name, column_name, data_type in rows:
        tables.setdefault(
            str(table_name),
            [],
        ).append(
            {
                "name": str(column_name),
                "type": str(data_type),
            }
        )

    return tables


def quote_identifier(name: str) -> str:
    """
    Return an exact PostgreSQL quoted identifier.

    PostgreSQL folds unquoted identifiers to lowercase. BIRD and Spider
    can contain mixed-case identifiers such as CDSCode and AvgScrMath,
    so quoting preserves the migrated schema exactly.
    """
    return '"' + str(name).replace('"', '""') + '"'


def format_table_for_prompt(
    table_name: str,
    columns: list[dict[str, str]],
) -> str:
    """Format one table for the LLM using exact PostgreSQL identifiers."""

    lines = [
        f"Table: {quote_identifier(table_name)}",
        "Columns:",
    ]

    for column in columns:
        lines.append(
            f"- {quote_identifier(column['name'])} "
            f"({column['type']})"
        )

    return "\n".join(lines)


def full_schema_context(
    db_id: str,
    dataset: str | None = None,
) -> tuple[str, list[str]]:
    """Format the complete PostgreSQL schema for the LLM."""

    tables = schema_metadata(
        db_id=db_id,
        dataset=dataset,
    )

    sections: list[str] = []

    for table_name, columns in tables.items():
        sections.append(
            format_table_for_prompt(
                table_name,
                columns,
            )
        )

    return (
        "\n\n".join(sections).strip(),
        list(tables.keys()),
    )


# ============================================================
# TF-IDF schema retrieval
# ============================================================

class Retriever:
    """Retrieve the most relevant tables for a question."""

    def __init__(self) -> None:
        self.cache: dict[
            tuple[str, str],
            tuple[
                dict[str, list[dict[str, str]]],
                list[str],
                TfidfVectorizer,
                Any,
            ],
        ] = {}

    def clear(self) -> None:
        """Clear cached schema indexes."""

        self.cache.clear()

    def retrieve(
        self,
        question: str,
        db_id: str,
        top_k: int,
        dataset: str | None = None,
    ) -> tuple[str, list[str]]:
        """Return a prompt context containing the top-k tables."""

        dataset_key, clean_db_id = split_dataset_and_db_id(
            dataset,
            db_id,
        )

        cache_key = (
            dataset_key,
            clean_db_id,
        )

        if cache_key not in self.cache:
            tables = schema_metadata(
                db_id=clean_db_id,
                dataset=dataset_key,
            )

            names = list(tables.keys())

            documents = [
                (
                    f"table {table_name} columns "
                    + " ".join(
                        column["name"]
                        for column in columns
                    )
                )
                for table_name, columns in tables.items()
            ]

            if not documents:
                return "", []

            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
            )

            matrix = vectorizer.fit_transform(
                documents
            )

            self.cache[cache_key] = (
                tables,
                names,
                vectorizer,
                matrix,
            )

        (
            tables,
            names,
            vectorizer,
            matrix,
        ) = self.cache[cache_key]

        scores = cosine_similarity(
            vectorizer.transform([question]),
            matrix,
        ).flatten()

        limit = min(
            max(int(top_k), 1),
            len(names),
        )

        selected_indices = scores.argsort()[::-1][:limit]

        selected_tables = [
            names[index]
            for index in selected_indices
        ]

        sections: list[str] = []

        for table_name in selected_tables:
            sections.append(
                format_table_for_prompt(
                    table_name,
                    tables[table_name],
                )
            )

        return (
            "\n\n".join(sections).strip(),
            selected_tables,
        )


retriever = Retriever()


def schema_for_strategy(
    question: str,
    db_id: str,
    strategy: str,
    dataset: str | None = None,
) -> tuple[str, list[str]]:
    """
    Return schema context for an experimental strategy.

    Supported strategies:
        full
        top_1
        top_3
        top_5
    """

    strategy_key = str(strategy).strip().lower()

    if strategy_key == "full":
        return full_schema_context(
            db_id=db_id,
            dataset=dataset,
        )

    match = re.fullmatch(
        r"top_(\d+)",
        strategy_key,
    )

    if not match:
        raise ValueError(
            "Unknown strategy. Use full, top_1, top_3 or top_5."
        )

    return retriever.retrieve(
        question=question,
        db_id=db_id,
        top_k=int(match.group(1)),
        dataset=dataset,
    )


# ============================================================
# OpenAI SQL generation
# ============================================================

def count_tokens(value: str) -> int:
    """Count or estimate tokens."""

    try:
        encoding = tiktoken.encoding_for_model(
            LLM_MODEL
        )
        return len(encoding.encode(value))
    except Exception:
        return max(1, len(value) // 4)


def clean_sql(value: str) -> str:
    """Remove markdown fences and common prefixes."""

    cleaned = str(value or "").strip()

    fenced_match = re.search(
        r"```(?:sql)?\s*(.*?)```",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if fenced_match:
        cleaned = fenced_match.group(1).strip()

    cleaned = re.sub(
        r"^\s*sql\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()

    return cleaned


def generate_sql(
    question: str,
    schema_context: str,
    evidence: str | None = None,
) -> tuple[str, float, int, int, bool]:
    """Generate exactly one read-only PostgreSQL query."""

    evidence_text = str(evidence or "").strip()

    evidence_section = (
        f"""
BENCHMARK EVIDENCE / HINT
-------------------------
{evidence_text}
"""
        if evidence_text
        else ""
    )

    prompt = f"""
You are a PostgreSQL Text-to-SQL system.

DATABASE SCHEMA
---------------
{schema_context}

USER QUESTION
-------------
{question}

{evidence_section}

STRICT RULES
------------
1. Return exactly one PostgreSQL SELECT query.
2. WITH is allowed only when the final statement is SELECT.
3. Use ONLY tables and columns shown in DATABASE SCHEMA.
4. Preserve every table and column name exactly as shown.
5. Any identifier shown inside double quotes MUST remain double-quoted.
6. Never change the spelling or capitalization of an identifier.
7. Never invent a table or column.
8. Build JOIN conditions only from columns that actually exist.
9. Use BENCHMARK EVIDENCE / HINT when supplied, but never copy a gold SQL answer.
10. Do not guess coded values unless their meaning is supplied by the question,
    schema, or benchmark evidence.
11. Return SQL only.
12. Do not return markdown.
13. Do not provide explanations or comments.
""".strip()

    started_at = time.perf_counter()

    if not OPENAI_API_KEY:
        demo_sql = "SELECT 1 AS demo_mode;"

        return (
            demo_sql,
            time.perf_counter() - started_at,
            count_tokens(prompt),
            count_tokens(demo_sql),
            True,
        )

    client = OpenAI(
        api_key=OPENAI_API_KEY
    )

    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Generate one valid read-only PostgreSQL query. "
                    "Use only the supplied schema. Preserve PostgreSQL "
                    "identifiers exactly and keep quoted identifiers quoted. "
                    "Return SQL only."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    generated_sql = clean_sql(
        response.choices[0].message.content or ""
    )

    usage = response.usage

    prompt_tokens = (
        int(usage.prompt_tokens)
        if usage is not None
        else count_tokens(prompt)
    )

    completion_tokens = (
        int(usage.completion_tokens)
        if usage is not None
        else count_tokens(generated_sql)
    )

    return (
        generated_sql,
        time.perf_counter() - started_at,
        prompt_tokens,
        completion_tokens,
        False,
    )


# ============================================================
# SQL validation and execution
# ============================================================

def validate_sql(
    sql_query: str,
) -> tuple[bool, str | None]:
    """Allow only one read-only SELECT or WITH statement."""

    cleaned = str(sql_query or "").strip()

    if not cleaned:
        return False, "Generated SQL is empty."

    if BLOCKED_SQL.search(cleaned):
        return False, "Only read-only SQL queries are allowed."

    without_final_semicolon = cleaned.rstrip(";")

    if ";" in without_final_semicolon:
        return False, "Multiple SQL statements are blocked."

    if not re.match(
        r"^\s*(select|with)\b",
        cleaned,
        flags=re.IGNORECASE,
    ):
        return False, "Query must start with SELECT or WITH."

    return True, None


def execute_sql(
    sql_query: str,
    db_id: str,
    dataset: str | None = None,
) -> tuple[
    bool,
    list[str],
    list[list[Any]],
    str | None,
    float,
]:
    """Execute SQL in the correct migrated PostgreSQL schema."""

    started_at = time.perf_counter()

    valid, validation_error = validate_sql(
        sql_query
    )

    if not valid:
        return (
            False,
            [],
            [],
            validation_error,
            time.perf_counter() - started_at,
        )

    schema = postgres_schema_name(
        dataset,
        db_id,
    )

    try:
        with engine.connect() as connection:
            transaction = connection.begin()

            try:
                connection.execute(
                    text(
                        f'SET LOCAL search_path TO '
                        f'"{schema}", public'
                    )
                )

                connection.execute(
                    text(
                        "SET LOCAL TRANSACTION READ ONLY"
                    )
                )

                connection.execute(
                    text(
                        "SET LOCAL statement_timeout = 30000"
                    )
                )

                result = connection.execute(
                    text(sql_query)
                )

                columns = (
                    list(result.keys())
                    if result.returns_rows
                    else []
                )

                rows = (
                    [
                        list(row)
                        for row in result.fetchmany(
                            MAX_RESULT_ROWS
                        )
                    ]
                    if result.returns_rows
                    else []
                )

                transaction.rollback()

            except Exception:
                transaction.rollback()
                raise

        return (
            True,
            columns,
            rows,
            None,
            time.perf_counter() - started_at,
        )

    except Exception as exc:
        return (
            False,
            [],
            [],
            str(exc),
            time.perf_counter() - started_at,
        )


# ============================================================
# Evaluation
# ============================================================

def normalise_sql(sql_query: str) -> str:
    """Normalise SQL for exact-match evaluation."""

    return re.sub(
        r"\s+",
        " ",
        str(sql_query or "").strip().lower(),
    ).rstrip(";")


def execution_accuracy(
    predicted_sql: str,
    gold_sql: str | None,
    db_id: str,
    dataset: str | None = None,
) -> int | None:
    """Compare predicted and benchmark result sets."""

    if not gold_sql:
        return None

    (
        predicted_success,
        predicted_columns,
        predicted_rows,
        _,
        _,
    ) = execute_sql(
        sql_query=predicted_sql,
        db_id=db_id,
        dataset=dataset,
    )

    (
        gold_success,
        gold_columns,
        gold_rows,
        _,
        _,
    ) = execute_sql(
        sql_query=gold_sql,
        db_id=db_id,
        dataset=dataset,
    )

    if not predicted_success or not gold_success:
        return 0

    if (
        predicted_columns == gold_columns
        and predicted_rows == gold_rows
    ):
        return 1

    predicted_set = sorted(
        map(repr, predicted_rows)
    )

    gold_set = sorted(
        map(repr, gold_rows)
    )

    return int(predicted_set == gold_set)


def run_strategy(
    question: str,
    db_id: str,
    strategy: str,
    gold_sql: str | None = None,
    dataset: str | None = None,
    evidence: str | None = None,
) -> dict[str, Any]:
    """Run retrieval, generation, execution and evaluation."""

    dataset_key, clean_db_id = split_dataset_and_db_id(
        dataset,
        db_id,
    )

    schema_context, retrieved_tables = (
        schema_for_strategy(
            question=question,
            db_id=clean_db_id,
            strategy=strategy,
            dataset=dataset_key,
        )
    )

    (
        generated_sql,
        generation_latency,
        prompt_tokens,
        completion_tokens,
        demo_mode,
    ) = generate_sql(
        question=question,
        schema_context=schema_context,
        evidence=evidence,
    )

    valid, validation_error = validate_sql(
        generated_sql
    )

    if valid:
        (
            success,
            columns,
            rows,
            execution_error,
            execution_time,
        ) = execute_sql(
            sql_query=generated_sql,
            db_id=clean_db_id,
            dataset=dataset_key,
        )
    else:
        success = False
        columns = []
        rows = []
        execution_error = validation_error
        execution_time = 0.0

    exact_match = (
        None
        if not gold_sql
        else int(
            normalise_sql(generated_sql)
            == normalise_sql(gold_sql)
        )
    )

    return {
        "dataset": dataset_key,
        "strategy": strategy,
        "db_id": clean_db_id,
        "postgres_schema": postgres_schema_name(
            dataset_key,
            clean_db_id,
        ),
        "question": question,
        "evidence": evidence or "",
        "schema_context": schema_context,
        "retrieved_tables": retrieved_tables,
        "generated_sql": generated_sql,
        "success": int(success),
        "columns": columns,
        "rows": rows[:MAX_RESULT_ROWS],
        "error": execution_error,
        "generation_latency": generation_latency,
        "execution_time": execution_time,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "exact_match": exact_match,
        "execution_accuracy": execution_accuracy(
            predicted_sql=generated_sql,
            gold_sql=gold_sql,
            db_id=clean_db_id,
            dataset=dataset_key,
        ),
        "demo_mode": demo_mode,
    }


# ============================================================
# Benchmark question loading
# ============================================================

def discover_question_file(
    dataset: str = "bird",
) -> Path | None:
    """Find a local BIRD or Spider benchmark question file."""

    dataset_key = normalise_dataset(dataset)

    if dataset_key == "bird":
        root = Path(config.BIRD_DATA_DIR)
        preferred = [
            root / "mini_dev_sqlite.json",
            root / "dev.json",
            root / "mini_dev_data" / "mini_dev_sqlite.json",
        ]
    else:
        root = Path(config.SPIDER_DATA_DIR)
        preferred = [
            root / "dev.json",
            root / "train_spider.json",
            root / "train_others.json",
        ]

    for path in preferred:
        if path.exists():
            return path

    excluded_tokens = (
        "predict",
        "submission",
        "output",
        "result",
        "exp_result",
        "ta_output",
        "tables",
    )

    candidates: list[tuple[int, Path]] = []

    for path in root.rglob("*.json"):
        lowered_path = str(path).lower()

        if any(
            token in lowered_path
            for token in excluded_tokens
        ):
            continue

        try:
            payload = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

            records = (
                payload
                if isinstance(payload, list)
                else payload.get("data", [])
                if isinstance(payload, dict)
                else []
            )

            if not records or not isinstance(
                records[0],
                dict,
            ):
                continue

            keys = {
                str(key).lower()
                for key in records[0]
            }

            score = (
                20 * int("question" in keys)
                + 20 * int(
                    "db_id" in keys
                    or "database_id" in keys
                )
                + 20 * int(
                    bool(
                        {
                            "sql",
                            "query",
                            "gold_sql",
                        }
                        & keys
                    )
                )
            )

            candidates.append(
                (score, path)
            )

        except Exception:
            continue

    return max(
        candidates,
        default=(0, None),
        key=lambda item: item[0],
    )[1]


def load_questions(
    limit: int | None = None,
    db_id: str | None = None,
    dataset: str = "bird",
) -> list[dict[str, Any]]:
    """Load benchmark questions for BIRD or Spider."""

    dataset_key = normalise_dataset(dataset)
    path = discover_question_file(dataset_key)

    if path is None:
        return []

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    records = (
        payload
        if isinstance(payload, list)
        else payload.get("data", [])
        if isinstance(payload, dict)
        else []
    )

    output: list[dict[str, Any]] = []

    for index, item in enumerate(records):
        if not isinstance(item, dict):
            continue

        item_db_id = str(
            item.get("db_id")
            or item.get("database_id")
            or item.get("database")
            or ""
        )

        if db_id and item_db_id != db_id:
            continue

        question = str(
            item.get("question")
            or item.get("utterance")
            or item.get("text")
            or ""
        )

        if not item_db_id or not question:
            continue

        output.append(
            {
                "question_id": str(
                    item.get(
                        "question_id",
                        item.get("id", index),
                    )
                ),
                "dataset": dataset_key,
                "db_id": item_db_id,
                "question": question,
                "gold_sql": (
                    item.get("SQL")
                    or item.get("sql")
                    or item.get("query")
                    or item.get("gold_sql")
                    or ""
                ),
                "evidence": item.get(
                    "evidence",
                    "",
                ),
                "difficulty": item.get(
                    "difficulty",
                    item.get(
                        "question_type",
                        "",
                    ),
                ),
                "source_file": str(path),
            }
        )

        if limit and len(output) >= int(limit):
            break

    return output


# ============================================================
# Project status and schema statistics
# ============================================================

def project_status() -> dict[str, Any]:
    """Return project and data health information."""

    connected = database_health()
    schemas = list_schemas() if connected else []

    bird_questions = load_questions(
        dataset="bird"
    )

    spider_questions = load_questions(
        dataset="spider"
    )

    return {
        "database_connected": connected,
        "bird_questions": len(
            bird_questions
        ),
        "spider_questions": len(
            spider_questions
        ),
        "questions": (
            len(bird_questions)
            + len(spider_questions)
        ),
        "bird_question_file": (
            bird_questions[0]["source_file"]
            if bird_questions
            else None
        ),
        "spider_question_file": (
            spider_questions[0]["source_file"]
            if spider_questions
            else None
        ),
        "schemas": schemas,
        "bird_schemas": len(
            [
                schema
                for schema in schemas
                if schema.startswith("bird_")
            ]
        ),
        "spider_schemas": len(
            [
                schema
                for schema in schemas
                if schema.startswith("spider_")
            ]
        ),
        "openai_configured": bool(
            OPENAI_API_KEY
        ),
        "output_dir": str(
            config.OUTPUT_DIR
        ),
        "bird_data_dir": str(
            config.BIRD_DATA_DIR
        ),
        "spider_data_dir": str(
            config.SPIDER_DATA_DIR
        ),
    }


def schema_statistics(
    dataset: str | None = None,
) -> list[dict[str, Any]]:
    """Count tables and rows in migrated PostgreSQL schemas."""

    statistics: list[dict[str, Any]] = []

    with engine.connect() as connection:
        for schema in list_schemas(dataset):
            table_rows = connection.execute(
                text(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = :schema
                      AND table_type = 'BASE TABLE'
                    ORDER BY table_name
                    """
                ),
                {"schema": schema},
            ).fetchall()

            table_names = [
                str(row[0])
                for row in table_rows
            ]

            total_rows = 0

            for table_name in table_names:
                escaped_schema = schema.replace(
                    '"',
                    '""',
                )
                escaped_table = table_name.replace(
                    '"',
                    '""',
                )

                try:
                    total_rows += int(
                        connection.execute(
                            text(
                                f'SELECT COUNT(*) '
                                f'FROM "{escaped_schema}".'
                                f'"{escaped_table}"'
                            )
                        ).scalar_one()
                    )
                except Exception:
                    continue

            if schema.startswith("bird_"):
                schema_dataset = "bird"
                db_id = schema[5:]
            elif schema.startswith("spider_"):
                schema_dataset = "spider"
                db_id = schema[7:]
            else:
                schema_dataset = "unknown"
                db_id = schema

            statistics.append(
                {
                    "dataset": schema_dataset,
                    "db_id": db_id,
                    "schema": schema,
                    "tables": len(table_names),
                    "rows": total_rows,
                }
            )

    return statistics


def schema_statistics_frame(
    dataset: str | None = None,
) -> pd.DataFrame:
    """Return PostgreSQL schema statistics as a DataFrame."""

    return pd.DataFrame(
        schema_statistics(dataset)
    )
core_full_updated.py


from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import tiktoken
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src import config


# ============================================================
# Configuration
# ============================================================

SQLALCHEMY_URL = getattr(
    config,
    "SQLALCHEMY_URL",
    getattr(config, "DATABASE_URL", ""),
)

if not SQLALCHEMY_URL:
    raise RuntimeError(
        "PostgreSQL connection URL is missing. Define SQLALCHEMY_URL "
        "or DATABASE_URL in src/config.py."
    )

OPENAI_API_KEY = getattr(config, "OPENAI_API_KEY", "")
LLM_MODEL = getattr(
    config,
    "LLM_MODEL",
    getattr(config, "OPENAI_MODEL", "gpt-4o-mini"),
)
MAX_RESULT_ROWS = int(getattr(config, "MAX_RESULT_ROWS", 200))

engine: Engine = create_engine(
    SQLALCHEMY_URL,
    pool_pre_ping=True,
    future=True,
)

BLOCKED_SQL = re.compile(
    r"\b("
    r"insert|update|delete|drop|alter|create|truncate|grant|revoke|"
    r"copy|call|do|merge|vacuum|analyze|refresh|reindex|cluster"
    r")\b",
    re.IGNORECASE,
)


# ============================================================
# Dataset and PostgreSQL schema helpers
# ============================================================

def safe_identifier(name: str) -> str:
    """Convert a value into a safe PostgreSQL-style identifier."""

    value = re.sub(
        r"[^a-zA-Z0-9_]+",
        "_",
        str(name),
    ).strip("_").lower()

    if not value:
        value = "database"

    if value[0].isdigit():
        value = f"db_{value}"

    return value[:63]


def normalise_dataset(dataset: str | None) -> str:
    """Validate and return a supported dataset name."""

    value = str(dataset or "bird").strip().lower()

    if value not in {"bird", "spider"}:
        raise ValueError(
            f"Unsupported dataset '{dataset}'. Use 'bird' or 'spider'."
        )

    return value


def split_dataset_and_db_id(
    dataset: str | None,
    db_id: str,
) -> tuple[str, str]:
    """
    Resolve the dataset and clean database ID.

    Supported inputs:
        dataset='bird', db_id='financial'
        dataset='bird', db_id='bird_financial'
        dataset=None, db_id='bird_financial'
        dataset=None, db_id='spider_concert_singer'
    """

    raw_db_id = str(db_id or "").strip()

    if not raw_db_id:
        raise ValueError("db_id cannot be empty.")

    lowered = raw_db_id.lower()

    if dataset is None:
        if lowered.startswith("bird_"):
            return "bird", raw_db_id[5:]

        if lowered.startswith("spider_"):
            return "spider", raw_db_id[7:]

        # Backward-compatible default for the original BIRD-only project.
        return "bird", raw_db_id

    dataset_key = normalise_dataset(dataset)
    prefix = f"{dataset_key}_"

    if lowered.startswith(prefix):
        raw_db_id = raw_db_id[len(prefix):]

    return dataset_key, raw_db_id


def postgres_schema_name(
    dataset: str | None,
    db_id: str,
) -> str:
    """Return the migrated PostgreSQL schema name."""

    dataset_key, clean_db_id = split_dataset_and_db_id(
        dataset,
        db_id,
    )

    if hasattr(config, "postgres_schema_name"):
        return config.postgres_schema_name(
            dataset_key,
            clean_db_id,
        )

    return safe_identifier(
        f"{dataset_key}_{clean_db_id}"
    )


DATABASE_HEALTH_ERROR = ""


def database_health() -> bool:
    """Return True when PostgreSQL is reachable and expose the real error."""

    global DATABASE_HEALTH_ERROR

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        DATABASE_HEALTH_ERROR = ""
        print("PostgreSQL connection successful.", flush=True)
        return True

    except Exception as exc:
        DATABASE_HEALTH_ERROR = f"{type(exc).__name__}: {exc}"
        print("POSTGRESQL CONNECTION ERROR:", DATABASE_HEALTH_ERROR, flush=True)
        return False


def list_schemas(
    dataset: str | None = None,
) -> list[str]:
    """List migrated PostgreSQL schemas."""

    query = """
        SELECT schema_name
        FROM information_schema.schemata
        WHERE schema_name NOT IN (
            'public',
            'pg_catalog',
            'information_schema'
        )
          AND schema_name NOT LIKE 'pg_%'
    """

    parameters: dict[str, Any] = {}

    if dataset is not None:
        dataset_key = normalise_dataset(dataset)
        query += " AND schema_name LIKE :prefix"
        parameters["prefix"] = f"{dataset_key}_%"

    query += " ORDER BY schema_name"

    with engine.connect() as connection:
        rows = connection.execute(
            text(query),
            parameters,
        ).fetchall()

    return [str(row[0]) for row in rows]


def list_databases(dataset: str) -> list[str]:
    """Return clean database IDs for a selected dataset."""

    dataset_key = normalise_dataset(dataset)
    prefix = f"{dataset_key}_"

    return [
        schema[len(prefix):]
        for schema in list_schemas(dataset_key)
        if schema.startswith(prefix)
    ]


def schema_exists(
    db_id: str,
    dataset: str | None = None,
) -> bool:
    """Check whether a migrated PostgreSQL schema exists."""

    schema = postgres_schema_name(dataset, db_id)

    query = text(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.schemata
            WHERE schema_name = :schema
        )
        """
    )

    with engine.connect() as connection:
        return bool(
            connection.execute(
                query,
                {"schema": schema},
            ).scalar_one()
        )


def schema_metadata(
    db_id: str,
    dataset: str | None = None,
) -> dict[str, list[dict[str, str]]]:
    """Read table and column metadata from PostgreSQL."""

    schema = postgres_schema_name(
        dataset,
        db_id,
    )

    query = text(
        """
        SELECT
            table_name,
            column_name,
            data_type
        FROM information_schema.columns
        WHERE table_schema = :schema
        ORDER BY table_name, ordinal_position
        """
    )

    tables: dict[str, list[dict[str, str]]] = {}

    with engine.connect() as connection:
        rows = connection.execute(
            query,
            {"schema": schema},
        ).fetchall()

    for table_name, column_name, data_type in rows:
        tables.setdefault(
            str(table_name),
            [],
        ).append(
            {
                "name": str(column_name),
                "type": str(data_type),
            }
        )

    return tables


def quote_identifier(name: str) -> str:
    """
    Return an exact PostgreSQL quoted identifier.

    PostgreSQL folds unquoted identifiers to lowercase. BIRD and Spider
    can contain mixed-case identifiers such as CDSCode and AvgScrMath,
    so quoting preserves the migrated schema exactly.
    """
    return '"' + str(name).replace('"', '""') + '"'


def format_table_for_prompt(
    table_name: str,
    columns: list[dict[str, str]],
) -> str:
    """Format one table for the LLM using exact PostgreSQL identifiers."""

    lines = [
        f"Table: {quote_identifier(table_name)}",
        "Columns:",
    ]

    for column in columns:
        lines.append(
            f"- {quote_identifier(column['name'])} "
            f"({column['type']})"
        )

    return "\n".join(lines)


def full_schema_context(
    db_id: str,
    dataset: str | None = None,
) -> tuple[str, list[str]]:
    """Format the complete PostgreSQL schema for the LLM."""

    tables = schema_metadata(
        db_id=db_id,
        dataset=dataset,
    )

    sections: list[str] = []

    for table_name, columns in tables.items():
        sections.append(
            format_table_for_prompt(
                table_name,
                columns,
            )
        )

    return (
        "\n\n".join(sections).strip(),
        list(tables.keys()),
    )


# ============================================================
# TF-IDF schema retrieval
# ============================================================

class Retriever:
    """Retrieve the most relevant tables for a question."""

    def __init__(self) -> None:
        self.cache: dict[
            tuple[str, str],
            tuple[
                dict[str, list[dict[str, str]]],
                list[str],
                TfidfVectorizer,
                Any,
            ],
        ] = {}

    def clear(self) -> None:
        """Clear cached schema indexes."""

        self.cache.clear()

    def retrieve(
        self,
        question: str,
        db_id: str,
        top_k: int,
        dataset: str | None = None,
    ) -> tuple[str, list[str]]:
        """Return a prompt context containing the top-k tables."""

        dataset_key, clean_db_id = split_dataset_and_db_id(
            dataset,
            db_id,
        )

        cache_key = (
            dataset_key,
            clean_db_id,
        )

        if cache_key not in self.cache:
            tables = schema_metadata(
                db_id=clean_db_id,
                dataset=dataset_key,
            )

            names = list(tables.keys())

            documents = [
                (
                    f"table {table_name} columns "
                    + " ".join(
                        column["name"]
                        for column in columns
                    )
                )
                for table_name, columns in tables.items()
            ]

            if not documents:
                return "", []

            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
            )

            matrix = vectorizer.fit_transform(
                documents
            )

            self.cache[cache_key] = (
                tables,
                names,
                vectorizer,
                matrix,
            )

        (
            tables,
            names,
            vectorizer,
            matrix,
        ) = self.cache[cache_key]

        scores = cosine_similarity(
            vectorizer.transform([question]),
            matrix,
        ).flatten()

        limit = min(
            max(int(top_k), 1),
            len(names),
        )

        selected_indices = scores.argsort()[::-1][:limit]

        selected_tables = [
            names[index]
            for index in selected_indices
        ]

        sections: list[str] = []

        for table_name in selected_tables:
            sections.append(
                format_table_for_prompt(
                    table_name,
                    tables[table_name],
                )
            )

        return (
            "\n\n".join(sections).strip(),
            selected_tables,
        )


retriever = Retriever()


def schema_for_strategy(
    question: str,
    db_id: str,
    strategy: str,
    dataset: str | None = None,
) -> tuple[str, list[str]]:
    """
    Return schema context for an experimental strategy.

    Supported strategies:
        full
        top_1
        top_3
        top_5
    """

    strategy_key = str(strategy).strip().lower()

    if strategy_key == "full":
        return full_schema_context(
            db_id=db_id,
            dataset=dataset,
        )

    match = re.fullmatch(
        r"top_(\d+)",
        strategy_key,
    )

    if not match:
        raise ValueError(
            "Unknown strategy. Use full, top_1, top_3 or top_5."
        )

    return retriever.retrieve(
        question=question,
        db_id=db_id,
        top_k=int(match.group(1)),
        dataset=dataset,
    )


# ============================================================
# OpenAI SQL generation
# ============================================================

def count_tokens(value: str) -> int:
    """Count or estimate tokens."""

    try:
        encoding = tiktoken.encoding_for_model(
            LLM_MODEL
        )
        return len(encoding.encode(value))
    except Exception:
        return max(1, len(value) // 4)


def clean_sql(value: str) -> str:
    """Remove markdown fences and common prefixes."""

    cleaned = str(value or "").strip()

    fenced_match = re.search(
        r"```(?:sql)?\s*(.*?)```",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if fenced_match:
        cleaned = fenced_match.group(1).strip()

    cleaned = re.sub(
        r"^\s*sql\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()

    return cleaned


def generate_sql(
    question: str,
    schema_context: str,
    evidence: str | None = None,
) -> tuple[str, float, int, int, bool]:
    """Generate exactly one read-only PostgreSQL query."""

    evidence_text = str(evidence or "").strip()

    evidence_section = (
        f"""
BENCHMARK EVIDENCE / HINT
-------------------------
{evidence_text}
"""
        if evidence_text
        else ""
    )

    prompt = f"""
You are a PostgreSQL Text-to-SQL system.

DATABASE SCHEMA
---------------
{schema_context}

USER QUESTION
-------------
{question}

{evidence_section}

STRICT RULES
------------
1. Return exactly one PostgreSQL SELECT query.
2. WITH is allowed only when the final statement is SELECT.
3. Use ONLY tables and columns shown in DATABASE SCHEMA.
4. Preserve every table and column name exactly as shown.
5. Any identifier shown inside double quotes MUST remain double-quoted.
6. Never change the spelling or capitalization of an identifier.
7. Never invent a table or column.
8. Build JOIN conditions only from columns that actually exist.
9. Use BENCHMARK EVIDENCE / HINT when supplied, but never copy a gold SQL answer.
10. Do not guess coded values unless their meaning is supplied by the question,
    schema, or benchmark evidence.
11. Return SQL only.
12. Do not return markdown.
13. Do not provide explanations or comments.
""".strip()

    started_at = time.perf_counter()

    if not OPENAI_API_KEY:
        demo_sql = "SELECT 1 AS demo_mode;"

        return (
            demo_sql,
            time.perf_counter() - started_at,
            count_tokens(prompt),
            count_tokens(demo_sql),
            True,
        )

    client = OpenAI(
        api_key=OPENAI_API_KEY
    )

    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Generate one valid read-only PostgreSQL query. "
                    "Use only the supplied schema. Preserve PostgreSQL "
                    "identifiers exactly and keep quoted identifiers quoted. "
                    "Return SQL only."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    generated_sql = clean_sql(
        response.choices[0].message.content or ""
    )

    usage = response.usage

    prompt_tokens = (
        int(usage.prompt_tokens)
        if usage is not None
        else count_tokens(prompt)
    )

    completion_tokens = (
        int(usage.completion_tokens)
        if usage is not None
        else count_tokens(generated_sql)
    )

    return (
        generated_sql,
        time.perf_counter() - started_at,
        prompt_tokens,
        completion_tokens,
        False,
    )


# ============================================================
# SQL validation and execution
# ============================================================

def validate_sql(
    sql_query: str,
) -> tuple[bool, str | None]:
    """Allow only one read-only SELECT or WITH statement."""

    cleaned = str(sql_query or "").strip()

    if not cleaned:
        return False, "Generated SQL is empty."

    if BLOCKED_SQL.search(cleaned):
        return False, "Only read-only SQL queries are allowed."

    without_final_semicolon = cleaned.rstrip(";")

    if ";" in without_final_semicolon:
        return False, "Multiple SQL statements are blocked."

    if not re.match(
        r"^\s*(select|with)\b",
        cleaned,
        flags=re.IGNORECASE,
    ):
        return False, "Query must start with SELECT or WITH."

    return True, None


def execute_sql(
    sql_query: str,
    db_id: str,
    dataset: str | None = None,
) -> tuple[
    bool,
    list[str],
    list[list[Any]],
    str | None,
    float,
]:
    """Execute SQL in the correct migrated PostgreSQL schema."""

    started_at = time.perf_counter()

    valid, validation_error = validate_sql(
        sql_query
    )

    if not valid:
        return (
            False,
            [],
            [],
            validation_error,
            time.perf_counter() - started_at,
        )

    schema = postgres_schema_name(
        dataset,
        db_id,
    )

    try:
        with engine.connect() as connection:
            transaction = connection.begin()

            try:
                connection.execute(
                    text(
                        f'SET LOCAL search_path TO '
                        f'"{schema}", public'
                    )
                )

                connection.execute(
                    text(
                        "SET LOCAL TRANSACTION READ ONLY"
                    )
                )

                connection.execute(
                    text(
                        "SET LOCAL statement_timeout = 30000"
                    )
                )

                result = connection.execute(
                    text(sql_query)
                )

                columns = (
                    list(result.keys())
                    if result.returns_rows
                    else []
                )

                rows = (
                    [
                        list(row)
                        for row in result.fetchmany(
                            MAX_RESULT_ROWS
                        )
                    ]
                    if result.returns_rows
                    else []
                )

                transaction.rollback()

            except Exception:
                transaction.rollback()
                raise

        return (
            True,
            columns,
            rows,
            None,
            time.perf_counter() - started_at,
        )

    except Exception as exc:
        return (
            False,
            [],
            [],
            str(exc),
            time.perf_counter() - started_at,
        )


# ============================================================
# Evaluation
# ============================================================

def normalise_sql(sql_query: str) -> str:
    """Normalise SQL for exact-match evaluation."""

    return re.sub(
        r"\s+",
        " ",
        str(sql_query or "").strip().lower(),
    ).rstrip(";")


def execution_accuracy(
    predicted_sql: str,
    gold_sql: str | None,
    db_id: str,
    dataset: str | None = None,
) -> int | None:
    """Compare predicted and benchmark result sets."""

    if not gold_sql:
        return None

    (
        predicted_success,
        predicted_columns,
        predicted_rows,
        _,
        _,
    ) = execute_sql(
        sql_query=predicted_sql,
        db_id=db_id,
        dataset=dataset,
    )

    (
        gold_success,
        gold_columns,
        gold_rows,
        _,
        _,
    ) = execute_sql(
        sql_query=gold_sql,
        db_id=db_id,
        dataset=dataset,
    )

    if not predicted_success or not gold_success:
        return 0

    if (
        predicted_columns == gold_columns
        and predicted_rows == gold_rows
    ):
        return 1

    predicted_set = sorted(
        map(repr, predicted_rows)
    )

    gold_set = sorted(
        map(repr, gold_rows)
    )

    return int(predicted_set == gold_set)


def run_strategy(
    question: str,
    db_id: str,
    strategy: str,
    gold_sql: str | None = None,
    dataset: str | None = None,
    evidence: str | None = None,
) -> dict[str, Any]:
    """Run retrieval, generation, execution and evaluation."""

    dataset_key, clean_db_id = split_dataset_and_db_id(
        dataset,
        db_id,
    )

    schema_context, retrieved_tables = (
        schema_for_strategy(
            question=question,
            db_id=clean_db_id,
            strategy=strategy,
            dataset=dataset_key,
        )
    )

    (
        generated_sql,
        generation_latency,
        prompt_tokens,
        completion_tokens,
        demo_mode,
    ) = generate_sql(
        question=question,
        schema_context=schema_context,
        evidence=evidence,
    )

    valid, validation_error = validate_sql(
        generated_sql
    )

    if valid:
        (
            success,
            columns,
            rows,
            execution_error,
            execution_time,
        ) = execute_sql(
            sql_query=generated_sql,
            db_id=clean_db_id,
            dataset=dataset_key,
        )
    else:
        success = False
        columns = []
        rows = []
        execution_error = validation_error
        execution_time = 0.0

    exact_match = (
        None
        if not gold_sql
        else int(
            normalise_sql(generated_sql)
            == normalise_sql(gold_sql)
        )
    )

    return {
        "dataset": dataset_key,
        "strategy": strategy,
        "db_id": clean_db_id,
        "postgres_schema": postgres_schema_name(
            dataset_key,
            clean_db_id,
        ),
        "question": question,
        "evidence": evidence or "",
        "schema_context": schema_context,
        "retrieved_tables": retrieved_tables,
        "generated_sql": generated_sql,
        "success": int(success),
        "columns": columns,
        "rows": rows[:MAX_RESULT_ROWS],
        "error": execution_error,
        "generation_latency": generation_latency,
        "execution_time": execution_time,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "exact_match": exact_match,
        "execution_accuracy": execution_accuracy(
            predicted_sql=generated_sql,
            gold_sql=gold_sql,
            db_id=clean_db_id,
            dataset=dataset_key,
        ),
        "demo_mode": demo_mode,
    }


# ============================================================
# Benchmark question loading
# ============================================================

def discover_question_file(
    dataset: str = "bird",
) -> Path | None:
    """Find a local BIRD or Spider benchmark question file."""

    dataset_key = normalise_dataset(dataset)

    if dataset_key == "bird":
        root = Path(config.BIRD_DATA_DIR)
        preferred = [
            root / "mini_dev_sqlite.json",
            root / "dev.json",
            root / "mini_dev_data" / "mini_dev_sqlite.json",
        ]
    else:
        root = Path(config.SPIDER_DATA_DIR)
        preferred = [
            root / "dev.json",
            root / "train_spider.json",
            root / "train_others.json",
        ]

    for path in preferred:
        if path.exists():
            return path

    excluded_tokens = (
        "predict",
        "submission",
        "output",
        "result",
        "exp_result",
        "ta_output",
        "tables",
    )

    candidates: list[tuple[int, Path]] = []

    for path in root.rglob("*.json"):
        lowered_path = str(path).lower()

        if any(
            token in lowered_path
            for token in excluded_tokens
        ):
            continue

        try:
            payload = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

            records = (
                payload
                if isinstance(payload, list)
                else payload.get("data", [])
                if isinstance(payload, dict)
                else []
            )

            if not records or not isinstance(
                records[0],
                dict,
            ):
                continue

            keys = {
                str(key).lower()
                for key in records[0]
            }

            score = (
                20 * int("question" in keys)
                + 20 * int(
                    "db_id" in keys
                    or "database_id" in keys
                )
                + 20 * int(
                    bool(
                        {
                            "sql",
                            "query",
                            "gold_sql",
                        }
                        & keys
                    )
                )
            )

            candidates.append(
                (score, path)
            )

        except Exception:
            continue

    return max(
        candidates,
        default=(0, None),
        key=lambda item: item[0],
    )[1]


def load_questions(
    limit: int | None = None,
    db_id: str | None = None,
    dataset: str = "bird",
) -> list[dict[str, Any]]:
    """Load benchmark questions for BIRD or Spider."""

    dataset_key = normalise_dataset(dataset)
    path = discover_question_file(dataset_key)

    if path is None:
        return []

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    records = (
        payload
        if isinstance(payload, list)
        else payload.get("data", [])
        if isinstance(payload, dict)
        else []
    )

    output: list[dict[str, Any]] = []

    for index, item in enumerate(records):
        if not isinstance(item, dict):
            continue

        item_db_id = str(
            item.get("db_id")
            or item.get("database_id")
            or item.get("database")
            or ""
        )

        if db_id and item_db_id != db_id:
            continue

        question = str(
            item.get("question")
            or item.get("utterance")
            or item.get("text")
            or ""
        )

        if not item_db_id or not question:
            continue

        output.append(
            {
                "question_id": str(
                    item.get(
                        "question_id",
                        item.get("id", index),
                    )
                ),
                "dataset": dataset_key,
                "db_id": item_db_id,
                "question": question,
                "gold_sql": (
                    item.get("SQL")
                    or item.get("sql")
                    or item.get("query")
                    or item.get("gold_sql")
                    or ""
                ),
                "evidence": item.get(
                    "evidence",
                    "",
                ),
                "difficulty": item.get(
                    "difficulty",
                    item.get(
                        "question_type",
                        "",
                    ),
                ),
                "source_file": str(path),
            }
        )

        if limit and len(output) >= int(limit):
            break

    return output


# ============================================================
# Project status and schema statistics
# ============================================================

def project_status() -> dict[str, Any]:
    """Return project and data health information."""

    connected = database_health()
    schemas = list_schemas() if connected else []

    bird_questions = load_questions(
        dataset="bird"
    )

    spider_questions = load_questions(
        dataset="spider"
    )

    return {
        "database_connected": connected,
        "bird_questions": len(
            bird_questions
        ),
        "spider_questions": len(
            spider_questions
        ),
        "questions": (
            len(bird_questions)
            + len(spider_questions)
        ),
        "bird_question_file": (
            bird_questions[0]["source_file"]
            if bird_questions
            else None
        ),
        "spider_question_file": (
            spider_questions[0]["source_file"]
            if spider_questions
            else None
        ),
        "schemas": schemas,
        "bird_schemas": len(
            [
                schema
                for schema in schemas
                if schema.startswith("bird_")
            ]
        ),
        "spider_schemas": len(
            [
                schema
                for schema in schemas
                if schema.startswith("spider_")
            ]
        ),
        "openai_configured": bool(
            OPENAI_API_KEY
        ),
        "output_dir": str(
            config.OUTPUT_DIR
        ),
        "bird_data_dir": str(
            config.BIRD_DATA_DIR
        ),
        "spider_data_dir": str(
            config.SPIDER_DATA_DIR
        ),
    }


def schema_statistics(
    dataset: str | None = None,
) -> list[dict[str, Any]]:
    """Count tables and rows in migrated PostgreSQL schemas."""

    statistics: list[dict[str, Any]] = []

    with engine.connect() as connection:
        for schema in list_schemas(dataset):
            table_rows = connection.execute(
                text(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = :schema
                      AND table_type = 'BASE TABLE'
                    ORDER BY table_name
                    """
                ),
                {"schema": schema},
            ).fetchall()

            table_names = [
                str(row[0])
                for row in table_rows
            ]

            total_rows = 0

            for table_name in table_names:
                escaped_schema = schema.replace(
                    '"',
                    '""',
                )
                escaped_table = table_name.replace(
                    '"',
                    '""',
                )

                try:
                    total_rows += int(
                        connection.execute(
                            text(
                                f'SELECT COUNT(*) '
                                f'FROM "{escaped_schema}".'
                                f'"{escaped_table}"'
                            )
                        ).scalar_one()
                    )
                except Exception:
                    continue

            if schema.startswith("bird_"):
                schema_dataset = "bird"
                db_id = schema[5:]
            elif schema.startswith("spider_"):
                schema_dataset = "spider"
                db_id = schema[7:]
            else:
                schema_dataset = "unknown"
                db_id = schema

            statistics.append(
                {
                    "dataset": schema_dataset,
                    "db_id": db_id,
                    "schema": schema,
                    "tables": len(table_names),
                    "rows": total_rows,
                }
            )

    return statistics


def schema_statistics_frame(
    dataset: str | None = None,
) -> pd.DataFrame:
    """Return PostgreSQL schema statistics as a DataFrame."""

    return pd.DataFrame(
        schema_statistics(dataset)
    )
