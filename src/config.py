# ------------------------------------------------------------------
# PostgreSQL configuration
# ------------------------------------------------------------------

DB_HOST = os.getenv("DB_HOST") or "localhost"
DB_NAME = os.getenv("DB_NAME") or "text2sql_rag"
DB_USER = os.getenv("DB_USER") or "postgres"
DB_PASSWORD = os.getenv("DB_PASSWORD") or ""

db_port_value = (os.getenv("DB_PORT") or "5432").strip()

try:
    DB_PORT = int(db_port_value)
except ValueError:
    DB_PORT = 5432


def normalise_database_url(value: str) -> str:
    """Prepare a PostgreSQL URL for SQLAlchemy."""

    url = value.strip().strip('"').strip("'")

    if url.startswith("postgres://"):
        url = "postgresql://"+url[len("postgres://"):]

    # Use the installed psycopg2-binary driver.
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg2://" + url[len("postgresql://"):]

    return url


render_database_url = (
    os.getenv("SQLALCHEMY_URL")
    or os.getenv("DATABASE_URL")
    or ""
).strip()

if render_database_url:
    DATABASE_URL = normalise_database_url(render_database_url)
else:
    DATABASE_URL = (
        f"postgresql+psycopg2://"
        f"{DB_USER}:{DB_PASSWORD}@"
        f"{DB_HOST}:{DB_PORT}/"
        f"{DB_NAME}"
    )

# core.py checks this variable first.
SQLALCHEMY_URL = DATABASE_URL
