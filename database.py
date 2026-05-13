import os
import re
import sqlite3
import logging
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./spy_wizard.db")

# Railway uses postgres:// but SQLAlchemy needs postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)


# Wire SQLite REGEXP + ON DELETE CASCADE so search uses the same
# word-boundary semantics as Postgres in tests, and ORM cascade
# deletes are enforced at the DB layer (not just the relationship
# level — important for raw-SQL deletes and for keeping orphans out
# of the index after a Store is dropped).
@event.listens_for(engine, "connect")
def _sqlite_connect_hooks(dbapi_conn, conn_record):
    if not isinstance(dbapi_conn, sqlite3.Connection):
        return
    # Enable foreign-key cascades on SQLite (off by default).
    dbapi_conn.execute("PRAGMA foreign_keys = ON")
    # Register REGEXP so column.op('REGEXP')(pat) works in tests.
    def _regexp(pattern, value):
        if value is None:
            return False
        try:
            return re.search(pattern, value, re.IGNORECASE) is not None
        except re.error:
            return False
    dbapi_conn.create_function("REGEXP", 2, _regexp)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def widen_text_columns():
    """Idempotent migration: convert handle/shopify_id/title columns from
    fixed-length VARCHAR to TEXT on existing Postgres deployments. Without
    this, long German/Dutch handles (>100 chars) trigger StringDataRightTruncation
    and roll back the entire scrape transaction.

    Safe to call on every startup; ALTER COLUMN ... TYPE TEXT is a no-op
    when the column is already TEXT. Skipped on SQLite (no-op there)."""
    if not DATABASE_URL.startswith(("postgresql://", "postgres://")):
        return
    statements = [
        "ALTER TABLE products ALTER COLUMN shopify_id TYPE TEXT",
        "ALTER TABLE products ALTER COLUMN handle TYPE TEXT",
        "ALTER TABLE products ALTER COLUMN title TYPE TEXT",
        # General-tab subniche classification. ADD COLUMN IF NOT EXISTS so
        # this is idempotent across redeploys.
        "ALTER TABLE products ADD COLUMN IF NOT EXISTS subniche VARCHAR(50) DEFAULT ''",
        # Fine-grained product category for the hybrid search. Populated
        # by assign_product_category() on every scrape and as a startup
        # backfill so existing rows get categorised without waiting for
        # the next scrape cycle.
        "ALTER TABLE products ADD COLUMN IF NOT EXISTS product_category VARCHAR(64) DEFAULT ''",
        "CREATE INDEX IF NOT EXISTS ix_products_product_category ON products(product_category)",
        # Persistent hero/villain event log. Table created via
        # Base.metadata.create_all on first boot; these statements are
        # idempotent guards for the indexes the model declares so
        # existing deployments pick them up after the schema change
        # without needing a separate Alembic migration.
        "CREATE INDEX IF NOT EXISTS ix_label_events_store_date ON label_events(store_id, date)",
        "CREATE INDEX IF NOT EXISTS ix_label_events_product_date ON label_events(product_id, date)",
        "CREATE INDEX IF NOT EXISTS ix_label_events_label_date ON label_events(label, date)",
    ]
    with engine.connect() as conn:
        for stmt in statements:
            try:
                conn.execute(text(stmt))
                conn.commit()
                logger.info(f"Migration OK: {stmt}")
            except Exception as e:
                # If table doesn't exist yet (fresh DB before create_all)
                # or column already TEXT, this is harmless. Log and continue.
                logger.info(f"Migration skipped ({stmt}): {e}")
