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


def enforce_fk_cascade():
    """Idempotent migration: ensure every foreign key that the ORM
    declares with `ondelete='CASCADE'` actually has `ON DELETE CASCADE`
    enforced at the Postgres level.

    Without this, the DB was created at a time when the ORM models
    didn't yet declare CASCADE; SQLAlchemy's `Base.metadata.create_all`
    only creates missing tables and never re-issues FK constraints
    against existing tables. Result: deleting a Store left orphan
    Products / PositionHistory / LabelEvent rows behind, which
    continued to show in the dashboard and counted toward stats.

    This migration:
      1. Queries pg_constraint to find each FK on the four relevant
         columns and the actual `confdeltype` ('c' = CASCADE).
      2. For any FK whose deltype != 'c', DROPs and recreates it with
         ON DELETE CASCADE.
      3. Logs each fix so the Railway log shows exactly what changed.

    Skipped on SQLite (we already do `PRAGMA foreign_keys = ON` in the
    connect hook, and SQLite re-creates tables on every test run).
    """
    if not DATABASE_URL.startswith(("postgresql://", "postgres://")):
        return
    # (child_table, child_column, parent_table) — every FK we want to
    # be ON DELETE CASCADE. Order doesn't matter for the migration
    # itself but is grouped by parent for readability.
    cascades = [
        ("products", "store_id", "stores"),
        ("position_history", "product_id", "products"),
        ("label_events", "store_id", "stores"),
        ("label_events", "product_id", "products"),
    ]
    with engine.connect() as conn:
        for child_table, child_column, parent_table in cascades:
            try:
                # Find the existing constraint name + cascade type.
                row = conn.execute(text("""
                    SELECT conname, confdeltype
                    FROM pg_constraint
                    WHERE conrelid = (
                        SELECT oid FROM pg_class
                        WHERE relname = :child AND relnamespace = (
                            SELECT oid FROM pg_namespace WHERE nspname = current_schema()
                        )
                    )
                    AND contype = 'f'
                    AND conkey = (
                        SELECT array_agg(attnum ORDER BY attnum)
                        FROM pg_attribute
                        WHERE attrelid = (
                            SELECT oid FROM pg_class WHERE relname = :child
                        )
                        AND attname = :col
                    )
                """), {"child": child_table, "col": child_column}).fetchone()
                if not row:
                    logger.info(
                        "enforce_fk_cascade: no FK found on %s.%s yet "
                        "(table may not exist; skipping)",
                        child_table, child_column,
                    )
                    continue
                conname, deltype = row
                if deltype == "c":
                    logger.info(
                        "enforce_fk_cascade: %s.%s already CASCADE",
                        child_table, child_column,
                    )
                    continue
                # Drop + recreate with CASCADE.
                conn.execute(text(
                    f'ALTER TABLE {child_table} '
                    f'DROP CONSTRAINT "{conname}"'
                ))
                conn.execute(text(
                    f'ALTER TABLE {child_table} '
                    f'ADD CONSTRAINT "{conname}" '
                    f'FOREIGN KEY ({child_column}) '
                    f'REFERENCES {parent_table}(id) ON DELETE CASCADE'
                ))
                conn.commit()
                logger.info(
                    "enforce_fk_cascade: switched %s.%s -> %s "
                    "to ON DELETE CASCADE (was '%s')",
                    child_table, child_column, parent_table, deltype,
                )
            except Exception as e:
                # Don't crash startup over a single FK that wasn't there
                # yet (e.g. label_events on first deploy before the
                # table is created). Log and move on; next deploy will
                # retry once the table exists.
                logger.warning(
                    "enforce_fk_cascade: skipped %s.%s -> %s: %s",
                    child_table, child_column, parent_table, e,
                )


def cleanup_orphans():
    """One-shot, idempotent: delete any Product / PositionHistory /
    LabelEvent row whose parent FK no longer points at a real row.

    This belt-and-braces complement to enforce_fk_cascade catches
    rows that slipped through before the FK was upgraded — without
    it the user keeps seeing stale products from stores they
    deleted weeks ago. After enforce_fk_cascade has run once, this
    function returns 0 on every subsequent boot.

    Skipped on SQLite (the test suite re-creates tables each run).
    """
    if not DATABASE_URL.startswith(("postgresql://", "postgres://")):
        return {"products": 0, "position_history": 0, "label_events": 0}
    counts = {"products": 0, "position_history": 0, "label_events": 0}
    statements = [
        # PositionHistory orphaned by a missing Product.
        ("position_history", """
            DELETE FROM position_history
            WHERE product_id NOT IN (SELECT id FROM products)
        """),
        # LabelEvent orphaned by a missing Store OR a missing Product.
        # Run before products cleanup so we don't double-count rows
        # the next cascade step would have removed.
        ("label_events", """
            DELETE FROM label_events
            WHERE store_id NOT IN (SELECT id FROM stores)
               OR product_id NOT IN (SELECT id FROM products)
        """),
        # Product orphaned by a missing Store.
        ("products", """
            DELETE FROM products
            WHERE store_id NOT IN (SELECT id FROM stores)
        """),
    ]
    with engine.connect() as conn:
        for label, stmt in statements:
            try:
                result = conn.execute(text(stmt))
                conn.commit()
                counts[label] = result.rowcount or 0
                if counts[label]:
                    logger.info(
                        "cleanup_orphans: deleted %d orphan %s rows",
                        counts[label], label,
                    )
            except Exception as e:
                logger.warning(
                    "cleanup_orphans: skipped %s cleanup: %s", label, e,
                )
    return counts


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
        # Vision classification timestamp — populated by
        # image_classifier.classify_images_batch. Nullable so backfill
        # can skip already-processed rows.
        "ALTER TABLE products ADD COLUMN IF NOT EXISTS vision_classified_at TIMESTAMP NULL",
        "CREATE INDEX IF NOT EXISTS ix_products_vision_classified_at ON products (vision_classified_at)",
        # Natural-language description from the vision model. Read by
        # the strict search judge for semantic matching (no exact-tag
        # match required).
        "ALTER TABLE products ADD COLUMN IF NOT EXISTS vision_description TEXT NULL",
        # Semantic embedding vector (JSON float array) + the text that
        # produced it. Primary multilingual search-retrieval signal.
        "ALTER TABLE products ADD COLUMN IF NOT EXISTS embedding TEXT NULL",
        "ALTER TABLE products ADD COLUMN IF NOT EXISTS embedding_text TEXT NULL",
        # Feed-page indexes: the fashion / general / per-store feeds do
        # ORDER BY current_position + LIMIT. Without these the free-tier
        # Postgres full-scans + sorts 7000+ rows on every page load
        # (the slow-initial-load the user hit). Composite so the DB can
        # satisfy the filter AND the sort from one index.
        "CREATE INDEX IF NOT EXISTS ix_products_fashion_pos ON products(is_fashion, current_position)",
        "CREATE INDEX IF NOT EXISTS ix_products_store_pos ON products(store_id, current_position)",
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
