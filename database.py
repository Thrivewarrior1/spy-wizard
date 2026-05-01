import os
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./spy_wizard.db")

# Railway uses postgres:// but SQLAlchemy needs postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)
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
