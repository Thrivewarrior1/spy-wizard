"""Seed-once persistence tests.

Bug the user reported: deleting a competitor store on the app brings
it back after the next Railway redeploy. Root cause was `seed_stores()`
running its INITIAL_STORES upsert on every startup. Fix: gate the seed
behind `Store.count() == 0` so it only runs on a fresh DB. Once the
table has any rows, the user's catalog is the source of truth and the
seed is a no-op.

Cascade: deleting a Store must also drop its Products and their
PositionHistory rows. The DB-level ON DELETE CASCADE FK + ORM-level
cascade='all, delete-orphan' both run; tests cover the ORM path that
the API endpoint uses.
"""
from datetime import datetime

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
import sqlite3
import re as _re

from models import Base, Store, Product, PositionHistory
from seed import seed_stores, INITIAL_STORES


@pytest.fixture
def db_engine():
    eng = create_engine("sqlite:///:memory:")
    # Mirror the production hook so SQLite enforces FK cascades.
    @event.listens_for(eng, "connect")
    def _enable_fk(dbapi_conn, conn_record):
        if isinstance(dbapi_conn, sqlite3.Connection):
            dbapi_conn.execute("PRAGMA foreign_keys = ON")
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture
def db(db_engine):
    Session = sessionmaker(bind=db_engine)
    s = Session()
    yield s
    s.close()


def test_seed_stores_runs_only_on_empty_db(db):
    """First call seeds INITIAL_STORES. Second call is a no-op even
    if a store has been deleted between calls — the user's catalog is
    authoritative once it exists."""
    inserted_first = seed_stores(db=db)
    assert inserted_first == len(INITIAL_STORES)
    assert db.query(Store).count() == len(INITIAL_STORES)

    # Second call must not re-insert anything.
    inserted_second = seed_stores(db=db)
    assert inserted_second == 0
    assert db.query(Store).count() == len(INITIAL_STORES)


def test_deleted_store_stays_deleted_after_seed_runs_again(db):
    """The exact bug: delete a seeded store, run seed again (simulating
    a Railway redeploy), confirm the deleted store does NOT come back."""
    seed_stores(db=db)
    target = db.query(Store).filter(Store.url == "https://novigood.com/").one()
    db.delete(target)
    db.commit()
    assert db.query(Store).filter(Store.url == "https://novigood.com/").first() is None

    seed_stores(db=db)  # simulate redeploy
    assert db.query(Store).filter(Store.url == "https://novigood.com/").first() is None
    assert db.query(Store).count() == len(INITIAL_STORES) - 1


def test_added_store_persists_across_seed_calls(db):
    """A user-added store must survive every subsequent seed call."""
    seed_stores(db=db)
    custom = Store(name="Wild Eye Vision", url="https://wild-eye-vision.com/",
                   monthly_visitors="60K", niche="General", country="USA, UK")
    db.add(custom)
    db.commit()
    custom_id = custom.id

    seed_stores(db=db)
    survived = db.query(Store).filter(Store.id == custom_id).one()
    assert survived.name == "Wild Eye Vision"
    assert survived.url == "https://wild-eye-vision.com/"


def test_deleting_store_cascades_to_products_and_history(db):
    """ORM cascade='all, delete-orphan' on Store→Product and
    Product→PositionHistory must drop dependent rows so the General /
    Fashion feeds and search index don't surface zombie data."""
    seed_stores(db=db)
    s = db.query(Store).filter(Store.url == "https://novigood.com/").one()

    # Seed some products + history under this store.
    p1 = Product(
        store_id=s.id, shopify_id="x1", title="X1", handle="x1",
        image_url="", price="", vendor="", product_type="", product_url="",
        current_position=1, previous_position=0, label="",
        ai_tags="", is_fashion=True, subniche="fashion",
        last_scraped=datetime.utcnow(),
    )
    p2 = Product(
        store_id=s.id, shopify_id="x2", title="X2", handle="x2",
        image_url="", price="", vendor="", product_type="", product_url="",
        current_position=2, previous_position=0, label="",
        ai_tags="", is_fashion=False, subniche="home",
        last_scraped=datetime.utcnow(),
    )
    db.add(p1); db.add(p2); db.commit()
    db.add(PositionHistory(product_id=p1.id, position=1, date=datetime.utcnow()))
    db.add(PositionHistory(product_id=p2.id, position=2, date=datetime.utcnow()))
    db.commit()

    assert db.query(Product).filter(Product.store_id == s.id).count() == 2
    assert db.query(PositionHistory).count() == 2

    db.delete(s)
    db.commit()

    assert db.query(Store).filter(Store.url == "https://novigood.com/").first() is None
    assert db.query(Product).filter(Product.store_id == s.id).count() == 0
    assert db.query(PositionHistory).count() == 0


def test_seed_idempotent_with_partial_user_catalog(db):
    """A DB containing only a user-added store (no INITIAL_STORES rows)
    is still considered 'populated' — the seed must NOT add the
    INITIAL_STORES alongside it."""
    db.add(Store(name="Custom", url="https://custom.example/",
                 monthly_visitors="1K", niche="Fashion", country="DE"))
    db.commit()

    inserted = seed_stores(db=db)
    assert inserted == 0
    assert db.query(Store).count() == 1
    assert db.query(Store).filter(Store.url == "https://custom.example/").one() is not None
