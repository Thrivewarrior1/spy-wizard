"""Tests for cleanup_non_product_rows — the safety net that purges
checkout-add-on rows from the DB on startup and after every scrape,
even when the per-feed retirement threshold isn't met.
"""
from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models import Base, Store, Product
from scraper import cleanup_non_product_rows


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.close()


@pytest.fixture
def store(db):
    s = Store(name="T", url="https://t.example/", monthly_visitors="1K", niche="Fashion", country="DE")
    db.add(s); db.commit(); db.refresh(s)
    return s


def _add(db, store, sid, title, *, is_fashion=True, subniche="fashion", ptype=""):
    p = Product(
        store_id=store.id, shopify_id=sid, title=title, handle=sid,
        current_position=1, previous_position=0, label="",
        is_fashion=is_fashion, subniche=subniche, product_type=ptype,
        last_scraped=datetime.utcnow(),
    )
    db.add(p); db.commit(); db.refresh(p)
    return p


def test_purges_known_junk_rows(db, store):
    """Stuck services rows from before the regex tightened must be
    cleared on the next cleanup pass — even though they were never
    in any current scrape's general list."""
    a = _add(db, store, "j1", "Shipping Protection",
             is_fashion=False, subniche="services")
    b = _add(db, store, "j2", "100% Coverage",
             is_fashion=False, subniche="services")
    c = _add(db, store, "j3", "Versicherter Versand",
             is_fashion=False, subniche="services")
    d = _add(db, store, "real", "Cotton T-Shirt",
             is_fashion=True, subniche="fashion")

    n = cleanup_non_product_rows(db)
    assert n == 3, "expected 3 junk rows purged, got %d" % n

    db.refresh(a); db.refresh(b); db.refresh(c); db.refresh(d)
    assert not a.is_fashion and a.subniche == ""
    assert not b.is_fashion and b.subniche == ""
    assert not c.is_fashion and c.subniche == ""
    # Real fashion product untouched
    assert d.is_fashion and d.subniche == "fashion"


def test_idempotent(db, store):
    _add(db, store, "j1", "Gift Card $50", is_fashion=False, subniche="services")
    first = cleanup_non_product_rows(db)
    second = cleanup_non_product_rows(db)
    assert first == 1
    assert second == 0  # already cleaned, nothing to purge


def test_purges_by_product_type(db, store):
    """If the title is benign but Shopify's product_type clearly
    flags it (e.g. 'Slidecart - Shipping Protection'), the row must
    still be purged."""
    p = _add(db, store, "j-ptype", "Some Title",
             is_fashion=False, subniche="services",
             ptype="Slidecart - Shipping Protection")
    n = cleanup_non_product_rows(db)
    assert n == 1
    db.refresh(p)
    assert not p.is_fashion and p.subniche == ""


def test_does_not_touch_legitimate_products(db, store):
    """Real products that contain protective wording in their titles
    (sun protection clothing, UV protection sunglasses, etc.) must NOT
    be purged."""
    a = _add(db, store, "r1", "Sun Protection Hat", subniche="fashion")
    b = _add(db, store, "r2", "UV400 Sunglasses Protection", is_fashion=False, subniche="accessories")
    c = _add(db, store, "r3", "Insured Card Holder Wallet", subniche="fashion")
    n = cleanup_non_product_rows(db)
    assert n == 0
    for p in (a, b, c):
        db.refresh(p)
    assert a.is_fashion and a.subniche == "fashion"
    assert b.subniche == "accessories"
    assert c.is_fashion and c.subniche == "fashion"
