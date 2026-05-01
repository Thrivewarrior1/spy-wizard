"""Hero/villain computation tests.

Source of truth for movement labels is PositionHistory, NOT the
deprecated Product.label column. Today's snapshot is the latest history
row; the prior snapshot is the most recent row dated < UTC midnight of
today. Day-over-day delta. New products (no row dated < today_start)
are explicitly labelled "new" — never "hero" or "villain".
"""
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models import Base, Store, Product, PositionHistory


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def store(db):
    s = Store(name="Test Store", url="https://test.example/", monthly_visitors="1K",
              niche="Fashion", country="DE")
    db.add(s)
    db.commit()
    db.refresh(s)
    return s


def _add_product(db, store, shopify_id, history, current_pos, *, is_fashion=True, subniche="fashion"):
    """history: list of (date, position) PositionHistory rows to seed."""
    p = Product(
        store_id=store.id, shopify_id=shopify_id, title=shopify_id,
        handle=shopify_id, current_position=current_pos, previous_position=0,
        label="", is_fashion=is_fashion, subniche=subniche,
        last_scraped=datetime.utcnow(),
    )
    db.add(p)
    db.commit()
    db.refresh(p)
    for d, pos in history:
        db.add(PositionHistory(product_id=p.id, position=pos, date=d))
    db.commit()
    return p


def _label(db, product):
    from main import _compute_label_map
    m = _compute_label_map(db, [product])
    return m[product.id]  # (label, position_change, prior_position)


YESTERDAY = datetime.utcnow() - timedelta(days=1)
TWO_DAYS_AGO = datetime.utcnow() - timedelta(days=2)
FIVE_DAYS_AGO = datetime.utcnow() - timedelta(days=5)


def test_hero_when_moved_up(db, store):
    """#5 yesterday → #2 today = hero, change +3."""
    p = _add_product(db, store, "h1", [(YESTERDAY, 5)], current_pos=2)
    label, change, prior = _label(db, p)
    assert label == "hero"
    assert change == 3
    assert prior == 5


def test_villain_when_moved_down(db, store):
    """#2 yesterday → #5 today = villain."""
    p = _add_product(db, store, "v1", [(YESTERDAY, 2)], current_pos=5)
    label, change, prior = _label(db, p)
    assert label == "villain"
    assert prior == 2
    # change is negative because cur > prior
    assert change == -3


def test_normal_when_position_unchanged(db, store):
    p = _add_product(db, store, "n1", [(YESTERDAY, 5)], current_pos=5)
    label, change, prior = _label(db, p)
    assert label == "normal"
    assert change == 0
    assert prior == 5


def test_new_when_no_prior_history(db, store):
    """No PositionHistory rows at all → 'new'."""
    p = _add_product(db, store, "new1", [], current_pos=10)
    label, change, prior = _label(db, p)
    assert label == "new"
    assert prior == 0


def test_new_when_only_today_history(db, store):
    """Only history rows from today (not before today_start) → 'new'."""
    today_now = datetime.utcnow()
    p = _add_product(db, store, "new2", [(today_now, 7)], current_pos=7)
    label, _, _ = _label(db, p)
    assert label == "new"


def test_uses_most_recent_prior_snapshot(db, store):
    """Multiple priors exist — only the most-recent one before today
    is used. Old #50 should be ignored once #10 lands yesterday."""
    p = _add_product(
        db, store, "mp1",
        [(FIVE_DAYS_AGO, 50), (YESTERDAY, 10)],
        current_pos=8,
    )
    label, change, prior = _label(db, p)
    assert label == "hero"
    assert prior == 10  # NOT 50
    assert change == 2


def test_dropped_products_are_filtered_at_query_time(db, store):
    """Products that disappeared from the latest scrape are flagged
    is_fashion=False by retirement logic; they don't show up in the
    Fashion feed query and therefore don't get labelled. We can still
    compute their label if asked directly — verify it reflects history."""
    p = _add_product(db, store, "dropped",
                     [(TWO_DAYS_AGO, 5), (YESTERDAY, 7)],
                     current_pos=7, is_fashion=False, subniche="")
    label, _, prior = _label(db, p)
    # Yesterday's row is the prior; current_position == prior so it's normal
    assert label == "normal"
    assert prior == 7


def test_no_prior_means_no_heroes_no_villains(db, store):
    """First-ever scrape: every product is brand-new, so heroes/villains
    must both be empty. Mirrors the user spec: 'If there is no prior
    snapshot at all yet, both lists must be empty.'"""
    products = []
    for i in range(5):
        p = _add_product(db, store, f"first-{i}", [], current_pos=i + 1)
        products.append(p)

    from main import _compute_label_map
    m = _compute_label_map(db, products)
    labels = [m[p.id][0] for p in products]
    assert all(lbl == "new" for lbl in labels), labels
    assert "hero" not in labels
    assert "villain" not in labels


def test_match_uses_product_id_not_title(db, store):
    """Two products with identical titles but different shopify_id
    must have INDEPENDENT label computations. Matching is by
    Product.id (which is bound 1:1 to shopify_id), never by title."""
    a = _add_product(db, store, "sku-a", [(YESTERDAY, 3)], current_pos=1)
    b = _add_product(db, store, "sku-b", [(YESTERDAY, 1)], current_pos=3)
    a.title = b.title = "Identical Title"
    db.commit()
    la, _, _ = _label(db, a)
    lb, _, _ = _label(db, b)
    assert la == "hero"     # 3 -> 1
    assert lb == "villain"  # 1 -> 3
