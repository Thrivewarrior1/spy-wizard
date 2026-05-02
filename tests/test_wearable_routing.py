"""Wearable subniche routing tests.

User spec: jewelry, accessories, and bags belong on the Fashion tab —
NOT on the General tab. The classifier may flag is_fashion=true on
its own, but as a safety net the scraper distribution code reconciles
any wearable subniche (jewelry/accessories/bags/fashion) to
is_fashion=True even when Gemini disagrees, and the per-feed upsert
preserves the wearable subniche so search expansion still works.

There's also a one-shot startup migration that promotes legacy
General-feed jewelry/accessories/bags rows to Fashion so the change
takes effect without waiting for tomorrow's scrape.
"""
from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models import Base, Store, Product
from scraper import (
    WEARABLE_SUBNICHES,
    update_products_in_db,
    migrate_wearables_to_fashion,
)


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
    s = Store(name="Test", url="https://test.example/", monthly_visitors="1K",
              niche="Fashion", country="DE")
    db.add(s)
    db.commit()
    db.refresh(s)
    return s


def _fashion_payload(handle, *, position, subniche="fashion"):
    return {
        "shopify_id": handle, "title": handle.replace("-", " ").title(),
        "handle": handle, "image_url": "", "price": "10.00",
        "vendor": "", "product_type": "", "ai_tags": "",
        "product_url": f"https://x/products/{handle}",
        "subniche": subniche, "position": position,
    }


def test_wearable_subniches_set_contains_expected_labels():
    """Spec pins exactly the four wearable categories so the rest of
    the codebase can rely on it. Adding 'electronics' here would be a
    bug — General-tab categories must NOT leak in."""
    assert WEARABLE_SUBNICHES == {"fashion", "bags", "accessories", "jewelry"}


def test_jewelry_item_keeps_subniche_jewelry_when_classified_fashion(db, store):
    """Gemini sometimes returns subniche='jewelry' on a jewelry item;
    update_products_in_db must preserve that subniche on the Fashion
    feed (used by backend search) instead of overwriting with
    'fashion'."""
    update_products_in_db(
        db, store,
        fashion_products=[_fashion_payload("gold-loop-earrings",
                                           position=1, subniche="jewelry")],
        general_products=[],
    )
    p = db.query(Product).filter(Product.shopify_id == "gold-loop-earrings").one()
    assert p.is_fashion is True
    assert p.subniche == "jewelry"


def test_accessories_item_keeps_subniche_accessories(db, store):
    update_products_in_db(
        db, store,
        fashion_products=[_fashion_payload("wool-scarf",
                                           position=1, subniche="accessories")],
        general_products=[],
    )
    p = db.query(Product).filter(Product.shopify_id == "wool-scarf").one()
    assert p.is_fashion is True
    assert p.subniche == "accessories"


def test_general_feed_rejects_wearable_subniche(db, store):
    """Defensive: if Gemini somehow puts a 'jewelry' item into the
    general bucket (is_fashion=false), the upsert must NOT store it
    under a wearable label. It buckets as 'other' instead so the
    General-feed query (subniche != '') doesn't surface it as
    jewelry on the General tab."""
    update_products_in_db(
        db, store,
        fashion_products=[],
        general_products=[_fashion_payload("misclassified-ring",
                                           position=1, subniche="jewelry")],
    )
    p = db.query(Product).filter(Product.shopify_id == "misclassified-ring").one()
    assert p.is_fashion is False
    assert p.subniche == "other"


def test_migrate_wearables_promotes_legacy_general_rows(db, store):
    """Existing rows previously stored as is_fashion=False with
    subniche in {jewelry, accessories, bags} must be promoted to
    is_fashion=True on startup so they appear on the Fashion tab
    immediately rather than waiting for tomorrow's scrape."""
    now = datetime.utcnow()
    legacy = [
        ("legacy-earring", "jewelry"),
        ("legacy-belt", "accessories"),
        ("legacy-tote", "bags"),
    ]
    keep_general = [
        ("legacy-lamp", "home"),
        ("legacy-phone", "electronics"),
    ]
    for handle, sub in legacy + keep_general:
        db.add(Product(
            store_id=store.id, shopify_id=handle, title=handle, handle=handle,
            image_url="", price="", vendor="", product_type="", product_url="",
            current_position=1, previous_position=0, label="",
            ai_tags="", is_fashion=False, subniche=sub, last_scraped=now,
        ))
    db.commit()

    promoted = migrate_wearables_to_fashion(db)
    assert promoted == 3

    for handle, _ in legacy:
        p = db.query(Product).filter(Product.shopify_id == handle).one()
        assert p.is_fashion is True, f"{handle} should be promoted"

    for handle, _ in keep_general:
        p = db.query(Product).filter(Product.shopify_id == handle).one()
        assert p.is_fashion is False, f"{handle} should stay on General"


def test_migrate_wearables_is_idempotent(db, store):
    """Running the migration twice should not move anything the
    second time — guards startup hooks from doing duplicate work."""
    now = datetime.utcnow()
    db.add(Product(
        store_id=store.id, shopify_id="x", title="x", handle="x",
        image_url="", price="", vendor="", product_type="", product_url="",
        current_position=1, previous_position=0, label="",
        ai_tags="", is_fashion=False, subniche="jewelry", last_scraped=now,
    ))
    db.commit()
    assert migrate_wearables_to_fashion(db) == 1
    assert migrate_wearables_to_fashion(db) == 0
