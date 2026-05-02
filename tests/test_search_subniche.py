"""Search-box wiring for the `subniche` backend metadata column.

The user's spec for Bug A: subniche is BACKEND-only metadata (Gemini
labels each non-fashion product with one of jewelry / accessories /
electronics / home / beauty / health / food / toys-books / other),
NOT a user-facing filter. Typing 'earring' into the search box must
return jewelry-tagged products even when the title is opaque
('Stud Studded Loops'). This is the regression guard.
"""
from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from main import (
    SUBNICHE_SYNONYMS,
    expand_single_term,
    build_ai_tag_filters,
    build_search_filters,
)
from models import Base, Store, Product


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
    s = Store(name="Gen", url="https://gen.example/", monthly_visitors="1K",
              niche="General", country="DE")
    db.add(s)
    db.commit()
    db.refresh(s)
    return s


def _add(db, store, *, title, subniche, ai_tags="", product_type="",
         is_fashion=False, current_position=1):
    p = Product(
        store_id=store.id, shopify_id=title.lower().replace(" ", "-"),
        title=title, handle=title.lower().replace(" ", "-"),
        product_type=product_type, ai_tags=ai_tags,
        is_fashion=is_fashion, subniche=subniche,
        current_position=current_position,
        last_scraped=datetime.utcnow(),
    )
    db.add(p)
    db.commit()
    db.refresh(p)
    return p


def test_earring_term_expands_to_jewelry_subniche():
    """Typing 'earring' must add 'jewelry' to the variant set so the
    SQL filter hits Product.subniche='jewelry'."""
    variants = expand_single_term("earring")
    assert "jewelry" in variants
    assert "earring" in variants


def test_phone_term_expands_to_electronics_subniche():
    variants = expand_single_term("phone")
    assert "electronics" in variants


def test_canonical_subniche_label_expands_to_synonyms():
    """Typing the canonical label itself ('jewelry') should also pull
    in the synonym list so we get matches in titles/ai_tags too."""
    variants = expand_single_term("jewelry")
    assert "earring" in variants
    assert "necklace" in variants


def test_pouch_term_expands_to_bags_subniche():
    """Bags are now on the Fashion tab with subniche='bags'. Typing
    'pouch' should match bags-tagged products via reverse-lookup."""
    variants = expand_single_term("pouch")
    assert "bags" in variants


def test_handbag_term_expands_to_bags_subniche():
    variants = expand_single_term("handbag")
    assert "bags" in variants


def test_search_matches_product_with_only_subniche_set(db, store):
    """An opaque-titled product whose only category info is in the
    subniche column must surface for a related search term."""
    studded = _add(db, store, title="Stud Studded Loops", subniche="jewelry")
    lamp = _add(db, store, title="Wall Lamp", subniche="home")

    conds = build_ai_tag_filters("earring")
    q = db.query(Product)
    for c in conds:
        q = q.filter(c)
    hits = {p.id for p in q.all()}
    assert studded.id in hits, "earring search must match jewelry-tagged product"
    assert lamp.id not in hits


def test_search_for_subniche_matches_via_title_too(db, store):
    """Searching 'jewelry' must hit anything tagged subniche=jewelry,
    AND also match products that mention a synonym (e.g. 'earring')
    in the title, even when their subniche column is empty."""
    a = _add(db, store, title="Stud Studded Loops", subniche="jewelry")
    b = _add(db, store, title="Daily Earring Hoops", subniche="")
    c = _add(db, store, title="Wall Lamp", subniche="home")

    strict, _ = build_search_filters("jewelry")
    q = db.query(Product)
    for cond in strict:
        q = q.filter(cond)
    hits = {p.id for p in q.all()}
    assert a.id in hits
    assert b.id in hits
    assert c.id not in hits


def test_subniche_synonyms_cover_all_taggable_subniche_labels():
    """Every Gemini subniche label that we expect on real data should
    have at least one English synonym registered, otherwise typing
    the obvious keyword for that category wouldn't match. Includes
    the wearable categories now on Fashion (bags / accessories /
    jewelry) plus the General-feed categories. 'fashion' is the
    catch-all clothing/shoes label and 'other' is intentionally
    empty, so neither needs synonyms here."""
    expected = {
        "bags", "jewelry", "accessories", "electronics", "home",
        "beauty", "health", "food", "toys-books",
    }
    missing = [k for k in expected if not SUBNICHE_SYNONYMS.get(k)]
    assert not missing, f"missing synonyms for: {missing}"
