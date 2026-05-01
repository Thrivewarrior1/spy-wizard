"""Hero/villain computation tests.

Source of truth for movement labels is PositionHistory, NOT the
deprecated Product.label column. Today's snapshot is the latest history
row; the prior snapshot is the most recent row dated < UTC midnight of
today AND on/after TRUST_EPOCH_UTC. Day-over-day delta. New products
(no row qualifying as a trustworthy prior) are explicitly labelled
"new" — never "hero" or "villain". A delta exceeding the per-store
threshold (min of 30 ranks or 30% of catalog size) is suppressed and
labelled "normal" instead of hero/villain.
"""
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import main
from models import Base, Store, Product, PositionHistory


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture(autouse=True)
def lenient_trust_epoch(monkeypatch):
    """Default tests to a far-past trust epoch so existing date-based
    fixtures (YESTERDAY etc.) aren't accidentally filtered out by the
    production default. Tests that specifically exercise the trust
    epoch override TRUST_EPOCH_UTC inside the test body."""
    monkeypatch.setattr(main, "TRUST_EPOCH_UTC", datetime(2000, 1, 1))
    yield


@pytest.fixture
def store(db):
    s = Store(name="Test Store", url="https://test.example/", monthly_visitors="1K",
              niche="Fashion", country="DE")
    db.add(s)
    db.commit()
    db.refresh(s)
    # Pad catalog with 20 filler fashion products. The per-store delta
    # threshold is min(30, 30% of catalog) — with 21 fashion rows the
    # threshold is min(30, 6) = 6, comfortably above the small deltas
    # (<=5) that the legacy hero/villain tests use.
    now = datetime.utcnow()
    for i in range(20):
        db.add(Product(
            store_id=s.id, shopify_id=f"_pad-{i}", title=f"Filler {i}",
            handle=f"_pad-{i}", current_position=200 + i, previous_position=0,
            label="", is_fashion=True, subniche="fashion",
            last_scraped=now,
        ))
    db.commit()
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
    m = main._compute_label_map(db, [product])
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

    m = main._compute_label_map(db, products)
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


# =====================================================================
# Trust epoch + same-day priors + delta-magnitude sanity-check tests.
# These cover the fixes that drop the spurious heroes/villains the
# user complained about (e.g. 26 heroes / 0 villains right after a
# breaking change to the catalog shape).
# =====================================================================


def test_pre_epoch_snapshot_ignored(db, store, monkeypatch):
    """Snapshots dated before TRUST_EPOCH_UTC are not trustworthy
    comparators — they reflect a different catalog shape (different
    cap, different hard-drop regex, different schema). Such priors
    must be invisible to the hero/villain logic, so the product
    shows up as 'new' even though a (pre-epoch) prior exists."""
    # Trust epoch = a few hours ago; YESTERDAY's snapshot is well
    # before the epoch and must not count.
    epoch = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    monkeypatch.setattr(main, "TRUST_EPOCH_UTC", epoch)
    p = _add_product(db, store, "pre-epoch", [(YESTERDAY, 5)], current_pos=2)
    label, change, prior = _label(db, p)
    assert label == "new"
    assert prior == 0


def test_same_calendar_day_prior_ignored(db, store):
    """A 12-hour backup scrape or a manual same-day re-scrape can
    write multiple snapshots on one UTC calendar day. None of them
    qualify as a 'prior' — the prior must be from a DIFFERENT UTC
    calendar day. With only same-day priors we expect 'new'."""
    # 30 minutes ago (same UTC day as today, after midnight)
    earlier_today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(minutes=30)
    p = _add_product(db, store, "same-day", [(earlier_today, 12)], current_pos=4)
    label, _, prior = _label(db, p)
    assert label == "new"
    assert prior == 0


def test_large_delta_suppressed_to_normal(db, store):
    """A 50-rank jump (e.g. 55 → 5) is not organic movement; it's
    almost certainly a structural reshuffle. The label MUST collapse
    to 'normal' instead of being surfaced as a hero. Delta of 3
    (within the 6-rank threshold for our 21-product catalog) still
    gets the hero label as a control."""
    suspect = _add_product(db, store, "fast-mover", [(YESTERDAY, 55)], current_pos=5)
    sane = _add_product(db, store, "small-mover", [(YESTERDAY, 5)], current_pos=2)

    s_label, s_change, s_prior = _label(db, suspect)
    assert s_label == "normal", f"50-rank delta should be suppressed, got {s_label}"
    assert s_change == 0
    assert s_prior == 55

    c_label, c_change, _ = _label(db, sane)
    assert c_label == "hero"
    assert c_change == 3


def test_no_trustworthy_prior_means_new_or_normal(db, store, monkeypatch):
    """The user spec: 'with the trust epoch set to today, AND
    same-day priors excluded, AND no snapshots from prior days yet,
    every product's label MUST be new or normal. Heroes/villains
    MUST be 0/0 until tomorrow's daily scrape produces a clean
    prior-day baseline.' This test pins that combined behaviour."""
    epoch = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    monkeypatch.setattr(main, "TRUST_EPOCH_UTC", epoch)

    today_now = datetime.utcnow()
    products = []
    # Mix: brand-new (no history), pre-epoch prior, same-day prior.
    products.append(_add_product(db, store, "fresh", [], current_pos=1))
    products.append(_add_product(db, store, "pre", [(YESTERDAY, 8)], current_pos=2))
    products.append(_add_product(db, store, "today", [(today_now, 4)], current_pos=4))

    m = main._compute_label_map(db, products)
    labels = sorted(m[p.id][0] for p in products)
    assert all(lbl in {"new", "normal"} for lbl in labels), labels
    assert "hero" not in labels
    assert "villain" not in labels


def test_one_trustworthy_prior_day_normal_logic(db, store):
    """Once a single trustworthy prior-day snapshot exists, normal
    hero/villain logic applies — hero on improvement, villain on
    decline, normal on no movement, with the per-store delta cap
    suppressing implausible reshuffles."""
    hero = _add_product(db, store, "h", [(YESTERDAY, 4)], current_pos=2)
    villain = _add_product(db, store, "v", [(YESTERDAY, 2)], current_pos=5)
    flat = _add_product(db, store, "f", [(YESTERDAY, 7)], current_pos=7)

    m = main._compute_label_map(db, [hero, villain, flat])
    assert m[hero.id][0] == "hero"
    assert m[villain.id][0] == "villain"
    assert m[flat.id][0] == "normal"
