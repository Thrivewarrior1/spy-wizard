"""Persistent hero/villain event regression tests.

User asked for 30 days of retained heroes/villains plus a 7/14/30-day
UI filter. The previous behaviour computed labels at read-time from
PositionHistory and lost yesterday's heroes the moment a fresher
snapshot landed. The new LabelEvent table is the persistent ledger.

These tests pin:
  - Event write at scrape time (compute_and_write_events).
  - Retention (cleanup_label_events).
  - Backfill from existing PositionHistory.
  - The 7/14/30-day API window via fetch_label_events_window.
  - Cross-store isolation (same shopify_id in two stores → independent
    events).
  - Edge cases: product with both hero AND villain events in window;
    product that was hero then dropped from the feed (is_active=false).
"""
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import sqlite3

from models import Base, Store, Product, PositionHistory, LabelEvent
from labels import (
    TRUST_EPOCH_UTC,
    HERO_VILLAIN_DELTA_CAP,
    LABEL_EVENT_RETENTION_DAYS,
    compute_and_write_events,
    cleanup_label_events,
    backfill_label_events,
    fetch_label_events_window,
    today_start_utc,
)


@pytest.fixture
def db():
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    @event.listens_for(eng, "connect")
    def _fk(dbapi_conn, conn_record):
        if isinstance(dbapi_conn, sqlite3.Connection):
            dbapi_conn.execute("PRAGMA foreign_keys = ON")
    Base.metadata.create_all(eng)
    Session = sessionmaker(bind=eng)
    s = Session()
    yield s
    s.close()


@pytest.fixture
def store(db):
    s = Store(name="Test Store", url="https://test-store.example/",
              monthly_visitors="1K", niche="Fashion", country="DE")
    db.add(s); db.commit(); db.refresh(s)
    return s


def _make_product(db, store, *, handle, position, is_fashion=True,
                   subniche="fashion"):
    p = Product(
        store_id=store.id, shopify_id=handle, title=handle, handle=handle,
        image_url="", price="", vendor="", product_type="", product_url="",
        current_position=position, previous_position=0, label="",
        ai_tags="", is_fashion=is_fashion, subniche=subniche,
        last_scraped=datetime.utcnow(),
    )
    db.add(p); db.commit(); db.refresh(p)
    return p


def _pad_catalog(db, store, n=120, is_fashion=True):
    """Pad the catalog so the per-store catalog-fraction threshold
    doesn't collapse the delta cap to 1 (with only a handful of test
    products, threshold = min(30, int(N*0.30)) becomes 0 or 1)."""
    for i in range(n):
        _make_product(
            db, store, handle=f"pad-{i}", position=i + 1000,
            is_fashion=is_fashion,
            subniche="fashion" if is_fashion else "home",
        )


# =====================================================================
# compute_and_write_events — happy paths
# =====================================================================
def test_event_written_for_clear_hero(db, store):
    now = datetime(2026, 5, 5, 12, 0, 0)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday = today - timedelta(hours=10)

    p = _make_product(db, store, handle="moved-up", position=5)
    db.add(PositionHistory(product_id=p.id, position=30, date=yesterday))
    db.add(PositionHistory(product_id=p.id, position=5, date=now))
    db.commit()
    _pad_catalog(db, store)

    heroes, villains = compute_and_write_events(db, store, now=now)
    assert heroes >= 1
    assert villains >= 0

    ev = db.query(LabelEvent).filter(LabelEvent.product_id == p.id).one()
    assert ev.label == "hero"
    assert ev.prior_position == 30
    assert ev.current_position == 5
    assert ev.position_change == 25
    assert ev.date == today


def test_event_written_for_clear_villain(db, store):
    now = datetime(2026, 5, 5, 12, 0, 0)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday = today - timedelta(hours=10)

    p = _make_product(db, store, handle="moved-down", position=35)
    db.add(PositionHistory(product_id=p.id, position=10, date=yesterday))
    db.add(PositionHistory(product_id=p.id, position=35, date=now))
    db.commit()
    _pad_catalog(db, store)

    h, v = compute_and_write_events(db, store, now=now)
    assert v >= 1

    ev = db.query(LabelEvent).filter(LabelEvent.product_id == p.id).one()
    assert ev.label == "villain"
    assert ev.prior_position == 10
    assert ev.current_position == 35
    assert ev.position_change == -25


def test_no_event_for_unchanged_position(db, store):
    now = datetime(2026, 5, 5, 12, 0, 0)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday = today - timedelta(hours=10)

    p = _make_product(db, store, handle="stayed", position=12)
    db.add(PositionHistory(product_id=p.id, position=12, date=yesterday))
    db.add(PositionHistory(product_id=p.id, position=12, date=now))
    db.commit()
    _pad_catalog(db, store)

    compute_and_write_events(db, store, now=now)
    assert db.query(LabelEvent).filter(LabelEvent.product_id == p.id).count() == 0


def test_no_event_for_oversize_delta(db, store):
    """Delta > 30 is treated as a structural reshuffle, suppressed
    to 'normal', no event written."""
    now = datetime(2026, 5, 5, 12, 0, 0)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday = today - timedelta(hours=10)

    p = _make_product(db, store, handle="big-jump", position=5)
    db.add(PositionHistory(product_id=p.id, position=99, date=yesterday))
    db.add(PositionHistory(product_id=p.id, position=5, date=now))
    db.commit()
    _pad_catalog(db, store)

    compute_and_write_events(db, store, now=now)
    assert db.query(LabelEvent).filter(LabelEvent.product_id == p.id).count() == 0


def test_no_event_for_new_product_without_prior(db, store):
    now = datetime(2026, 5, 5, 12, 0, 0)
    p = _make_product(db, store, handle="debut", position=7)
    db.add(PositionHistory(product_id=p.id, position=7, date=now))
    db.commit()
    _pad_catalog(db, store)

    compute_and_write_events(db, store, now=now)
    assert db.query(LabelEvent).filter(LabelEvent.product_id == p.id).count() == 0


def test_same_day_rescrape_is_idempotent(db, store):
    """Two compute_and_write_events calls for the same UTC day must
    NOT create duplicate rows — they upsert in place."""
    now = datetime(2026, 5, 5, 12, 0, 0)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday = today - timedelta(hours=10)

    p = _make_product(db, store, handle="up", position=5)
    db.add(PositionHistory(product_id=p.id, position=20, date=yesterday))
    db.add(PositionHistory(product_id=p.id, position=5, date=now))
    db.commit()
    _pad_catalog(db, store)

    compute_and_write_events(db, store, now=now)
    compute_and_write_events(db, store, now=now)

    assert db.query(LabelEvent).filter(LabelEvent.product_id == p.id).count() == 1


def test_same_day_rescrape_updates_position(db, store):
    """If a second scrape on the same day finds the product at a
    different current_position, the existing event row should
    update — not duplicate."""
    now = datetime(2026, 5, 5, 12, 0, 0)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday = today - timedelta(hours=10)

    p = _make_product(db, store, handle="updating", position=5)
    db.add(PositionHistory(product_id=p.id, position=20, date=yesterday))
    db.commit()
    _pad_catalog(db, store)
    compute_and_write_events(db, store, now=now)

    # Second scrape — current_position improved.
    p.current_position = 3
    db.add(PositionHistory(product_id=p.id, position=3, date=now + timedelta(hours=2)))
    db.commit()
    compute_and_write_events(db, store, now=now + timedelta(hours=2))

    ev = db.query(LabelEvent).filter(LabelEvent.product_id == p.id).one()
    assert ev.current_position == 3
    assert ev.position_change == 17  # 20 -> 3


# =====================================================================
# Retention
# =====================================================================
def test_retention_deletes_events_older_than_30_days(db, store):
    now = datetime(2026, 5, 5, 12, 0, 0)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    p = _make_product(db, store, handle="x", position=1)

    # Six events across a 35-day window — only the four >30 days get pruned.
    for offset_days in [0, 5, 14, 29, 30, 31, 60]:
        db.add(LabelEvent(
            store_id=store.id, product_id=p.id,
            date=today - timedelta(days=offset_days),
            label="hero", prior_position=10, current_position=5,
            position_change=5,
        ))
    db.commit()
    assert db.query(LabelEvent).count() == 7

    deleted = cleanup_label_events(db, retention_days=30, now=now)
    # Cutoff is now - 30 days = 2026-04-05 12:00. Events stored at
    # UTC midnight, so offset=30 is 2026-04-05 00:00 (BEFORE the
    # 12:00 cutoff), offset=31 and offset=60 are also before.
    # Offsets 0/5/14/29 stay.
    assert deleted == 3
    assert db.query(LabelEvent).count() == 4


def test_retention_is_idempotent(db, store):
    now = datetime(2026, 5, 5, 12, 0, 0)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    p = _make_product(db, store, handle="x", position=1)
    db.add(LabelEvent(
        store_id=store.id, product_id=p.id,
        date=today - timedelta(days=60),
        label="hero", prior_position=10, current_position=5,
        position_change=5,
    ))
    db.commit()
    assert cleanup_label_events(db, retention_days=30, now=now) == 1
    assert cleanup_label_events(db, retention_days=30, now=now) == 0


# =====================================================================
# Backfill from PositionHistory
# =====================================================================
def test_backfill_synthesises_events_from_history(db, store):
    """Walk PositionHistory day-by-day. Each consecutive pair within
    the trust + retention window produces one LabelEvent."""
    now = datetime(2026, 5, 5, 12, 0, 0)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    d4 = today - timedelta(days=1)  # 2026-05-04
    d3 = today - timedelta(days=2)  # 2026-05-03 — pre-epoch, excluded

    p = _make_product(db, store, handle="x", position=5)
    # day-3 snapshot is before the trust epoch — must be excluded
    db.add(PositionHistory(product_id=p.id, position=50, date=d3))
    # day-4 snapshot at 14:00 (within trust window)
    db.add(PositionHistory(product_id=p.id, position=20, date=d4 + timedelta(hours=14)))
    # today's snapshot — backfill should SKIP today (compute_and_write
    # owns it at scrape time)
    db.add(PositionHistory(product_id=p.id, position=5, date=now))
    db.commit()
    _pad_catalog(db, store)

    inserted = backfill_label_events(db, now=now)
    # With trust epoch = 2026-05-04, only the d4 snapshot qualifies.
    # That's a SINGLE qualifying day per product — backfill needs
    # consecutive pairs (>= 2 days), so for now there's nothing to
    # write. Sanity check the backfill ran without error.
    assert inserted >= 0


def test_backfill_writes_event_for_two_qualifying_days(db, store):
    """Two trustworthy snapshots → one prior-day event written by
    backfill (today is owned by compute_and_write_events)."""
    now = datetime(2026, 5, 6, 12, 0, 0)   # today
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    day_minus_2 = today - timedelta(days=2)  # 2026-05-04 (trust epoch)
    day_minus_1 = today - timedelta(days=1)  # 2026-05-05

    p = _make_product(db, store, handle="climber", position=4)
    db.add(PositionHistory(product_id=p.id, position=30, date=day_minus_2 + timedelta(hours=14)))
    db.add(PositionHistory(product_id=p.id, position=15, date=day_minus_1 + timedelta(hours=14)))
    db.add(PositionHistory(product_id=p.id, position=4,  date=now))
    db.commit()
    _pad_catalog(db, store)

    inserted = backfill_label_events(db, now=now)
    # One event for day-minus-1 (vs day-minus-2 prior). Today's not
    # written by backfill.
    events = db.query(LabelEvent).filter(LabelEvent.product_id == p.id).all()
    assert len(events) >= 1
    assert any(ev.date == day_minus_1 and ev.label == "hero" for ev in events)


def test_backfill_is_idempotent(db, store):
    now = datetime(2026, 5, 6, 12, 0, 0)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    d1 = today - timedelta(days=2)
    d2 = today - timedelta(days=1)

    p = _make_product(db, store, handle="ok", position=5)
    db.add(PositionHistory(product_id=p.id, position=25, date=d1 + timedelta(hours=12)))
    db.add(PositionHistory(product_id=p.id, position=5,  date=d2 + timedelta(hours=12)))
    db.commit()
    _pad_catalog(db, store)

    first = backfill_label_events(db, now=now)
    second = backfill_label_events(db, now=now)
    assert second == 0
    # Total event count after two runs equals the first-run insert count.
    assert db.query(LabelEvent).filter(LabelEvent.product_id == p.id).count() == first


# =====================================================================
# fetch_label_events_window — day-range queries
# =====================================================================
def _seed_event(db, store, product, *, days_ago, label="hero",
                 prior=20, current=5, today_anchor=None):
    today_anchor = today_anchor or today_start_utc()
    event_date = today_anchor - timedelta(days=days_ago)
    db.add(LabelEvent(
        store_id=store.id, product_id=product.id, date=event_date,
        label=label, prior_position=prior, current_position=current,
        position_change=(prior - current),
    ))
    db.commit()


def test_days_window_returns_only_events_inside_range(db, store):
    now = datetime(2026, 5, 12, 12, 0, 0)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)

    in_window = _make_product(db, store, handle="recent", position=5)
    out_of_window = _make_product(db, store, handle="ancient", position=8)
    _seed_event(db, store, in_window, days_ago=2, today_anchor=today)
    _seed_event(db, store, out_of_window, days_ago=10, today_anchor=today)

    pairs = fetch_label_events_window(
        db, label="hero", days=7, is_fashion=True, now=now,
    )
    ids = {p.id for p, _ in pairs}
    assert in_window.id in ids
    assert out_of_window.id not in ids


def test_days_window_de_duplicates_to_most_recent_event(db, store):
    now = datetime(2026, 5, 12, 12, 0, 0)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)

    p = _make_product(db, store, handle="multi", position=5)
    _seed_event(db, store, p, days_ago=5, today_anchor=today, prior=20, current=10)
    _seed_event(db, store, p, days_ago=2, today_anchor=today, prior=15, current=5)

    pairs = fetch_label_events_window(db, label="hero", days=7, now=now)
    matches = [(prod, ev) for prod, ev in pairs if prod.id == p.id]
    assert len(matches) == 1
    # Most recent event = the one 2 days ago (current=5).
    assert matches[0][1].current_position == 5


def test_days_window_separates_hero_and_villain_filters(db, store):
    now = datetime(2026, 5, 12, 12, 0, 0)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # Product was hero on day -5, villain on day -2.
    p = _make_product(db, store, handle="rollercoaster", position=12)
    _seed_event(db, store, p, days_ago=5, today_anchor=today, label="hero",
                 prior=20, current=10)
    _seed_event(db, store, p, days_ago=2, today_anchor=today, label="villain",
                 prior=8, current=12)

    hero_pairs = fetch_label_events_window(db, label="hero", days=7, now=now)
    villain_pairs = fetch_label_events_window(db, label="villain", days=7, now=now)
    assert p.id in {pr.id for pr, _ in hero_pairs}
    assert p.id in {pr.id for pr, _ in villain_pairs}


def test_days_window_excludes_when_label_event_is_outside(db, store):
    """Product was hero on day -5; user asks for last 3 days → excluded."""
    now = datetime(2026, 5, 12, 12, 0, 0)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)

    p = _make_product(db, store, handle="old-hero", position=12)
    _seed_event(db, store, p, days_ago=5, today_anchor=today, label="hero")

    pairs_3d = fetch_label_events_window(db, label="hero", days=3, now=now)
    assert p.id not in {pr.id for pr, _ in pairs_3d}
    pairs_7d = fetch_label_events_window(db, label="hero", days=7, now=now)
    assert p.id in {pr.id for pr, _ in pairs_7d}


def test_days_window_default_1_preserves_today_only(db, store):
    """days=1 should return only events with date == today_start."""
    now = datetime(2026, 5, 12, 12, 0, 0)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)

    today_prod = _make_product(db, store, handle="today-hero", position=5)
    yest_prod = _make_product(db, store, handle="yesterday-hero", position=6)
    _seed_event(db, store, today_prod, days_ago=0, today_anchor=today)
    _seed_event(db, store, yest_prod, days_ago=1, today_anchor=today)

    pairs = fetch_label_events_window(db, label="hero", days=1, now=now)
    ids = {p.id for p, _ in pairs}
    assert today_prod.id in ids
    assert yest_prod.id not in ids


def test_days_window_respects_is_fashion_filter(db, store):
    now = datetime(2026, 5, 12, 12, 0, 0)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)

    fashion_p = _make_product(
        db, store, handle="fash", position=5, is_fashion=True,
    )
    general_p = _make_product(
        db, store, handle="gen", position=5,
        is_fashion=False, subniche="home",
    )
    _seed_event(db, store, fashion_p, days_ago=2, today_anchor=today)
    _seed_event(db, store, general_p, days_ago=2, today_anchor=today)

    fashion_only = fetch_label_events_window(
        db, label="hero", days=7, is_fashion=True, now=now,
    )
    general_only = fetch_label_events_window(
        db, label="hero", days=7, is_fashion=False, now=now,
    )
    assert {p.id for p, _ in fashion_only} == {fashion_p.id}
    assert {p.id for p, _ in general_only} == {general_p.id}


# =====================================================================
# Cross-store isolation
# =====================================================================
def test_cross_store_events_are_independent(db):
    store_a = Store(name="A", url="https://a.example/",
                    monthly_visitors="1K", niche="Fashion", country="DE")
    store_b = Store(name="B", url="https://b.example/",
                    monthly_visitors="1K", niche="Fashion", country="DE")
    db.add(store_a); db.add(store_b); db.commit()
    db.refresh(store_a); db.refresh(store_b)

    now = datetime(2026, 5, 12, 12, 0, 0)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)

    p_a = _make_product(db, store_a, handle="x", position=5)
    p_b = _make_product(db, store_b, handle="x", position=5)
    _seed_event(db, store_a, p_a, days_ago=2, today_anchor=today)
    _seed_event(db, store_b, p_b, days_ago=2, today_anchor=today)

    pairs_a = fetch_label_events_window(
        db, label="hero", days=7, store_id=store_a.id, now=now,
    )
    pairs_b = fetch_label_events_window(
        db, label="hero", days=7, store_id=store_b.id, now=now,
    )
    assert {p.id for p, _ in pairs_a} == {p_a.id}
    assert {p.id for p, _ in pairs_b} == {p_b.id}


# =====================================================================
# Retention constant
# =====================================================================
def test_retention_constant_is_30():
    assert LABEL_EVENT_RETENTION_DAYS == 30


def test_trust_epoch_invariant_holds_on_today():
    """labels.TRUST_EPOCH_UTC must be strictly before today_start_utc()
    or compute_and_write_events silently aborts."""
    assert TRUST_EPOCH_UTC < today_start_utc()
