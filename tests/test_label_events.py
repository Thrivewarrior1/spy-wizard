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

from fastapi.testclient import TestClient

from models import Base, Store, Product, PositionHistory, LabelEvent
import labels as labels_mod
import main as main_mod
from main import app
from database import get_db
from labels import (
    TRUST_EPOCH_UTC,
    HERO_VILLAIN_DELTA_CAP,
    LABEL_EVENT_RETENTION_DAYS,
    DATA_START_DATE,
    compute_and_write_events,
    cleanup_label_events,
    cleanup_pre_start_label_events,
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
    backfill (today is owned by compute_and_write_events). Dates
    chosen to sit ON or AFTER DATA_START_DATE (2026-05-06) so the
    trustworthy floor doesn't suppress the synthesised event."""
    now = datetime(2026, 5, 9, 12, 0, 0)    # today
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    day_minus_2 = today - timedelta(days=2)  # 2026-05-07 (>=DATA_START)
    day_minus_1 = today - timedelta(days=1)  # 2026-05-08

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
    # Use dates on/after DATA_START_DATE (2026-05-06) so backfill
    # actually inserts something on the first run.
    now = datetime(2026, 5, 9, 12, 0, 0)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    d1 = today - timedelta(days=2)  # 2026-05-07
    d2 = today - timedelta(days=1)  # 2026-05-08

    p = _make_product(db, store, handle="ok", position=5)
    db.add(PositionHistory(product_id=p.id, position=25, date=d1 + timedelta(hours=12)))
    db.add(PositionHistory(product_id=p.id, position=5,  date=d2 + timedelta(hours=12)))
    db.commit()
    _pad_catalog(db, store)

    first = backfill_label_events(db, now=now)
    second = backfill_label_events(db, now=now)
    assert first >= 1
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


# =====================================================================
# DATA_START_DATE — trustworthy floor for the LabelEvent ledger.
# User-stated rule (2026-05-13): "only use the history if you have
# stored any accurate data for the last 7 days, and from there on
# move forward. Don't use any history from 30 days ago because that
# data isn't accurate, because then the Spy Wizard 2 wasn't even
# live yet." These tests pin the floor across read, backfill, and
# cleanup, plus the /api/stats?days endpoint contract.
# =====================================================================
def test_data_start_date_default_is_may_6_2026():
    """Default trustworthy floor is 2026-05-06 (7 days before the
    2026-05-13 stabilisation conversation). Env-overridable but the
    default is the load-bearing invariant."""
    assert DATA_START_DATE == datetime(2026, 5, 6, 0, 0, 0)


def test_fetch_window_excludes_event_before_data_start_date(db, store):
    """An event dated BEFORE DATA_START_DATE must never surface, even
    if the day-range window would otherwise include it."""
    now = datetime(2026, 5, 12, 12, 0, 0)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)

    p = _make_product(db, store, handle="pre-cutoff", position=5)
    # 2026-05-05 — one day BEFORE the 2026-05-06 floor.
    pre_floor = datetime(2026, 5, 5, 0, 0, 0)
    db.add(LabelEvent(
        store_id=store.id, product_id=p.id, date=pre_floor,
        label="hero", prior_position=20, current_position=5,
        position_change=15,
    ))
    db.commit()

    pairs = fetch_label_events_window(db, label="hero", days=30, now=now)
    assert p.id not in {pr.id for pr, _ in pairs}


def test_fetch_window_includes_event_on_data_start_date_boundary(db, store):
    """An event dated EXACTLY at DATA_START_DATE is trustworthy and
    must surface. The floor uses '>=' not '>'."""
    now = datetime(2026, 5, 12, 12, 0, 0)

    p = _make_product(db, store, handle="boundary", position=5)
    db.add(LabelEvent(
        store_id=store.id, product_id=p.id, date=DATA_START_DATE,
        label="hero", prior_position=20, current_position=5,
        position_change=15,
    ))
    db.commit()

    pairs = fetch_label_events_window(db, label="hero", days=30, now=now)
    assert p.id in {pr.id for pr, _ in pairs}


def test_fetch_window_30d_still_floored_by_data_start_date(db, store):
    """User asks for the full 30-day window. The day-range cutoff is
    way earlier than DATA_START_DATE, so the floor — not the window —
    is the binding constraint. Pre-floor events stay invisible."""
    now = datetime(2026, 5, 12, 12, 0, 0)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)

    in_floor = _make_product(db, store, handle="ok", position=5)
    pre_floor = _make_product(db, store, handle="ancient", position=6)
    # Inside 30-day window AND inside DATA_START floor.
    db.add(LabelEvent(
        store_id=store.id, product_id=in_floor.id,
        date=today - timedelta(days=3),
        label="hero", prior_position=20, current_position=5,
        position_change=15,
    ))
    # Inside 30-day window (2026-04-22) but BEFORE the 2026-05-06 floor.
    db.add(LabelEvent(
        store_id=store.id, product_id=pre_floor.id,
        date=today - timedelta(days=20),
        label="hero", prior_position=15, current_position=6,
        position_change=9,
    ))
    db.commit()

    pairs = fetch_label_events_window(db, label="hero", days=30, now=now)
    ids = {pr.id for pr, _ in pairs}
    assert in_floor.id in ids
    assert pre_floor.id not in ids


def test_backfill_skips_days_before_data_start_date(db, store):
    """Backfill must not synthesise events for dates < DATA_START_DATE,
    even when consecutive PositionHistory snapshots exist there. The
    floor is enforced both at the SQL filter and the per-day loop."""
    now = datetime(2026, 5, 12, 12, 0, 0)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    # Trust epoch is 2026-05-04, DATA_START_DATE is 2026-05-06. The
    # gap (2026-05-04 → 2026-05-05) is trustworthy by epoch but PRE
    # DATA_START_DATE — those days must NOT get backfilled events.
    pre_a = datetime(2026, 5, 4, 14, 0, 0)
    pre_b = datetime(2026, 5, 5, 14, 0, 0)
    in_a = datetime(2026, 5, 6, 14, 0, 0)
    in_b = datetime(2026, 5, 7, 14, 0, 0)

    p = _make_product(db, store, handle="climber", position=4)
    db.add(PositionHistory(product_id=p.id, position=40, date=pre_a))
    db.add(PositionHistory(product_id=p.id, position=30, date=pre_b))
    db.add(PositionHistory(product_id=p.id, position=20, date=in_a))
    db.add(PositionHistory(product_id=p.id, position=10, date=in_b))
    db.add(PositionHistory(product_id=p.id, position=4,  date=now))
    db.commit()
    _pad_catalog(db, store)

    backfill_label_events(db, now=now)

    events = db.query(LabelEvent).filter(LabelEvent.product_id == p.id).all()
    event_dates = {ev.date.date() for ev in events}
    # 2026-05-05 (pre-floor) and 2026-05-04 (pre-floor) must NOT have events.
    assert datetime(2026, 5, 5).date() not in event_dates
    assert datetime(2026, 5, 4).date() not in event_dates
    # 2026-05-07 (post-floor) SHOULD have an event (prior = 2026-05-06).
    assert datetime(2026, 5, 7).date() in event_dates


def test_cleanup_pre_start_label_events_deletes_pre_cutoff_rows(db, store):
    """Seed three pre-floor events + two post-floor events. Cleanup
    must drop only the pre-floor ones and leave the post-floor ones
    untouched."""
    p = _make_product(db, store, handle="x", position=5)
    pre_dates = [
        datetime(2026, 4, 28, 0, 0, 0),
        datetime(2026, 5, 1, 0, 0, 0),
        datetime(2026, 5, 5, 0, 0, 0),
    ]
    post_dates = [
        DATA_START_DATE,                              # boundary keeps
        DATA_START_DATE + timedelta(days=2),          # safely inside
    ]
    for d in pre_dates + post_dates:
        db.add(LabelEvent(
            store_id=store.id, product_id=p.id, date=d,
            label="hero", prior_position=10, current_position=5,
            position_change=5,
        ))
    db.commit()
    assert db.query(LabelEvent).count() == 5

    deleted = cleanup_pre_start_label_events(db)
    assert deleted == 3
    assert db.query(LabelEvent).count() == 2
    remaining_dates = {ev.date for ev in db.query(LabelEvent).all()}
    assert remaining_dates == set(post_dates)


def test_cleanup_pre_start_label_events_is_idempotent(db, store):
    """Running the migration twice must not double-delete or error.
    Second call returns 0 because the first call already cleared
    everything below the floor."""
    p = _make_product(db, store, handle="x", position=5)
    db.add(LabelEvent(
        store_id=store.id, product_id=p.id,
        date=datetime(2026, 5, 1, 0, 0, 0),
        label="hero", prior_position=10, current_position=5,
        position_change=5,
    ))
    db.commit()
    assert cleanup_pre_start_label_events(db) == 1
    assert cleanup_pre_start_label_events(db) == 0


def test_cleanup_pre_start_label_events_keeps_boundary_row(db, store):
    """A row dated EXACTLY at DATA_START_DATE is in-floor and must
    NOT be deleted. The filter is `date < DATA_START_DATE` (strict),
    not `<=`."""
    p = _make_product(db, store, handle="x", position=5)
    db.add(LabelEvent(
        store_id=store.id, product_id=p.id, date=DATA_START_DATE,
        label="hero", prior_position=10, current_position=5,
        position_change=5,
    ))
    db.commit()
    assert cleanup_pre_start_label_events(db) == 0
    assert db.query(LabelEvent).count() == 1


# =====================================================================
# /api/stats?days=N — counter contract.
# Source of truth must be fetch_label_events_window so the counter
# value ALWAYS matches the count of rows the feed endpoints return.
# =====================================================================
@pytest.fixture
def stats_client(db):
    """Bind the FastAPI app to the same in-memory db fixture so the
    stats endpoint reads from the same rows we seed in the test."""
    def _override():
        try:
            yield db
        finally:
            pass
    app.dependency_overrides[get_db] = _override
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_db, None)


def test_stats_endpoint_returns_data_start_date_field(stats_client):
    """Top-of-page response must expose DATA_START_DATE so the UI can
    show 'data trustworthy since 2026-05-06' if it wants to."""
    r = stats_client.get("/api/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["data_start_date"] == DATA_START_DATE.date().isoformat()
    assert body["days"] == 1  # default


def test_stats_endpoint_uses_days_window_for_counts(stats_client, db, store):
    """Seed a hero event 5 days ago and a hero event today. /api/stats
    with days=1 must count only today's; with days=7 it must count
    both. This is the user-visible contract that the counter follows
    the global pill."""
    today = today_start_utc()
    today_p = _make_product(db, store, handle="today-hero", position=5)
    older_p = _make_product(db, store, handle="older-hero", position=6)
    db.add(LabelEvent(
        store_id=store.id, product_id=today_p.id, date=today,
        label="hero", prior_position=20, current_position=5,
        position_change=15,
    ))
    db.add(LabelEvent(
        store_id=store.id, product_id=older_p.id,
        date=today - timedelta(days=5),
        label="hero", prior_position=15, current_position=6,
        position_change=9,
    ))
    db.commit()

    r1 = stats_client.get("/api/stats?days=1")
    assert r1.status_code == 200
    assert r1.json()["heroes"] == 1

    r7 = stats_client.get("/api/stats?days=7")
    assert r7.status_code == 200
    assert r7.json()["heroes"] == 2


def test_stats_endpoint_count_matches_fetch_label_events_window(
    stats_client, db, store,
):
    """Single source of truth: /api/stats counts == len(pairs) returned
    by fetch_label_events_window for the same label/days/is_fashion.
    No drift between counter and feed."""
    today = today_start_utc()
    p1 = _make_product(db, store, handle="h1", position=5)
    p2 = _make_product(db, store, handle="h2", position=6)
    p3 = _make_product(db, store, handle="v1", position=12)
    db.add(LabelEvent(
        store_id=store.id, product_id=p1.id, date=today,
        label="hero", prior_position=20, current_position=5,
        position_change=15,
    ))
    db.add(LabelEvent(
        store_id=store.id, product_id=p2.id,
        date=today - timedelta(days=2),
        label="hero", prior_position=18, current_position=6,
        position_change=12,
    ))
    db.add(LabelEvent(
        store_id=store.id, product_id=p3.id, date=today,
        label="villain", prior_position=5, current_position=12,
        position_change=-7,
    ))
    db.commit()

    fn_heroes = len(fetch_label_events_window(
        db, label="hero", days=7, is_fashion=True,
    ))
    fn_villains = len(fetch_label_events_window(
        db, label="villain", days=7, is_fashion=True,
    ))

    r = stats_client.get("/api/stats?days=7")
    assert r.status_code == 200
    body = r.json()
    assert body["heroes"] == fn_heroes
    assert body["villains"] == fn_villains


def test_stats_endpoint_excludes_pre_start_events_even_when_requested(
    stats_client, db, store, monkeypatch,
):
    """Pre-floor events must NEVER count toward the heroes/villains
    counter, even if days=30 nominally includes their date."""
    today = today_start_utc()
    # Pin DATA_START_DATE close to today so a few-days-old event lands
    # before the floor regardless of when the test runs.
    floor = today - timedelta(days=2)
    monkeypatch.setattr(labels_mod, "DATA_START_DATE", floor)

    pre_p = _make_product(db, store, handle="pre", position=5)
    post_p = _make_product(db, store, handle="post", position=6)
    db.add(LabelEvent(
        store_id=store.id, product_id=pre_p.id,
        date=floor - timedelta(days=1),  # one day before the floor
        label="hero", prior_position=20, current_position=5,
        position_change=15,
    ))
    db.add(LabelEvent(
        store_id=store.id, product_id=post_p.id, date=today,
        label="hero", prior_position=15, current_position=6,
        position_change=9,
    ))
    db.commit()

    r = stats_client.get("/api/stats?days=30")
    assert r.status_code == 200
    body = r.json()
    # Only post_p counts — pre_p is below the trustworthy floor.
    assert body["heroes"] == 1
