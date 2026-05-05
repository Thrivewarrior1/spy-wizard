"""Trust-epoch + hero/villain regression tests.

The user has hit the silent-0/0 regression twice now. The cause both
times was a TRUST_EPOCH_UTC value that landed at or past today's UTC
midnight, which makes the prior-position filter

    PositionHistory.date < today_start AND
    PositionHistory.date >= TRUST_EPOCH_UTC

mutually exclusive — no row can satisfy both, so heroes_q and
villains_q both return 0. Silent. No log line. No alert.

These tests pin:
  - the boundary semantics: TRUST_EPOCH_UTC is INCLUSIVE (>=), today's
    midnight is EXCLUSIVE (<). A snapshot dated EXACTLY at the trust
    epoch IS a valid prior.
  - the user's exact scenario: yesterday's snapshot at trust-epoch +
    1 day, today's snapshot at trust-epoch + 2 days, product moved up
    25 ranks → label = 'hero', position_change = 25, prior = the
    yesterday position.
  - the symmetric villain case (moved down).
  - the unchanged case → 'normal'.
  - the no-prior case → 'new' (debut).
  - the over-cap case (delta > HERO_VILLAIN_DELTA_CAP) → 'normal',
    NOT hero/villain (the user's earlier 55→5 complaint).
  - the invariant guard catches an epoch == today_start mistake so
    future deploys can't silently re-introduce the regression.
"""
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models import Base, Store, Product, PositionHistory


@pytest.fixture
def db():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    Session = sessionmaker(bind=eng)
    s = Session()
    yield s
    s.close()


@pytest.fixture
def store(db):
    s = Store(name="Test", url="https://test.example/", monthly_visitors="1K",
              niche="Fashion", country="DE")
    db.add(s); db.commit(); db.refresh(s)
    return s


def _seed_product(db, store, *, position, is_fashion=True, subniche="fashion",
                   handle="x"):
    p = Product(
        store_id=store.id, shopify_id=handle, title=handle, handle=handle,
        image_url="", price="", vendor="", product_type="", product_url="",
        current_position=position, previous_position=0, label="",
        ai_tags="", is_fashion=is_fashion, subniche=subniche,
        last_scraped=datetime.utcnow(),
    )
    db.add(p); db.commit(); db.refresh(p)
    return p


def _seed_history(db, product, *, position, date):
    db.add(PositionHistory(product_id=product.id, position=position, date=date))
    db.commit()


def _pad_catalog(db, store, *, n=120, yesterday=None, today=None):
    """Pad a store's fashion catalog so the per-store 30%-of-catalog
    threshold doesn't suppress the label under test. The threshold is
    `min(HERO_VILLAIN_DELTA_CAP, int(catalog_size * 0.30))`; with only
    the product under test in the catalog, the threshold collapses to
    1 and any delta > 1 demotes to 'normal'. Padding to 120 fashion
    rows lifts the threshold back to the absolute cap of 30."""
    for i in range(n):
        pad = _seed_product(
            db, store, position=i + 100, handle=f"pad-{i}",
            is_fashion=True, subniche="fashion",
        )
        if yesterday is not None:
            _seed_history(db, pad, position=i + 100, date=yesterday)
        if today is not None:
            _seed_history(db, pad, position=i + 100, date=today)


# === The user's exact scenario ===
def test_yesterday_prior_produces_hero_today(db, store):
    """May 4 snapshot at position 30, May 5 product at position 5.
    With TRUST_EPOCH_UTC = May 4 00:00 (inclusive), the May 4 prior
    qualifies and the comparison produces label='hero', delta=25."""
    epoch = datetime(2026, 5, 4, 0, 0, 0)
    today = datetime(2026, 5, 5, 12, 0, 0)  # any time on May 5

    # Yesterday's snapshot timestamp lands AFTER the trust epoch and
    # BEFORE today's UTC midnight — both filters satisfied.
    yesterday_snapshot = datetime(2026, 5, 4, 14, 30, 0)

    p = _seed_product(db, store, position=5, handle="moved-up")
    _seed_history(db, p, position=30, date=yesterday_snapshot)
    # Today's snapshot also exists (that's how current_position got
    # populated), but the prior subquery picks the most-recent row
    # dated < today_start.
    _seed_history(db, p, position=5, date=today)
    _pad_catalog(db, store, yesterday=yesterday_snapshot, today=today)

    with patch("main.TRUST_EPOCH_UTC", epoch), \
         patch("main._today_start_utc", lambda: today.replace(
             hour=0, minute=0, second=0, microsecond=0,
         )):
        from main import _compute_label_map
        label_map = _compute_label_map(db, [p])

    assert p.id in label_map
    label, delta, prior = label_map[p.id]
    assert label == "hero", f"expected hero, got {label}"
    assert delta == 25
    assert prior == 30


def test_yesterday_prior_produces_villain_today(db, store):
    """Symmetric — product moved DOWN from position 10 to position 35.
    Within the 30-rank delta cap (35-10 = 25), so label = villain."""
    epoch = datetime(2026, 5, 4, 0, 0, 0)
    today = datetime(2026, 5, 5, 12, 0, 0)
    yesterday = datetime(2026, 5, 4, 14, 30, 0)

    p = _seed_product(db, store, position=35, handle="moved-down")
    _seed_history(db, p, position=10, date=yesterday)
    _seed_history(db, p, position=35, date=today)
    _pad_catalog(db, store, yesterday=yesterday, today=today)

    with patch("main.TRUST_EPOCH_UTC", epoch), \
         patch("main._today_start_utc", lambda: today.replace(
             hour=0, minute=0, second=0, microsecond=0,
         )):
        from main import _compute_label_map
        label_map = _compute_label_map(db, [p])

    label, delta, prior = label_map[p.id]
    assert label == "villain", f"expected villain, got {label}"
    # _compute_label_map stores delta as prior - cur (negative for villain)
    assert delta == -25
    assert prior == 10


def test_unchanged_position_is_normal(db, store):
    epoch = datetime(2026, 5, 4, 0, 0, 0)
    today = datetime(2026, 5, 5, 12, 0, 0)
    yesterday = datetime(2026, 5, 4, 14, 30, 0)

    p = _seed_product(db, store, position=12, handle="static")
    _seed_history(db, p, position=12, date=yesterday)
    _seed_history(db, p, position=12, date=today)

    with patch("main.TRUST_EPOCH_UTC", epoch), \
         patch("main._today_start_utc", lambda: today.replace(
             hour=0, minute=0, second=0, microsecond=0,
         )):
        from main import _compute_label_map
        label_map = _compute_label_map(db, [p])

    label, delta, prior = label_map[p.id]
    assert label == "normal"
    assert delta == 0
    assert prior == 12


def test_no_prior_snapshot_is_new(db, store):
    """Product with no PositionHistory rows dated < today_start gets
    label='new', NOT hero/villain. This is the 'just-debuted product
    has nothing to compare to' case."""
    epoch = datetime(2026, 5, 4, 0, 0, 0)
    today = datetime(2026, 5, 5, 12, 0, 0)

    p = _seed_product(db, store, position=7, handle="debut")
    # Only today's snapshot exists.
    _seed_history(db, p, position=7, date=today)

    with patch("main.TRUST_EPOCH_UTC", epoch), \
         patch("main._today_start_utc", lambda: today.replace(
             hour=0, minute=0, second=0, microsecond=0,
         )):
        from main import _compute_label_map
        label_map = _compute_label_map(db, [p])

    label, delta, prior = label_map[p.id]
    assert label == "new"
    assert prior == 0


# === Boundary semantics ===
def test_snapshot_exactly_at_trust_epoch_qualifies(db, store):
    """A PositionHistory row dated EXACTLY at TRUST_EPOCH_UTC must
    qualify as a prior — the >= comparison is INCLUSIVE on the lower
    bound. Bug regression: an off-by-one to > would silently skip the
    first day's worth of trustworthy snapshots."""
    epoch = datetime(2026, 5, 4, 0, 0, 0)
    today = datetime(2026, 5, 5, 12, 0, 0)

    p = _seed_product(db, store, position=10, handle="boundary")
    # Snapshot at the exact epoch second.
    _seed_history(db, p, position=20, date=epoch)
    _seed_history(db, p, position=10, date=today)
    _pad_catalog(db, store, yesterday=epoch, today=today)

    with patch("main.TRUST_EPOCH_UTC", epoch), \
         patch("main._today_start_utc", lambda: today.replace(
             hour=0, minute=0, second=0, microsecond=0,
         )):
        from main import _compute_label_map
        label_map = _compute_label_map(db, [p])

    label, delta, prior = label_map[p.id]
    # Snapshot at exact epoch IS a valid prior → hero with delta=10.
    assert label == "hero"
    assert delta == 10
    assert prior == 20


def test_snapshot_just_before_trust_epoch_does_NOT_qualify(db, store):
    """A snapshot dated 1 second before TRUST_EPOCH_UTC must be
    excluded — that's the structural-reshuffle window the epoch is
    supposed to guard against."""
    epoch = datetime(2026, 5, 4, 0, 0, 0)
    today = datetime(2026, 5, 5, 12, 0, 0)
    pre_epoch = epoch - timedelta(seconds=1)

    p = _seed_product(db, store, position=10, handle="pre-epoch")
    _seed_history(db, p, position=99, date=pre_epoch)
    _seed_history(db, p, position=10, date=today)

    with patch("main.TRUST_EPOCH_UTC", epoch), \
         patch("main._today_start_utc", lambda: today.replace(
             hour=0, minute=0, second=0, microsecond=0,
         )):
        from main import _compute_label_map
        label_map = _compute_label_map(db, [p])

    label, delta, prior = label_map[p.id]
    # Pre-epoch snapshot rejected → no trustworthy prior → new.
    assert label == "new"
    assert prior == 0


def test_today_snapshot_does_NOT_qualify_as_prior(db, store):
    """Same-UTC-day snapshots are NOT priors — the < today_start
    filter ensures only previous-day or earlier snapshots compare."""
    epoch = datetime(2026, 5, 4, 0, 0, 0)
    today = datetime(2026, 5, 5, 12, 0, 0)
    today_start = today.replace(hour=0, minute=0, second=0, microsecond=0)
    earlier_today = today_start + timedelta(hours=4)

    p = _seed_product(db, store, position=10, handle="same-day")
    _seed_history(db, p, position=99, date=earlier_today)
    _seed_history(db, p, position=10, date=today)

    with patch("main.TRUST_EPOCH_UTC", epoch), \
         patch("main._today_start_utc", lambda: today_start):
        from main import _compute_label_map
        label_map = _compute_label_map(db, [p])

    label, delta, prior = label_map[p.id]
    assert label == "new"  # both snapshots are same-UTC-day → no valid prior


# === Delta cap (the 55→5 user complaint) ===
def test_delta_exceeding_30_is_demoted_to_normal(db, store):
    """The user's earlier complaint: a 55→5 jump (delta 50) is almost
    certainly a structural reshuffle, not organic movement. The cap
    suppresses it to 'normal' rather than labelling it hero/villain."""
    epoch = datetime(2026, 5, 4, 0, 0, 0)
    today = datetime(2026, 5, 5, 12, 0, 0)
    yesterday = datetime(2026, 5, 4, 14, 30, 0)

    p = _seed_product(db, store, position=5, handle="big-jump")
    _seed_history(db, p, position=55, date=yesterday)
    _seed_history(db, p, position=5, date=today)
    # Pad the catalog so per-store fraction doesn't kick in.
    for i in range(50):
        pad = _seed_product(
            db, store, position=i + 100, handle=f"pad-{i}",
            is_fashion=True, subniche="fashion",
        )
        _seed_history(db, pad, position=i + 100, date=yesterday)
        _seed_history(db, pad, position=i + 100, date=today)

    with patch("main.TRUST_EPOCH_UTC", epoch), \
         patch("main._today_start_utc", lambda: today.replace(
             hour=0, minute=0, second=0, microsecond=0,
         )):
        from main import _compute_label_map
        label_map = _compute_label_map(db, [p])

    label, delta, prior = label_map[p.id]
    assert label == "normal", (
        f"Delta of 50 ranks should be suppressed to 'normal' (structural "
        f"reshuffle), not '{label}'. The 30-rank cap is the user's "
        f"explicit guard against the prior 55→5 false-positive bug."
    )


# === Invariant guard ===
def test_trust_epoch_invariant_catches_epoch_eq_today_start():
    """If TRUST_EPOCH_UTC is set to today's UTC midnight, the prior-
    position filter becomes mutually exclusive (< today_start AND
    >= today_start) and the hero/villain query silently returns 0/0.
    The startup invariant check must flag this loudly so a deploy
    can't silently disable the labels."""
    today = datetime(2026, 5, 5, 12, 0, 0)
    today_start = today.replace(hour=0, minute=0, second=0, microsecond=0)

    # Epoch == today_start → invariant violated.
    with patch("main.TRUST_EPOCH_UTC", today_start), \
         patch("main.datetime") as mock_dt:
        mock_dt.utcnow.return_value = today
        from main import _trust_epoch_invariant_check
        msg = _trust_epoch_invariant_check()
    assert msg is not None
    assert "mutually exclusive" in msg.lower()


def test_trust_epoch_invariant_catches_epoch_after_today():
    """Epoch > today_start is even worse. Must also flag."""
    today = datetime(2026, 5, 5, 12, 0, 0)
    epoch_future = datetime(2026, 5, 6, 0, 0, 0)

    with patch("main.TRUST_EPOCH_UTC", epoch_future), \
         patch("main.datetime") as mock_dt:
        mock_dt.utcnow.return_value = today
        from main import _trust_epoch_invariant_check
        msg = _trust_epoch_invariant_check()
    assert msg is not None


def test_trust_epoch_invariant_passes_when_epoch_is_yesterday():
    """The good case — epoch one day before today is the user's spec
    and the system's intended steady state."""
    today = datetime(2026, 5, 5, 12, 0, 0)
    epoch_yesterday = datetime(2026, 5, 4, 0, 0, 0)

    with patch("main.TRUST_EPOCH_UTC", epoch_yesterday), \
         patch("main.datetime") as mock_dt:
        mock_dt.utcnow.return_value = today
        from main import _trust_epoch_invariant_check
        msg = _trust_epoch_invariant_check()
    assert msg is None


def test_default_trust_epoch_is_2026_05_04():
    """The constant must be 2026-05-04 (yesterday relative to the
    user's spec date 2026-05-05). If a future deploy bumps this to a
    date that isn't strictly before the deploy date, the silent
    0/0 regression returns. Pin the invariant here so any such bump
    fails CI before it can silently break live."""
    from main import _DEFAULT_TRUST_EPOCH
    assert _DEFAULT_TRUST_EPOCH == datetime(2026, 5, 4, 0, 0, 0)
