"""Hero/villain label events — persistence, retention, backfill.

Previously labels were computed ONLY at read time from PositionHistory,
which meant every day's heroes/villains evaporated the moment the next
scrape landed. User asked for 30 days of retention plus a UI filter for
7/14/30-day windows.

Now: every scrape commits one LabelEvent row per qualifying product
(hero / villain within the 30-rank delta cap AND the per-store catalog-
fraction guard), and the API serves heroes/villains by reading from
LabelEvent within a `days` window. PositionHistory is still the source
of truth for the underlying snapshots, but LabelEvent is the canonical
ledger of "this product was a hero on day X".

This module also owns the trust-epoch + delta-cap constants so they
can be imported by both main.py and scraper.py without circular
dependency. Both modules used to keep their own copies — that risked
drift between scrape-time and read-time label computation.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import func, and_
from sqlalchemy.orm import Session

from models import Product, PositionHistory, LabelEvent, Store

logger = logging.getLogger(__name__)


# =====================================================================
# Trust epoch + delta cap constants.
# =====================================================================
# Snapshots dated BEFORE TRUST_EPOCH_UTC are not trustworthy comparators
# because the underlying product set differed (different cap, different
# hard-drop regex, different fashion-only filter, different schema).
# Bump this every time we ship a change that meaningfully reshapes the
# catalog so day-over-day deltas don't get polluted by structural
# reshuffles.
#
# Override at deploy time with TRUST_EPOCH_UTC env var (ISO 8601).
#
# INVARIANT: TRUST_EPOCH_UTC must ALWAYS be < today's 00:00 UTC, or the
# prior-position filter becomes mutually exclusive and silently returns
# zero heroes/villains. The startup invariant check in main.py
# (`_trust_epoch_invariant_check`) catches this loudly.
_DEFAULT_TRUST_EPOCH = datetime(2026, 5, 4, 0, 0, 0)


def _parse_trust_epoch(raw: Optional[str]) -> datetime:
    if not raw:
        return _DEFAULT_TRUST_EPOCH
    raw = raw.strip()
    if raw.endswith("Z"):
        raw = raw[:-1]
    try:
        parsed = datetime.fromisoformat(raw)
        return parsed.replace(tzinfo=None)
    except ValueError:
        logger.warning(
            "TRUST_EPOCH_UTC=%r is not parseable ISO 8601; falling back to %s",
            raw, _DEFAULT_TRUST_EPOCH.isoformat(),
        )
        return _DEFAULT_TRUST_EPOCH


TRUST_EPOCH_UTC = _parse_trust_epoch(os.getenv("TRUST_EPOCH_UTC"))


# Hard cap on plausible day-over-day rank movement. Anything larger is
# almost certainly a structural reshuffle (catalog change, scrape source
# change) rather than organic movement, so we suppress the
# hero/villain label and call it 'normal' instead. The per-store
# catalog-size sanity check (30% of catalog) further tightens this for
# smaller stores.
HERO_VILLAIN_DELTA_CAP = 30
HERO_VILLAIN_CATALOG_FRACTION = 0.30

# Retention window for LabelEvent rows (days). Events older than this
# get pruned by cleanup_label_events() at the tail of every
# scrape_all_stores().
LABEL_EVENT_RETENTION_DAYS = 30


# =====================================================================
# Today / prior-position helpers (used at both read and scrape time).
# =====================================================================
def today_start_utc() -> datetime:
    """UTC midnight of today. Boundary between 'today's snapshot' and
    'prior snapshot' for the day-over-day delta. Same-UTC-day priors
    do NOT count, so the boundary excludes everything from 00:00 of
    today onward."""
    return datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)


def trustworthy_prior_filters(today_start: datetime):
    """Two filter clauses every prior-position lookup must apply:
    (1) prior must come from a DIFFERENT UTC calendar day (i.e. before
        today_start), AND
    (2) prior must be from on or after TRUST_EPOCH_UTC.
    """
    return (
        PositionHistory.date < today_start,
        PositionHistory.date >= TRUST_EPOCH_UTC,
    )


def delta_threshold(catalog_size: int) -> int:
    """Maximum plausible day-over-day rank delta for a store with
    `catalog_size` tracked products. Smaller of:
      - HERO_VILLAIN_DELTA_CAP (absolute, currently 30), and
      - 30% of catalog_size (rounded down, floored at 1).
    """
    if catalog_size <= 0:
        return HERO_VILLAIN_DELTA_CAP
    pct = max(1, int(catalog_size * HERO_VILLAIN_CATALOG_FRACTION))
    return min(HERO_VILLAIN_DELTA_CAP, pct)


# =====================================================================
# Compute & write events at SCRAPE time.
# =====================================================================
def compute_and_write_events(
    db: Session,
    store: Store,
    *,
    now: Optional[datetime] = None,
) -> tuple[int, int]:
    """After a per-store scrape commits its products + PositionHistory,
    walk every product in this store, compute its day-over-day rank
    label (hero / villain / normal / new), and INSERT one LabelEvent
    row per hero or villain.

    Idempotent for same-day re-scrapes: events are keyed on
    (product_id, date) where `date` is today's UTC midnight. If a row
    already exists for this product today, we UPDATE the position
    fields rather than inserting a duplicate.

    Returns (heroes_written, villains_written).
    """
    now = now or datetime.utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # Sanity guard: a misconfigured trust epoch silently disables the
    # whole pipeline. Skip with a loud warning instead of writing
    # nothing.
    if TRUST_EPOCH_UTC >= today_start:
        logger.warning(
            "compute_and_write_events: TRUST_EPOCH_UTC (%s) >= today_start "
            "(%s) — invariant violation, skipping event writes for %s",
            TRUST_EPOCH_UTC.isoformat(), today_start.isoformat(),
            store.name,
        )
        return (0, 0)

    products = (
        db.query(Product)
        .filter(Product.store_id == store.id)
        .all()
    )
    if not products:
        return (0, 0)

    # Pull the most-recent trustworthy prior position per product in
    # one query. Same logic as main._prior_position_subquery but
    # without the subquery wrapper since we just need a dict here.
    prior_rows = (
        db.query(PositionHistory.product_id, PositionHistory.position)
        .filter(PositionHistory.product_id.in_([p.id for p in products]))
        .filter(*trustworthy_prior_filters(today_start))
        .order_by(
            PositionHistory.product_id,
            PositionHistory.date.desc(),
        )
        .all()
    )
    prior_map: dict[int, int] = {}
    for pid, pos in prior_rows:
        if pid not in prior_map:
            prior_map[pid] = pos

    # Per-feed catalog size for the threshold. Mirrors main's
    # _store_catalog_sizes — same store_id can have separate fashion
    # and general feeds, each sized independently.
    fashion_size = sum(1 for p in products if p.is_fashion)
    general_size = sum(
        1 for p in products if (not p.is_fashion) and p.subniche
    )

    # Pre-fetch existing events for today so we can upsert without an
    # N+1.
    existing_today = (
        db.query(LabelEvent)
        .filter(LabelEvent.store_id == store.id, LabelEvent.date == today_start)
        .all()
    )
    by_pid: dict[int, LabelEvent] = {e.product_id: e for e in existing_today}

    heroes = 0
    villains = 0
    for p in products:
        prior = prior_map.get(p.id, 0)
        cur = p.current_position or 0
        if not prior or not cur or cur == prior:
            continue
        catalog_size = fashion_size if p.is_fashion else general_size
        thresh = delta_threshold(catalog_size)
        delta = abs(cur - prior)
        if delta == 0 or delta > thresh:
            continue
        if cur < prior:
            label = "hero"
            position_change = prior - cur   # positive
            heroes += 1
        else:
            label = "villain"
            position_change = prior - cur   # negative
            villains += 1

        ev = by_pid.get(p.id)
        if ev is None:
            db.add(LabelEvent(
                store_id=store.id, product_id=p.id, date=today_start,
                label=label, prior_position=prior,
                current_position=cur, position_change=position_change,
            ))
        else:
            ev.label = label
            ev.prior_position = prior
            ev.current_position = cur
            ev.position_change = position_change

    if heroes or villains:
        db.commit()
        logger.info(
            "compute_and_write_events[%s]: %d heroes + %d villains "
            "recorded for %s",
            store.name, heroes, villains, today_start.date().isoformat(),
        )
    return (heroes, villains)


# =====================================================================
# Retention.
# =====================================================================
def cleanup_label_events(
    db: Session,
    *,
    retention_days: int = LABEL_EVENT_RETENTION_DAYS,
    now: Optional[datetime] = None,
) -> int:
    """Delete LabelEvent rows older than retention_days. Idempotent.
    Returns the number of rows deleted."""
    now = now or datetime.utcnow()
    cutoff = now - timedelta(days=retention_days)
    deleted = (
        db.query(LabelEvent)
        .filter(LabelEvent.date < cutoff)
        .delete(synchronize_session=False)
    )
    db.commit()
    if deleted:
        logger.info(
            "cleanup_label_events: pruned %d events older than %s "
            "(retention=%dd)",
            deleted, cutoff.isoformat(), retention_days,
        )
    return deleted


# =====================================================================
# Backfill from existing PositionHistory.
# =====================================================================
def backfill_label_events(
    db: Session,
    *,
    retention_days: int = LABEL_EVENT_RETENTION_DAYS,
    now: Optional[datetime] = None,
) -> int:
    """Walk PositionHistory day-by-day for each product. For each pair
    of consecutive trustworthy snapshots (different UTC days, on or
    after the trust epoch), compute the hero/villain label and write
    one LabelEvent.

    Idempotent: if an event already exists for (product_id, date) and
    its label matches what we'd write, we leave it alone. Capped at
    `retention_days` so an old PositionHistory doesn't repopulate
    events the cleanup would immediately delete.

    Returns the number of events inserted (existing events that were
    consistent are not counted).
    """
    now = now or datetime.utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    cutoff = today_start - timedelta(days=retention_days)

    # Pull every product with at least one PositionHistory row inside
    # the window. We compute labels per (store, product) so the
    # catalog-fraction threshold uses the live store size.
    products = db.query(Product).all()
    products_by_store: dict[int, list[Product]] = {}
    for p in products:
        products_by_store.setdefault(p.store_id, []).append(p)

    # Existing event keys so we can de-dup.
    existing_keys: set[tuple[int, datetime]] = set(
        db.query(LabelEvent.product_id, LabelEvent.date)
        .filter(LabelEvent.date >= cutoff)
        .all()
    )
    # `query(...).all()` returns rows; SQLAlchemy 2.x yields tuples.
    # Convert to a set of tuples explicitly to defend against either.
    existing_keys = {(pid, dt) for (pid, dt) in existing_keys}

    total_inserted = 0
    for store_id, store_products in products_by_store.items():
        fashion_size = sum(1 for p in store_products if p.is_fashion)
        general_size = sum(
            1 for p in store_products if (not p.is_fashion) and p.subniche
        )

        # Pull all in-window snapshots for these products, sorted.
        pids = [p.id for p in store_products]
        snaps = (
            db.query(PositionHistory.product_id, PositionHistory.position, PositionHistory.date)
            .filter(PositionHistory.product_id.in_(pids))
            .filter(PositionHistory.date >= TRUST_EPOCH_UTC)
            .filter(PositionHistory.date >= cutoff)
            .order_by(PositionHistory.product_id, PositionHistory.date.asc())
            .all()
        )

        # Group snapshots by product_id, then by UTC date — pick the
        # LAST snapshot of each UTC day to represent that day's
        # position (mirrors the read-time logic).
        by_pid_then_day: dict[int, dict[datetime, int]] = {}
        for pid, pos, dt in snaps:
            day = dt.replace(hour=0, minute=0, second=0, microsecond=0)
            by_pid_then_day.setdefault(pid, {})[day] = pos

        # For each product, walk consecutive days and emit events.
        product_by_id = {p.id: p for p in store_products}
        for pid, day_pos in by_pid_then_day.items():
            sorted_days = sorted(day_pos.keys())
            if len(sorted_days) < 2:
                continue
            product = product_by_id[pid]
            cat_size = (
                fashion_size if product.is_fashion else general_size
            )
            thresh = delta_threshold(cat_size)
            for i in range(1, len(sorted_days)):
                day = sorted_days[i]
                if day == today_start:
                    # Today's events are owned by compute_and_write_events()
                    # at scrape time so the live position is fresh.
                    # Skip in backfill to avoid duplicate insert races.
                    continue
                prior_day = sorted_days[i - 1]
                prior_pos = day_pos[prior_day]
                cur_pos = day_pos[day]
                delta = abs(cur_pos - prior_pos)
                if delta == 0 or delta > thresh:
                    continue
                if cur_pos < prior_pos:
                    label = "hero"
                    pos_change = prior_pos - cur_pos
                else:
                    label = "villain"
                    pos_change = prior_pos - cur_pos
                if (pid, day) in existing_keys:
                    continue
                db.add(LabelEvent(
                    store_id=store_id, product_id=pid, date=day,
                    label=label,
                    prior_position=prior_pos,
                    current_position=cur_pos,
                    position_change=pos_change,
                ))
                existing_keys.add((pid, day))
                total_inserted += 1

    if total_inserted:
        db.commit()
        logger.info(
            "backfill_label_events: inserted %d events from PositionHistory",
            total_inserted,
        )
    return total_inserted


# =====================================================================
# Read-side helper used by the API.
# =====================================================================
def fetch_label_events_window(
    db: Session,
    *,
    label: str,
    days: int,
    is_fashion: Optional[bool] = None,
    store_id: Optional[int] = None,
    now: Optional[datetime] = None,
) -> list[tuple[Product, LabelEvent]]:
    """Return (product, latest_event) pairs for every product whose
    most-recent LabelEvent matching `label` falls within the last
    `days` UTC days.

    De-duplication: each product appears at most once, paired with the
    NEWEST qualifying event. A product can therefore appear in both
    'hero' and 'villain' filtered lists in the same window if it has
    events of both kinds.

    Includes products that are no longer 'active' in the current feed
    (is_fashion flipped, subniche cleared) — the caller decides
    whether to render them differently.
    """
    if label not in ("hero", "villain"):
        return []
    now = now or datetime.utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    cutoff = today_start - timedelta(days=days - 1) if days >= 1 else today_start

    q = (
        db.query(LabelEvent, Product)
        .join(Product, Product.id == LabelEvent.product_id)
        .filter(LabelEvent.label == label)
        .filter(LabelEvent.date >= cutoff)
    )
    if store_id is not None:
        q = q.filter(LabelEvent.store_id == store_id)
    # Order so the LATEST event per product comes first; we'll de-dup
    # in Python below (cross-DB compatible).
    q = q.order_by(LabelEvent.product_id, LabelEvent.date.desc())

    seen: set[int] = set()
    out: list[tuple[Product, LabelEvent]] = []
    for ev, prod in q.all():
        if prod.id in seen:
            continue
        seen.add(prod.id)
        if is_fashion is True and not prod.is_fashion:
            continue
        if is_fashion is False and prod.is_fashion:
            continue
        if is_fashion is False and not prod.subniche:
            continue
        out.append((prod, ev))
    return out
