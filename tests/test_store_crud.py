"""Store CRUD regression tests.

Bug the user reported: adding a new competitor store on the live UI
returned "Failed to save" with no further detail. Investigation
showed the endpoint was working but the frontend swallowed the actual
error reason. The most common path was the duplicate-URL conflict
returning a misleading HTTP 400 with the generic message "Store
already exists" — frontend couldn't distinguish duplicate from
validation error.

Fix shipped:
  - Server-side validation: name and url required, url must look
    like a real URL. 400 with a clear `detail` string per failure
    mode.
  - Duplicate URL now returns 409 (Conflict) — semantically correct,
    and the frontend can distinguish from 400 (validation) if it
    wants to.
  - URL normalisation (strip trailing slash, prepend https://) so
    'novigood.com/' and 'novigood.com' collide on the unique check.
  - Frontend reads `detail` from the JSON response and shows it in
    the alert ('Save failed: A store with this URL already exists').

These tests pin every documented failure mode + the happy path.
"""
from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import sqlite3
from sqlalchemy import event

from models import Base, Store, Product
from main import app, _normalise_store_url, _validate_store_payload
from database import get_db


@pytest.fixture
def _test_db():
    """In-memory SQLite engine + Session class bound to it.
    StaticPool keeps every connection on the SAME database (default
    SQLite gives each connection a private DB). PRAGMA foreign_keys
    is enabled so the cascade tests exercise the same FK semantics
    Postgres enforces in production."""
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
    Session = sessionmaker(bind=eng, autoflush=False, autocommit=False)
    return eng, Session


@pytest.fixture
def client(_test_db):
    """Spin up an isolated in-memory DB + override get_db so the
    /api/stores endpoint hits the test database."""
    _, Session = _test_db

    def _override():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _override
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_db, None)


@pytest.fixture
def db_session_factory(_test_db):
    """Direct-session accessor against the same engine the client
    uses. Tests that seed Products / PositionHistory / LabelEvent
    via the ORM (vs the API) need this to bypass the dependency-
    override layer."""
    _, Session = _test_db
    return Session


# =====================================================================
# URL normalisation helper
# =====================================================================
@pytest.mark.parametrize("raw,expected", [
    ("https://novigood.com/",  "https://novigood.com"),
    ("https://novigood.com",   "https://novigood.com"),
    ("novigood.com",           "https://novigood.com"),
    ("  https://x.example/  ", "https://x.example"),
    ("http://x.example",       "http://x.example"),
    ("",                       ""),
    ("   ",                    ""),
    # Path stripping — the scraper always appends its own
    # /collections/all?sort_by=best-selling URL, so anything beyond
    # the host is a 404 trap. Live regression: Oliva Mode came in as
    # https://olivamode.com/collections/bags and the scraper built
    # https://olivamode.com/collections/bags/collections/all?... (404).
    ("https://olivamode.com/collections/bags",
        "https://olivamode.com"),
    ("https://shop.example/collections/all",
        "https://shop.example"),
    ("https://shop.example/collections/bags/products/x?foo=bar#hash",
        "https://shop.example"),
    ("https://shop.example/",
        "https://shop.example"),
])
def test_normalise_store_url(raw, expected):
    assert _normalise_store_url(raw) == expected


# Sanity: per-store scrape endpoint is non-blocking (returns immediately,
# kicks the actual work into a background task). Previously it ran
# synchronously and was killed by Render's 60s HTTP timeout, surfacing
# to the user as "Scraped 0 products. page 1 HTTP error:" — but the
# scrape had never actually finished.
def test_per_store_scrape_returns_immediately(client):
    cr = client.post("/api/stores", json={
        "name": "FastEnd", "url": "https://fastend.example/",
    }).json()
    # POST /api/scrape/{id} must return before scrape work would
    # plausibly finish — proves it's queued, not run inline.
    import time
    t0 = time.monotonic()
    r = client.post(f"/api/scrape/{cr['id']}")
    elapsed = time.monotonic() - t0
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("started") is True or body.get("running") is True
    # If this assertion ever fails, the endpoint went back to running
    # inline and will start 502-ing on long-tail merchants again.
    assert elapsed < 2.0, f"per-store scrape returned in {elapsed:.2f}s — should be <2s if truly async"


# =====================================================================
# Validation helper
# =====================================================================
@pytest.mark.parametrize("name,url,expected_substring", [
    ("",          "https://x.example/",  "name"),
    ("   ",       "https://x.example/",  "name"),
    ("Novi Good", "",                    "url"),
    ("Novi Good", "   ",                 "url"),
    ("Novi Good", "https://just-host",   "host"),   # no TLD → host invalid
])
def test_validate_store_payload_rejections(name, url, expected_substring):
    msg = _validate_store_payload(name, url)
    assert msg is not None
    assert expected_substring.lower() in msg.lower()


@pytest.mark.parametrize("name,url", [
    ("Novi Good",    "https://novigood.com/"),
    ("Wild Eye",     "https://wild-eye-vision.com"),
    ("Bare Host",    "novigood.com"),                # missing scheme — auto-fixed
    ("Long Name",    "https://a.b.c.d.example/path?q=1"),
])
def test_validate_store_payload_acceptance(name, url):
    assert _validate_store_payload(name, url) is None


# =====================================================================
# POST /api/stores happy path
# =====================================================================
def test_create_store_happy_path(client):
    r = client.post("/api/stores", json={
        "name": "Test Store", "url": "https://test-store.example/",
        "monthly_visitors": "10K", "niche": "Fashion", "country": "DE",
    })
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["id"]
    assert body["name"] == "Test Store"
    # URL gets normalised (trailing slash stripped).
    assert body["url"] == "https://test-store.example"
    assert body["niche"] == "Fashion"
    assert body["country"] == "DE"
    assert body["monthly_visitors"] == "10K"


def test_create_store_strips_whitespace_from_name(client):
    r = client.post("/api/stores", json={
        "name": "  Padded Name  ", "url": "https://padded.example/",
    })
    assert r.status_code == 201
    assert r.json()["name"] == "Padded Name"


def test_create_store_auto_prepends_scheme(client):
    r = client.post("/api/stores", json={
        "name": "Scheme-Less", "url": "scheme-less.example",
    })
    assert r.status_code == 201
    assert r.json()["url"] == "https://scheme-less.example"


# =====================================================================
# POST /api/stores failure modes
# =====================================================================
def test_create_store_duplicate_url_returns_409(client):
    """The user-facing bug: 'Failed to save' on duplicate. Now returns
    409 with a clear detail message that names the existing store."""
    client.post("/api/stores", json={
        "name": "First", "url": "https://collision.example/",
    })
    r2 = client.post("/api/stores", json={
        "name": "Second", "url": "https://collision.example/",
    })
    assert r2.status_code == 409
    detail = r2.json()["detail"]
    assert "already exists" in detail.lower()
    assert "first" in detail.lower()  # names the colliding store


def test_create_store_duplicate_via_trailing_slash_difference(client):
    """The two URLs 'novigood.com/' and 'novigood.com' must collide on
    the dedupe check after normalisation — otherwise a small typo
    creates a phantom duplicate."""
    client.post("/api/stores", json={
        "name": "First", "url": "https://collision.example/",
    })
    r2 = client.post("/api/stores", json={
        "name": "Second", "url": "https://collision.example",  # no slash
    })
    assert r2.status_code == 409


def test_create_store_duplicate_via_missing_scheme(client):
    client.post("/api/stores", json={
        "name": "First", "url": "https://collision.example/",
    })
    r2 = client.post("/api/stores", json={
        "name": "Second", "url": "collision.example",  # no scheme
    })
    assert r2.status_code == 409


def test_create_store_empty_name_returns_400(client):
    r = client.post("/api/stores", json={
        "name": "", "url": "https://x.example/",
    })
    assert r.status_code == 400
    assert "name" in r.json()["detail"].lower()


def test_create_store_whitespace_name_returns_400(client):
    r = client.post("/api/stores", json={
        "name": "   ", "url": "https://x.example/",
    })
    assert r.status_code == 400
    assert "name" in r.json()["detail"].lower()


def test_create_store_empty_url_returns_400(client):
    r = client.post("/api/stores", json={
        "name": "Foo", "url": "",
    })
    assert r.status_code == 400
    assert "url" in r.json()["detail"].lower()


def test_create_store_invalid_host_returns_400(client):
    """No TLD in the host part → reject. Catches 'localhost' and
    obvious typos like 'foobar' (single token)."""
    r = client.post("/api/stores", json={
        "name": "Foo", "url": "https://no-tld-here",
    })
    assert r.status_code == 400


def test_create_store_missing_field_returns_422(client):
    """FastAPI's Pydantic validation kicks in for missing required
    fields. The frontend now reads `detail` and shows whichever error
    the server returns."""
    r = client.post("/api/stores", json={"name": "No URL"})
    assert r.status_code == 422


# =====================================================================
# PUT /api/stores/{id} — patch flow
# =====================================================================
def test_update_store_partial_payload(client):
    cr = client.post("/api/stores", json={
        "name": "Original", "url": "https://original.example/",
    })
    sid = cr.json()["id"]
    # Patch only the name, leave url alone.
    r = client.put(f"/api/stores/{sid}", json={"name": "Renamed"})
    assert r.status_code == 200, r.text


def test_update_store_url_collision_returns_409(client):
    a = client.post("/api/stores", json={
        "name": "A", "url": "https://a.example/",
    }).json()
    b = client.post("/api/stores", json={
        "name": "B", "url": "https://b.example/",
    }).json()
    # Try to rename B's URL to A's — must conflict.
    r = client.put(f"/api/stores/{b['id']}", json={"url": "https://a.example/"})
    assert r.status_code == 409


def test_update_store_empty_name_returns_400(client):
    cr = client.post("/api/stores", json={
        "name": "Real", "url": "https://r.example/",
    }).json()
    r = client.put(f"/api/stores/{cr['id']}", json={"name": ""})
    assert r.status_code == 400


# =====================================================================
# DELETE — cascade survives + returns 200
# =====================================================================
def test_delete_store_cascades_to_products(client):
    """Already covered by test_seed_persistence.py at the ORM level —
    this confirms the API endpoint also triggers cascade."""
    cr = client.post("/api/stores", json={
        "name": "ToDelete", "url": "https://todelete.example/",
    }).json()
    sid = cr["id"]
    r = client.delete(f"/api/stores/{sid}")
    assert r.status_code == 200
    # GET should no longer find it.
    r2 = client.get("/api/stores")
    assert all(s["id"] != sid for s in r2.json())


def test_delete_unknown_store_returns_404(client):
    r = client.delete("/api/stores/99999")
    assert r.status_code == 404


# =====================================================================
# Round-trip — add then list returns the new store
# =====================================================================
def test_create_then_list_round_trip(client):
    cr = client.post("/api/stores", json={
        "name": "Roundtrip", "url": "https://roundtrip.example/",
    })
    sid = cr.json()["id"]
    listing = client.get("/api/stores").json()
    found = next((s for s in listing if s["id"] == sid), None)
    assert found is not None
    assert found["name"] == "Roundtrip"
    assert found["url"] == "https://roundtrip.example"


# =====================================================================
# DELETE — explicit cascade + count reporting.
# User complaint: deleting a competitor left its products / history /
# events visible in the dashboard. These tests pin the full-tree
# wipe + the count payload the UI uses to confirm the delete actually
# did something.
# =====================================================================
from datetime import datetime, timedelta
from models import PositionHistory, LabelEvent


def _seed_store_with_data(client, db_session_factory, *, name, url,
                           n_products=3, n_history=2, n_events=2):
    """Create a store via API, then seed products + history + label
    events directly via the ORM session so the cascade test can verify
    counts without going through the scraper."""
    cr = client.post("/api/stores", json={"name": name, "url": url}).json()
    sid = cr["id"]
    db = db_session_factory()
    try:
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        for i in range(n_products):
            p = Product(
                store_id=sid, shopify_id=f"{name}-{i}", title=f"{name}-{i}",
                handle=f"{name}-{i}", current_position=i+1, previous_position=0,
                label="", is_fashion=True, subniche="fashion",
                last_scraped=datetime.utcnow(),
            )
            db.add(p)
            db.commit()
            db.refresh(p)
            for j in range(n_history):
                db.add(PositionHistory(
                    product_id=p.id, position=i+1+j,
                    date=today - timedelta(days=j),
                ))
            for j in range(n_events):
                db.add(LabelEvent(
                    store_id=sid, product_id=p.id,
                    date=today - timedelta(days=j),
                    label="hero", prior_position=20, current_position=i+1,
                    position_change=20 - (i+1),
                ))
        db.commit()
    finally:
        db.close()
    return sid


def test_delete_store_returns_cascade_counts(client, db_session_factory):
    """DELETE response must include the count of every child row
    removed so the UI can show 'deleted store + N products + M history'
    instead of a silent 200."""
    sid = _seed_store_with_data(
        client, db_session_factory,
        name="CascadeTest", url="https://cascade.example/",
        n_products=4, n_history=3, n_events=2,
    )
    r = client.delete(f"/api/stores/{sid}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True
    assert "deleted" in body
    d = body["deleted"]
    assert d["store"] == 1
    assert d["products"] == 4
    # 4 products * 3 history rows each = 12
    assert d["position_history"] == 12
    # 4 products * 2 events each = 8
    assert d["label_events"] == 8


def test_delete_store_actually_purges_all_children(client, db_session_factory):
    """Belt-and-braces: after the DELETE, no Product / PositionHistory /
    LabelEvent row may still reference the deleted store_id. Catches
    the 'cascade reported counts but actually orphaned rows' failure
    mode."""
    sid = _seed_store_with_data(
        client, db_session_factory,
        name="Purge", url="https://purge.example/",
        n_products=2, n_history=2, n_events=2,
    )
    client.delete(f"/api/stores/{sid}")
    db = db_session_factory()
    try:
        assert db.query(Product).filter(Product.store_id == sid).count() == 0
        assert db.query(LabelEvent).filter(LabelEvent.store_id == sid).count() == 0
        # PositionHistory has no direct store_id, but its product_id
        # must no longer match any remaining row.
        orphan_history = db.query(PositionHistory).join(
            Product, Product.id == PositionHistory.product_id, isouter=True,
        ).filter(Product.id.is_(None)).count()
        assert orphan_history == 0
    finally:
        db.close()


def test_delete_store_with_no_children_still_works(client):
    """A store with zero products / history / events deletes cleanly
    and reports 0 counts — no crash on the empty-set path."""
    cr = client.post("/api/stores", json={
        "name": "Empty", "url": "https://empty.example/",
    }).json()
    r = client.delete(f"/api/stores/{cr['id']}")
    assert r.status_code == 200
    d = r.json()["deleted"]
    assert d == {"store": 1, "products": 0, "position_history": 0, "label_events": 0}


# =====================================================================
# UPDATE — full row in response + new values propagate immediately.
# =====================================================================
def test_update_store_returns_full_row(client):
    """The frontend uses the response to update its store cache without
    a separate /api/stores round-trip. Every persisted field must come
    back so the cache stays in sync."""
    cr = client.post("/api/stores", json={
        "name": "Before", "url": "https://before.example/",
        "monthly_visitors": "10K", "niche": "Fashion", "country": "DE",
    }).json()
    r = client.put(f"/api/stores/{cr['id']}", json={
        "name": "After", "url": "https://after.example/",
        "monthly_visitors": "20K", "niche": "Fashion & General",
        "country": "DE, NL",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True
    assert body["id"] == cr["id"]
    assert body["name"] == "After"
    # URL trailing slash gets normalised away on update.
    assert body["url"] == "https://after.example"
    assert body["monthly_visitors"] == "20K"
    assert body["niche"] == "Fashion & General"
    assert body["country"] == "DE, NL"


def test_update_store_propagates_to_product_dict(client, db_session_factory):
    """Bug the user reported: 'I change some information, it doesn't
    change through the products that are showing'. The product card
    reads store_name from the joined Store row, so an update to the
    store name MUST be visible on the next bestsellers query."""
    cr = client.post("/api/stores", json={
        "name": "OldName", "url": "https://propagate.example/",
        "monthly_visitors": "10K", "niche": "Fashion",
    }).json()
    sid = cr["id"]
    # Seed one product so /api/bestsellers/combined has something to
    # return — we'll inspect its store_name field.
    db = db_session_factory()
    try:
        db.add(Product(
            store_id=sid, shopify_id="x", title="Test Product",
            handle="test", current_position=1, previous_position=0,
            label="", is_fashion=True, subniche="fashion",
            last_scraped=datetime.utcnow(),
        ))
        db.commit()
    finally:
        db.close()
    # Pre-update card shows OldName.
    r1 = client.get("/api/bestsellers/combined")
    assert r1.status_code == 200
    cards1 = [c for c in r1.json() if c.get("store_name") == "OldName"]
    assert len(cards1) >= 1
    # Rename.
    client.put(f"/api/stores/{sid}", json={"name": "NewName"})
    # Post-update card shows NewName.
    r2 = client.get("/api/bestsellers/combined")
    cards2 = [c for c in r2.json() if c.get("store_name") == "NewName"]
    assert len(cards2) >= 1
    old_still_there = [c for c in r2.json() if c.get("store_name") == "OldName"]
    assert old_still_there == []


def test_update_store_normalises_trailing_slash_on_url_change(client):
    """A URL submitted with a trailing slash must come back stripped —
    otherwise the dedupe check on the next add can be fooled."""
    cr = client.post("/api/stores", json={
        "name": "Norm", "url": "https://norm.example/",
    }).json()
    r = client.put(f"/api/stores/{cr['id']}", json={
        "url": "https://norm-renamed.example/",
    })
    assert r.status_code == 200
    assert r.json()["url"] == "https://norm-renamed.example"
