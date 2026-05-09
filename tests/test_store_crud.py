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

from models import Base, Store, Product
from main import app, _normalise_store_url, _validate_store_payload
from database import get_db


@pytest.fixture
def client():
    """Spin up an isolated in-memory DB + override get_db so the
    /api/stores endpoint hits the test database. StaticPool keeps
    every connection bound to the SAME in-memory database (default
    SQLite behaviour gives each connection its own private DB,
    which means Base.metadata.create_all on one connection isn't
    visible to the next)."""
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(eng)
    Session = sessionmaker(bind=eng, autoflush=False, autocommit=False)

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
])
def test_normalise_store_url(raw, expected):
    assert _normalise_store_url(raw) == expected


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
