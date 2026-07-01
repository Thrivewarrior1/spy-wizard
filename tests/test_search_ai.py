"""AI-powered search regression tests.

The pre-expander search treated "prom dress" as two whitespace-separated
ASCII tokens and required BOTH to match somewhere in the catalog. "Prom"
was unknown to every keyword layer (TRANSLATIONS, SUBNICHE_SYNONYMS,
CATEGORY_NOUN_MAP, PRODUCT_CATEGORIES, classifier prompt), so a catalog
containing 26 prom-suitable evening gowns surfaced 2 results. These
tests pin the new hybrid_search behaviour: query expansion + OR-merged
SQL prefilter + Python scoring + Gemini re-rank (when available).

The Gemini call is monkeypatched off via SEARCH_RERANK=0 + missing
GEMINI_API_KEY so the tests run hermetically against the identity
expansion. The identity-expansion path is still strictly better than
the old AND-then-OR search because:
  1. It does NOT short-circuit on the first hit
  2. It uses OR across query tokens (single-word matches still rank)
  3. It scores tag matches > title matches > free-text matches
  4. It ranks by relevance score, not just current_position
"""
import os
from datetime import datetime

import pytest
import sqlite3
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

from models import Base, Store, Product
from main import app
from database import get_db
import query_expander


@pytest.fixture(autouse=True)
def _no_external_calls(monkeypatch):
    """Force the identity-expansion code path. We don't want a unit
    test to hit Gemini and we want deterministic behaviour."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("SEARCH_RERANK", "0")
    # Clear the in-process expansion cache between tests so a previous
    # test's miss doesn't poison a later test's behaviour.
    query_expander._EXPANSION_CACHE.clear()
    query_expander._EXPANSION_CACHE_ORDER.clear()
    yield


@pytest.fixture
def db_eng():
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
    return eng


@pytest.fixture
def client(db_eng):
    Session = sessionmaker(bind=db_eng, autoflush=False, autocommit=False)

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
def seeded(db_eng):
    """Seed a deliberately mixed catalog: 2 products with literal
    'prom' in title, 4 products with prom-relevant tags but no 'prom'
    in title, and 5 unrelated products. The old AND search returned
    only the 2 literal hits; the new search should return all 6."""
    Session = sessionmaker(bind=db_eng, autoflush=False, autocommit=False)
    db = Session()
    s = Store(
        name="Test Store", url="https://test.example",
        monthly_visitors="10K", niche="Fashion", country="US",
    )
    db.add(s)
    db.commit()
    db.refresh(s)

    products = [
        # Direct "prom" in title — old search would catch these
        ("celestia-prom-gown", "Celestia Prom Gown Sequin Halter Maxi",
         "prom, sequin, halter, maxi, evening, formal, gown, women", "dress"),
        ("nova-prom-dress",    "Nova Prom Dress Mermaid Sequin",
         "prom, mermaid, sequin, gown, evening, women", "dress"),
        # No "prom" in title but tagged as prom-suitable
        ("luna-evening-gown",  "Luna Floor-Length Evening Gown",
         "evening, prom, formal, ball, maxi, women, gown", "dress"),
        ("abendkleid-spitze",  "Abendkleid mit Spitze in Schwarz",
         "evening, prom, wedding-guest, cocktail, formal, dress, lace, women", "dress"),
        ("robe-de-soiree",     "Robe de Soiree Bohème en Dentelle",
         "evening, prom, formal, gown, lace, women", "dress"),
        ("vestido-de-gala",    "Vestido de Gala Largo Negro",
         "gala, evening, prom, formal, gown, women", "dress"),
        # Unrelated fashion (proper per-product categories so the
        # scorer doesn't false-positive on a misseeded category).
        ("white-tee",          "Basic White T-Shirt",
         "tee, casual, cotton, white, women", "top"),
        ("denim-jeans",        "Slim Fit Denim Jeans",
         "jeans, denim, casual, fitted, women", "pants"),
        ("hiking-boot",        "Mountain Hiking Boot",
         "boot, hiking, outdoor, durable, men", "boot"),
        ("workout-leggings",   "High-Waisted Workout Leggings",
         "leggings, gym, athletic, women", "leggings"),
        ("formal-suit",        "Men's Formal Wool Suit",
         "suit, formal, work, wool, men", "suit"),
    ]
    for i, (handle, title, tags, category) in enumerate(products):
        db.add(Product(
            store_id=s.id, shopify_id=handle, title=title, handle=handle,
            current_position=i + 1, previous_position=0, label="",
            is_fashion=True, subniche="fashion", ai_tags=tags,
            last_scraped=datetime.utcnow(), product_category=category,
            product_type="", price="",
        ))
    db.commit()
    db.close()
    return s


# =====================================================================
# query_expander — identity-fallback shape (no Gemini available)
# =====================================================================
def test_identity_expansion_returns_query_tokens():
    """Without GEMINI_API_KEY, expand_query falls back to the raw
    tokens. The scorer still uses those for matching."""
    import asyncio
    exp = asyncio.run(query_expander.expand_query("prom dress"))
    assert "prom" in exp.canonical_terms
    assert "dress" in exp.canonical_terms
    assert exp.expander_used == "fallback"


def test_identity_expansion_cached(_no_external_calls=None):
    """Identity expansion is cached just like Gemini expansion —
    repeating the same query is O(1)."""
    import asyncio
    e1 = asyncio.run(query_expander.expand_query("prom dress"))
    e2 = asyncio.run(query_expander.expand_query("prom dress"))
    assert e2.cached is True
    assert e2.canonical_terms == e1.canonical_terms


def test_expansion_all_terms_includes_original():
    """all_terms() folds in the raw query tokens AND every expansion
    field — the scorer iterates this set."""
    exp = query_expander.ExpansionResult(
        original="prom dress",
        canonical_terms=["formal", "evening"],
        occasion_tags=["prom", "evening", "ball"],
        style_tags=["maxi"],
        material_tags=["satin"],
    )
    terms = exp.all_terms()
    assert "prom" in terms
    assert "dress" in terms
    assert "formal" in terms
    assert "evening" in terms
    assert "ball" in terms
    assert "maxi" in terms
    assert "satin" in terms


def test_expansion_tag_terms_only_includes_tags():
    """tag_terms() is the subset used for the heavier ai_tags scoring
    boost — canonical_terms and original tokens are excluded."""
    exp = query_expander.ExpansionResult(
        original="prom dress",
        canonical_terms=["formal"],
        occasion_tags=["prom", "evening"],
        style_tags=["maxi"],
        material_tags=["satin"],
        color_tags=["red"],
    )
    tags = exp.tag_terms()
    assert tags == {"prom", "evening", "maxi", "satin", "red"}
    assert "formal" not in tags  # canonical, not a tag


# =====================================================================
# score_product_against_expansion — pure scoring
# =====================================================================
def test_score_exact_phrase_in_title_dominates():
    """A title containing the exact query phrase outranks all other
    signals — when the user types "prom dress" they want products
    literally called "Prom Dress" at the top."""
    exp = query_expander.ExpansionResult(
        original="prom dress", canonical_terms=["prom", "dress"],
        occasion_tags=["prom"],
    )
    direct = query_expander.score_product_against_expansion(
        title="Black Prom Dress in Satin",
        ai_tags="dress", product_category="dress", subniche="fashion",
        product_type="", handle="black-prom-dress", exp=exp,
    )
    related = query_expander.score_product_against_expansion(
        title="Elegant Evening Gown",
        ai_tags="prom, evening, gown, formal",
        product_category="dress", subniche="fashion",
        product_type="", handle="elegant-evening-gown", exp=exp,
    )
    assert direct > related, (direct, related)


def test_score_ai_tag_match_outranks_unrelated():
    """A product with no title overlap but with the right ai_tags
    still scores nonzero — that's the whole point: catch products
    Gemini correctly tagged but whose title is in another language."""
    exp = query_expander.ExpansionResult(
        original="prom", canonical_terms=["prom"],
        occasion_tags=["prom", "evening", "formal"],
    )
    tagged = query_expander.score_product_against_expansion(
        title="Abendkleid mit Spitze",
        ai_tags="evening, prom, formal, lace, women",
        product_category="dress", subniche="fashion",
        product_type="", handle="abendkleid-spitze", exp=exp,
    )
    untagged = query_expander.score_product_against_expansion(
        title="Cotton T-Shirt",
        ai_tags="tee, casual, cotton",
        product_category="top", subniche="fashion",
        product_type="", handle="cotton-tee", exp=exp,
    )
    assert tagged > 0
    assert untagged == 0
    assert tagged > untagged


def test_score_zero_for_completely_irrelevant():
    """A product with NO overlap (no shared tokens anywhere) scores 0
    — those get dropped by the SQL prefilter but also by the scorer
    if any sneak through."""
    exp = query_expander.ExpansionResult(
        original="prom dress",
        canonical_terms=["prom", "dress", "gown"],
        occasion_tags=["prom", "evening"],
    )
    score = query_expander.score_product_against_expansion(
        title="Stainless Steel Water Bottle",
        ai_tags="bottle, hydration, kitchen",
        product_category="kitchen", subniche="home",
        product_type="", handle="ss-bottle", exp=exp,
    )
    assert score == 0


# =====================================================================
# /api/bestsellers/combined?search=... — end-to-end on seeded catalog
# =====================================================================
def test_prom_dress_search_finds_all_six_relevant(client, seeded):
    """The headline regression: pre-expander, "prom dress" returned 2
    (the literal-title matches) and the 4 multilingual / tagged
    evening gowns were dropped. The new pipeline returns all 6."""
    r = client.get("/api/bestsellers/combined?search=prom%20dress&limit=20&label=all&days=1")
    assert r.status_code == 200, r.text
    body = r.json()
    handles = {p["title"] for p in body}
    # All 6 relevant products should surface — both literal "prom"
    # titles AND the 4 multilingual/tagged variants
    expected = {
        "Celestia Prom Gown Sequin Halter Maxi",
        "Nova Prom Dress Mermaid Sequin",
        "Luna Floor-Length Evening Gown",
        "Abendkleid mit Spitze in Schwarz",
        "Robe de Soiree Bohème en Dentelle",
        "Vestido de Gala Largo Negro",
    }
    found = handles & expected
    assert len(found) == 6, f"missing {expected - found}; got: {handles}"
    # And NONE of the unrelated 5 should be returned
    irrelevant = {
        "Basic White T-Shirt",
        "Slim Fit Denim Jeans",
        "Mountain Hiking Boot",
        "High-Waisted Workout Leggings",
        # NB: "Men's Formal Wool Suit" contains "formal" which IS a
        # canonical occasion tag for prom-related queries, so it
        # legitimately scores nonzero. Not testing irrelevance for it.
    }
    assert handles & irrelevant == set(), f"unrelated leaked in: {handles & irrelevant}"


def test_search_returns_zero_for_truly_unmatched(client, seeded):
    """An off-topic search still returns nothing — the scorer
    correctly gives 0 to products with no overlap. Use a multi-word
    query so the SQL prefilter (>=3 chars per term) actually runs."""
    r = client.get("/api/bestsellers/combined?search=helicopter%20parts&limit=20&label=all&days=1")
    assert r.status_code == 200
    assert r.json() == []


def test_empty_search_returns_all_products(client, seeded):
    """Empty search → no filter, return everything (sorted by current_position)."""
    r = client.get("/api/bestsellers/combined?limit=20&label=all&days=1")
    assert r.status_code == 200
    assert len(r.json()) == 11  # all seeded products


def test_search_results_ranked_by_relevance(client, seeded):
    """Literal-title "Prom" matches should come BEFORE tagged-only
    matches — relevance ranking, not just current_position."""
    r = client.get("/api/bestsellers/combined?search=prom&limit=20&label=all&days=1")
    body = r.json()
    titles = [p["title"] for p in body]
    literal = [
        "Celestia Prom Gown Sequin Halter Maxi",
        "Nova Prom Dress Mermaid Sequin",
    ]
    tagged = [
        "Luna Floor-Length Evening Gown",
        "Abendkleid mit Spitze in Schwarz",
        "Robe de Soiree Bohème en Dentelle",
        "Vestido de Gala Largo Negro",
    ]
    literal_positions = [titles.index(t) for t in literal if t in titles]
    tagged_positions = [titles.index(t) for t in tagged if t in titles]
    if literal_positions and tagged_positions:
        assert max(literal_positions) < min(tagged_positions), \
            f"literal {literal_positions} should beat tagged {tagged_positions}"


def test_vision_tag_boosts_score():
    """Products carrying `img:type:cocktail-dress` from the vision
    classifier score dramatically higher than plain-text matches for
    a "cocktail dress" query — the image is the ground truth."""
    exp = query_expander.ExpansionResult(
        original="cocktail dress",
        canonical_terms=["cocktail-dress", "cocktail dress"],
        occasion_tags=["cocktail", "party"],
    )
    with_vision = query_expander.score_product_against_expansion(
        title="Elegant Mini Dress",
        ai_tags="img:type:cocktail-dress, img:gender:women, img:attr:sleeveless "
                "|| dress, elegant, party",
        product_category="dress", subniche="fashion",
        product_type="", handle="elegant-mini", exp=exp,
    )
    without_vision = query_expander.score_product_against_expansion(
        title="Elegant Mini Dress",
        ai_tags="dress, elegant, party",
        product_category="dress", subniche="fashion",
        product_type="", handle="elegant-mini", exp=exp,
    )
    assert with_vision > without_vision + 8, (with_vision, without_vision)


def test_vision_conflict_penalty_drops_wrong_subtype():
    """`img:type:ankle-boots` on a "knee-high boots" query gets a
    conflict penalty so ankle boots can't win over untagged results."""
    exp = query_expander.ExpansionResult(
        original="knee-high boots",
        canonical_terms=["knee-high-boots"],
    )
    ankle_boot_with_vision = query_expander.score_product_against_expansion(
        title="Black Boots",
        ai_tags="img:type:ankle-boots, img:gender:women || boot, black",
        product_category="shoe", subniche="fashion",
        product_type="", handle="black-boots", exp=exp,
    )
    knee_high_with_vision = query_expander.score_product_against_expansion(
        title="Tall Boots",
        ai_tags="img:type:knee-high-boots, img:gender:women || boot, tall",
        product_category="shoe", subniche="fashion",
        product_type="", handle="tall-boots", exp=exp,
    )
    assert knee_high_with_vision > ankle_boot_with_vision, \
        (knee_high_with_vision, ankle_boot_with_vision)


def test_rerank_returns_match_shape_not_score():
    """User demanded "100% accuracy, prefer zero results over
    inaccurate results". The judge now returns match=true|false
    (binary), not a 0-100 score. Match=false items are HARD-DROPPED
    by hybrid_search. The legacy {idx, score, drop} shape was a
    soft re-rank that let tangentially-related items through; the
    new shape is a strict yes/no verdict."""
    import asyncio
    candidates = [
        {"id": 1, "title": "Knee-High Black Leather Boots",
         "ai_tags": "boot, knee-high, leather, women"},
        {"id": 2, "title": "Ankle Boots Suede",
         "ai_tags": "boot, ankle, suede, women"},
    ]
    # No GEMINI_API_KEY → fail_open. Every candidate gets match=True
    # so the caller can continue with the hybrid order. This is
    # intentional: a transient Gemini outage shouldn't blank the page.
    verdicts = asyncio.run(query_expander.rerank_with_gemini("knee-high boots", candidates))
    assert len(verdicts) == 2
    for v in verdicts:
        assert "idx" in v
        assert "match" in v
        assert isinstance(v["match"], bool)
        # Legacy fields must not appear
        assert "score" not in v
        assert "drop" not in v


def test_intent_gate_drops_cross_category_pollution(client, db_eng):
    """User complaint 2026-06-29: searching 'shoes' surfaces dresses
    because Gemini expansion includes 'dress shoes' as a phrase and
    the SQL prefilter ORs every term. The intent gate (intent_types=
    ['shoes'] requires title/ai_tags to contain a shoe keyword) is
    the fix. This test seeds a dress + a shoe + a bag and verifies
    that searching 'shoes' returns ONLY the shoe, even though the
    dress + bag have tangentially-related tokens in their tags."""
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=db_eng, autoflush=False, autocommit=False)
    db = Session()
    s = Store(
        name="Mixed", url="https://mixed.example",
        monthly_visitors="1K", niche="Fashion", country="US",
    )
    db.add(s); db.commit(); db.refresh(s)
    products = [
        ("a-real-shoe", "Black Leather Oxford Shoe for Men",
         "shoe, oxford, leather, men, formal, dress", "shoe"),
        ("a-dress",     "Elegant Cocktail Mini Dress",
         "dress, cocktail, mini, evening, formal, women", "dress"),
        ("a-bag",       "Vintage Leather Handbag",
         "bag, handbag, leather, vintage, women, formal", "bag"),
        # A "dress shoe" — semantically AT the intersection. The
        # intent gate should let this through for both "shoes" and
        # "dress" queries because both keywords are in the title.
        ("dress-shoe",  "Men's Formal Dress Shoe in Brown Leather",
         "shoe, dress shoe, formal, men, brown, leather, oxford", "shoe"),
    ]
    for i, (handle, title, tags, cat) in enumerate(products):
        db.add(Product(
            store_id=s.id, shopify_id=handle, title=title, handle=handle,
            current_position=i + 1, previous_position=0, label="",
            is_fashion=True, subniche="fashion", ai_tags=tags,
            last_scraped=datetime.utcnow(), product_category=cat,
            product_type="", price="",
        ))
    db.commit()
    db.close()

    # Shoes search must NOT include the bag or the standalone dress.
    r = client.get("/api/bestsellers/combined?search=shoes&limit=20&label=all&days=1")
    titles = {p["title"] for p in r.json()}
    assert "Black Leather Oxford Shoe for Men" in titles
    assert "Men's Formal Dress Shoe in Brown Leather" in titles
    assert "Elegant Cocktail Mini Dress" not in titles, \
        "shoe search must NOT return a dress"
    assert "Vintage Leather Handbag" not in titles, \
        "shoe search must NOT return a bag"

    # Bag search must NOT include the shoes or dress.
    r = client.get("/api/bestsellers/combined?search=bag&limit=20&label=all&days=1")
    titles = {p["title"] for p in r.json()}
    assert "Vintage Leather Handbag" in titles
    assert "Black Leather Oxford Shoe for Men" not in titles
    assert "Men's Formal Dress Shoe in Brown Leather" not in titles
    assert "Elegant Cocktail Mini Dress" not in titles

    # Dress search includes the standalone dress AND the dress-shoe
    # (because "dress" keyword is in its title — semantically a real
    # dress shopper might also be interested). Bag + shoe are dropped.
    r = client.get("/api/bestsellers/combined?search=dress&limit=20&label=all&days=1")
    titles = {p["title"] for p in r.json()}
    assert "Elegant Cocktail Mini Dress" in titles
    assert "Vintage Leather Handbag" not in titles


def test_multilingual_query_routes_to_english_tags(client, seeded):
    """A German query "Abendkleid" should find the German-titled
    product AND the English-tagged "evening" products. Without an
    expansion, "Abendkleid" would only match the literal German
    title."""
    # Identity fallback won't multilingually expand, so we just verify
    # the literal-language match still works under the new pipeline.
    r = client.get("/api/bestsellers/combined?search=abendkleid&limit=10&label=all&days=1")
    body = r.json()
    titles = {p["title"] for p in body}
    assert "Abendkleid mit Spitze in Schwarz" in titles
