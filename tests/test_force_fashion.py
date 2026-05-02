"""Apparel / footwear / eyewear / intimate safety-net regression tests.

The General tab kept leaking obvious fashion items even after the
'wearables-to-fashion' migration shipped — bathrobes were tagged 'home',
underwear 'beauty', orthopedic shoes 'health', wedding-guest dresses
'other'. The user pulled a screenshot showing 10 specific examples and
asked for a multilingual hard-coded allowlist that forces is_fashion=True
regardless of what Gemini said, plus a sweep migration that fixes the
existing DB rows.

These tests pin both halves:
  - parametrised regression: each visible-from-the-screenshot title MUST
    be flagged by `_is_forced_fashion` so the per-scrape distribution
    and the migration both promote it.
  - negative regression: a sample of titles that legitimately belong on
    the General tab MUST NOT match, so the safety net doesn't drag
    home/electronics/beauty rows onto Fashion.
  - migration end-to-end: rows already in the DB with is_fashion=False
    get promoted to Fashion on startup, with subniche rewritten to
    'fashion' if Gemini had previously bucketed them as
    electronics/home/beauty/health/other.
"""
from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models import Base, Store, Product
from scraper import (
    FORCE_FASHION_TITLE_RE,
    FORCE_FASHION_BH_RE,
    _is_forced_fashion,
    migrate_apparel_to_fashion,
    update_products_in_db,
)


# === Regression: titles visible on the live General tab today MUST
# === be classified as fashion by `_is_forced_fashion`. The list is
# === pulled directly from the user's screenshot — adding to it is
# === fine, but never delete an entry without confirming the regex
# === actually still catches the spirit of that case via something else.
_USER_SCREENSHOT_TITLES = [
    # 1. Bathrobe (German)
    "Luxus Bademantel - Damen - Weiche Kapuze und Praktische Details",
    # 2. Orthopedic shoes (German compound — Schuh suffix)
    "Plantarfasziitis-Linderung Orthoschuh-Set",
    # 3. Seamless underwear (German)
    "R&B - Seidenweiche Nahtlose Unterwäsche",
    # 4. Lingerie/satin pack — caught via 'nahtlos' even though title
    #    is otherwise opaque
    "5 Stück nahtloses Satin Soft - Vanee",
    # 5. Wedding-guest dress
    "für Hochzeitsgäste",
    # 6. Orthopedic slip-ons
    "Hochwertige orthopädische Slip-ons",
    # 7. Bare-word German shoe
    "Norla - Bequemer und eleganter Schuh",
    # 8. Truncated brand token from a sneaker leaker
    "von Sakin",
    # 9. German bra abbreviation (BH)
    "R&B | Flexibler Multiway-Rücken-BH für Damen Mia",
    # 10. Eyewear (English)
    "German Intelligent Progressive Glasses for Clear Vision",
]


@pytest.mark.parametrize("title", _USER_SCREENSHOT_TITLES)
def test_screenshot_titles_are_forced_fashion(title):
    assert _is_forced_fashion(title=title), (
        f"_is_forced_fashion did NOT flag {title!r}. The user's screenshot "
        "explicitly named this as a General-tab leaker that must move to "
        "Fashion — adjust _FORCE_FASHION_PATTERNS so the safety net catches it."
    )


# === Additional regression coverage from the live snapshot scan
# === (https://spy-wizard-production.up.railway.app/api/general/combined).
# === These titles also live on the General tab today and must promote.
_LIVE_SNAPSHOT_LEAKERS = [
    "Veste Courte Femme Ample et Polyvalente Doublée de Fourrure",   # FR jacket
    "Manteau long en laine à col montant et double boutonnage",      # FR coat
    "3-in-1 Waterproof Outdoor Rain Poncho",                          # poncho
    "The Hungerford Men's Grey Fleece Hooded Bathrobe with Belt and Pockets",
    "The Kenilworth Men's Hooded Fleece Long Bathrobe with Belt",
    "The Eastbourne Men's Cotton Waffle Lightweight Spa Bathoobe with Belt",  # typo
    "Wasserdichter und unzerstörbarer Schuhüberzug",                  # DE compound shoe
    "von Salkin",                                                     # truncated brand
    "BowLift | Men's Magnetic Massage Underwear | Supportive Comfort Design",
    # Traditional / cultural / kids-wearable survivors that slipped past
    # the Gemini-driven reclassify endpoint and required explicit regex
    # pinning. Pin them here so any future regex regression fails loud.
    "Ilsa - Authentisch Traditionelles Oktoberfest Ensemble",
    "Foldable Kids' Sleeping Bag in Animal Shape – Snoozi",
]


@pytest.mark.parametrize("title", _LIVE_SNAPSHOT_LEAKERS)
def test_live_snapshot_leakers_are_forced_fashion(title):
    assert _is_forced_fashion(title=title), (
        f"Live snapshot leaker {title!r} not caught — broaden patterns."
    )


# === Negative regression: General-tab products that genuinely belong
# === there must NOT be promoted by the safety net. False positives are
# === preferable to false negatives per the user, but a few obvious
# === non-fashion classes (lamps, candles, costumes, foot-pain pads,
# === hairtools) must stay General to keep the tab useful at all.
_LEGITIMATE_GENERAL_TITLES = [
    "Mysaglobe Scandinavian Round Ceiling Light",
    "Danish Designer Table Lamp",
    "Bauhaus Colored Glass Pendant Light",
    "Luxury Goose Down Bed Pillow – Hypoallergenic Neck Support Cushion for Deep Sleep",
    "Portable SSD External Drive with High-Speed Data Transfer and Compact Design",
    "BowLift | Smart Robot",
    "Cordless Hair Clipper for Precision Grooming and Styling",
    "BowLift | Realistic Latex Mask | Halloween Cosplay Costume | Full Face",
    "BowLift | Realistic Superhero Costume for Kids & Adults",
    "Large Dog Skeleton Bodysuit Costume",
    "ComfortStep Morton's Neuroma and Metatarsalgia Relief Pads | Gel Cushion and Metatarsal Support",
    "BowLift | Hands-Free Magnifying Glass with Neck Strap - 5X Magnification for Reading",
    "Elegante Herren Uhrenbox",
    "BowLift | Hip and Thigh Support Band for Pain Relief",
    "Bathroom Bar Wall Light For Above Mirror",
    "Bowling-style Pin Decoration",
    "BowLift | Snap-On Cosmetic Veneers",
    "BowLift | Mustang Car Shaped Whiskey Decanter",
    "Acrylic Light-Up Reindeer Lawn Ornament",
]


@pytest.mark.parametrize("title", _LEGITIMATE_GENERAL_TITLES)
def test_general_titles_are_not_forced_fashion(title):
    assert not _is_forced_fashion(title=title), (
        f"_is_forced_fashion incorrectly flagged {title!r}. The pattern is "
        "too broad — tighten _FORCE_FASHION_PATTERNS to avoid this leak."
    )


def test_force_fashion_re_does_not_match_garden_hose_compounds_unfortunately():
    """Documenting a known FP we accept: bare 'Hose' (German for pants)
    is included so 'Damen-Hose' / 'Hosen' get promoted, but it also
    matches 'Hose' as a substring in some English compounds. The user
    explicitly traded this risk for catching German pants — keep the
    pattern wide.
    """
    # Sanity check: the bare German word does match.
    assert _is_forced_fashion(title="Schwarze Hose Damen")
    # And the plural compound matches via the \w* suffix.
    assert _is_forced_fashion(title="Jogginghosen Herren Set")


def test_bh_pattern_matches_bare_word_and_dotted_form():
    assert FORCE_FASHION_BH_RE.search("Multiway-Rücken-BH für Damen")
    assert FORCE_FASHION_BH_RE.search("B.H. Damen")
    # Should NOT match arbitrary 2-letter sequences.
    assert not FORCE_FASHION_BH_RE.search("OBHaus Lampe")
    assert not FORCE_FASHION_BH_RE.search("ABH-Mode Sale")


def test_force_fashion_re_handles_url_and_image_fields():
    """The safety net must also fire on handle / product_url / image
    URL — disguised titles ('Mia Premium Item') sometimes only reveal
    their category through the slug or CDN path."""
    assert _is_forced_fashion(handle="luxus-bademantel-damen")
    assert _is_forced_fashion(
        product_url="https://example.com/products/orthoschuh-set",
    )
    assert _is_forced_fashion(
        image_url="https://cdn.shopify.com/files/seidenweiche-unterwaesche.jpg",
    )
    # Negative: a URL that mentions a candle slug is not fashion.
    assert not _is_forced_fashion(
        handle="bauhaus-pendant-light",
        product_url="https://example.com/products/bauhaus-pendant-light",
        image_url="https://cdn.shopify.com/files/pendant-light.jpg",
    )


# === DB migration end-to-end ===
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


def _seed_general(db, store, handle, title, subniche):
    db.add(Product(
        store_id=store.id, shopify_id=handle, title=title, handle=handle,
        image_url="", price="", vendor="", product_type="", product_url="",
        current_position=1, previous_position=0, label="",
        ai_tags="", is_fashion=False, subniche=subniche,
        last_scraped=datetime.utcnow(),
    ))
    db.commit()


def test_migrate_apparel_promotes_legacy_general_rows(db, store):
    """Seed a handful of representative leakers and confirm the
    migration flips them to Fashion AND rewrites their subniche."""
    cases = [
        ("legacy-bademantel", "R&B | Luxus Bademantel - Damen", "home"),
        ("legacy-orthoschuh", "Plantarfasziitis-Linderung Orthoschuh-Set", "health"),
        ("legacy-bra",        "R&B | Multiway-Rücken-BH für Damen", "other"),
        ("legacy-glasses",    "German Intelligent Progressive Glasses", "other"),
        ("legacy-poncho",     "3-in-1 Waterproof Outdoor Rain Poncho", "other"),
    ]
    for handle, title, sub in cases:
        _seed_general(db, store, handle, title, sub)

    promoted = migrate_apparel_to_fashion(db)
    assert promoted == len(cases)

    for handle, _, _ in cases:
        p = db.query(Product).filter(Product.shopify_id == handle).one()
        assert p.is_fashion is True
        assert p.subniche == "fashion"


def test_migrate_apparel_preserves_existing_wearable_subniche(db, store):
    """A row already labelled 'jewelry' / 'accessories' / 'bags' must
    keep its label even when the title triggers the apparel safety
    net — the wearable label is more specific and surfaces useful
    backend-search metadata."""
    _seed_general(db, store, "earring-with-shoe-charm",
                  "Gold earring with shoe-shaped charm", "jewelry")
    migrate_apparel_to_fashion(db)
    p = db.query(Product).filter(Product.shopify_id == "earring-with-shoe-charm").one()
    assert p.is_fashion is True
    assert p.subniche == "jewelry", "wearable subniche must be preserved"


def test_migrate_apparel_skips_legitimate_general_rows(db, store):
    """Lamps, costumes, hair tools must stay on General after the
    migration runs — the safety net should not paint everything as
    fashion."""
    safe_cases = [
        ("legacy-lamp", "Bauhaus Colored Glass Pendant Light", "home"),
        ("legacy-costume", "BowLift | Realistic Superhero Costume", "other"),
        ("legacy-hairtool", "Cordless Hair Clipper for Styling", "beauty"),
    ]
    for handle, title, sub in safe_cases:
        _seed_general(db, store, handle, title, sub)

    promoted = migrate_apparel_to_fashion(db)
    assert promoted == 0
    for handle, _, expected_sub in safe_cases:
        p = db.query(Product).filter(Product.shopify_id == handle).one()
        assert p.is_fashion is False
        assert p.subniche == expected_sub


def test_migrate_apparel_is_idempotent(db, store):
    _seed_general(db, store, "x-bademantel", "Luxus Bademantel Damen", "home")
    assert migrate_apparel_to_fashion(db) == 1
    assert migrate_apparel_to_fashion(db) == 0


def test_per_scrape_sweep_promotes_existing_apparel(db, store):
    """update_products_in_db runs an inline apparel sweep on every
    scrape, so even partial scrapes (no fashion / no general lists)
    can fix legacy rows that match the safety net."""
    _seed_general(db, store, "dormant-bra",
                  "R&B | Flexibler Multiway-Rücken-BH für Damen Mia", "beauty")
    update_products_in_db(db, store, fashion_products=[], general_products=[])
    p = db.query(Product).filter(Product.shopify_id == "dormant-bra").one()
    assert p.is_fashion is True
    assert p.subniche == "fashion"
