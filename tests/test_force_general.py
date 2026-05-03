"""FORCE_GENERAL safety-net regression tests.

The user's rule (also documented at the top of classifier.py):

    'Is this product chosen primarily because of how it makes you LOOK
     (style/aesthetic), or for what it DOES (function/medical/safety/
     utility)? If function dominates, it is General even if you wear it.'

`_is_forced_general` implements the function-driven side of that line.
Items it matches MUST land on the General feed even if their title also
matches the apparel allowlist (e.g. 'smartwatch' matches both \\bwatch
and smartwatch — General wins). The migration / per-scrape sweep run
the same check so any wearable-gadget that previously got promoted to
Fashion gets pulled back when the rule shifts.

These tests pin both halves:
  - parametrised positives: every category the user listed (smartwatch,
    posture corrector, dog raincoat, magnifying glass, trekking pole,
    snap-on veneers, etc.) MUST flag.
  - parametrised negatives: legitimate fashion items (mechanical watch,
    decorative bandana, costume, shapewear, sunglasses) MUST NOT flag.
  - precedence: when both regexes match, FORCE_GENERAL wins.
  - migration end-to-end: existing is_fashion=True rows that match the
    allowlist get demoted on startup with subniche re-bucketed sensibly.
"""
from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models import Base, Store, Product
from scraper import (
    FORCE_GENERAL_TITLE_RE,
    _is_forced_general,
    _is_forced_fashion,
    _classify_general_subniche,
    migrate_force_general_to_general,
    update_products_in_db,
)


# === Items the USER explicitly named in the demote list ===
_USER_DEMOTE_LIST = [
    "Waterproof Dog Raincoat with Reflective Hood",
    "LumaGuard LED Dog Collar",
    "BowLift | Posture Corrector for Neck and Back",
    "BowLift | Hip and Thigh Support Band for Pain Relief",
    "ToeGuard Orthopedic Toe Spacers | Bunionette Corrector",
    "ComfortStep Morton's Neuroma and Metatarsalgia Relief Pads | Gel Cushion",
    "BowLift | Hands-Free Magnifying Glass with Neck Strap - 5X Magnification",
    "BowLift | Adjustable Trekking Poles - Lightweight & Foldable for Hiking",
    "Automatic Self Defense Walking Stick for Hiking and Urban Safety",
    "Automatic Self Defense Walking Stick with Stability Support",
    "Smartwatch",
    "Fortgeschrittene Multifunktions-Smartwatch",
    "BowLift | Senior Smartwatch with Health & Fitness Tracking",
    "BowLift | One-Click Blood Sugar|Blood Glucose Blood Pressure ECG Heart Rate Monitor Health Smart Watch",
    "BowLift | Snap-On Cosmetic Veneers | Instant Smile Cover",
]


@pytest.mark.parametrize("title", _USER_DEMOTE_LIST)
def test_user_demote_titles_are_forced_general(title):
    assert _is_forced_general(title=title), (
        f"_is_forced_general did NOT flag {title!r}. The user's spec "
        "explicitly named this as a Fashion-tab survivor that must move to "
        "General — adjust _FORCE_GENERAL_PATTERNS so the safety net catches it."
    )


# === Lighting fixtures — ALWAYS general, never fashion. The user-
# === reported bug was 'Crystal Ring Chandelier' on Fashion (the
# === classifier was fooled by 'Ring' = jewelry-sounding modifier).
# === Identify the NOUN, not the modifiers.
_LIGHTING_TITLES = [
    # English
    "Crystal Ring Chandelier",
    "Pearl Drop Pendant Light",
    "Gold Wall Sconce",
    "Rose Gold Floor Lamp",
    "Diamond Crystal Table Lamp",
    "Modern Brass Chandelier",
    "Bauhaus Pendant Light",
    "Bathroom Wall Sconce",
    "Art Deco Floor Lamp",
    "Bedside Table Lamp",
    "Antique Brass Candelabra",
    "Silver Candle Holders Set",
    "LED Fairy Lights 100ft",
    "Christmas String Lights",
    "Reading Lamp Adjustable",
    "Gold Lamp Shade",
    "Solar Garden Lights Path",
    "Ceiling Light Fixture",
    "Ring Pendant Light Modern",
    "Diamond LED Ceiling Lamp",
    # German
    "Kristall Kronleuchter Modern",
    "Goldene Tischlampe Designer",
    "Hängelampe Wohnzimmer Industrial",
    "Pendelleuchte Esstisch Schwarz",
    "LED Wandleuchte Aussen",
    "Stehlampe Modern Design",
    "Lampenschirm Stoff Beige",
    "Lichterkette LED 10m",
    "Kerzenhalter Silber 3-er Set",
    "Designerlampe Glas",
    # French
    "Lustre Cristal Moderne",
    "Suspension Bois Scandinave",
    "Plafonnier LED Salon",
    "Applique Murale Dorée",
    "Lampe de Table Vintage",
    "Lampe de Chevet Rose",
    "Abat-Jour Tissu Beige",
    "Guirlande Lumineuse Solaire",
    "Bougeoir en Laiton",
    # Spanish
    "Araña de Cristal Moderna",
    "Lámpara de Techo LED",
    "Lámpara de Mesa Vintage",
    "Lámpara de Pie Industrial",
    "Candelabro de Plata",
    # Italian
    "Lampadario in Cristallo",
    "Lampada da Tavolo Designer",
    "Lampada da Terra Moderna",
    "Candelabro in Ottone",
    "Paralume Tessuto Bianco",
    # Dutch
    "Kroonluchter Kristal Modern",
    "Plafondlamp LED Woonkamer",
    "Tafellamp Industrieel Goud",
    "Vloerlamp Hout Scandinavisch",
    "Wandlamp Buitenverlichting",
    "Kandelaar Set Zilver",
]


@pytest.mark.parametrize("title", _LIGHTING_TITLES)
def test_lighting_titles_are_forced_general(title):
    assert _is_forced_general(title=title), (
        f"_is_forced_general missed lighting fixture {title!r}. "
        "Lighting (chandeliers, lamps, sconces, candle holders, "
        "string lights, etc.) must ALWAYS be General regardless of "
        "jewelry-sounding modifiers like 'Crystal' or 'Ring'."
    )


def test_crystal_ring_chandelier_subniche_is_home():
    """The user's exact bug case: Crystal Ring Chandelier was on
    Fashion. After demote, the subniche heuristic must put it on
    'home', not 'other'."""
    assert _classify_general_subniche("Crystal Ring Chandelier") == "home"
    assert _classify_general_subniche("Pearl Drop Pendant Light") == "home"
    assert _classify_general_subniche("Gold Wall Sconce") == "home"
    assert _classify_general_subniche("Kristall Kronleuchter") == "home"
    assert _classify_general_subniche("Lustre Cristal Moderne") == "home"
    assert _classify_general_subniche("Lampadario in Cristallo") == "home"


def test_jewelry_with_lighting_lookalike_modifiers_stays_fashion():
    """Inverse check — actual jewelry that uses lighting-adjacent
    words must NOT be demoted by the lighting regex. The noun is
    'bracelet' / 'ring' / 'necklace', not 'lamp' or 'chandelier'."""
    assert not _is_forced_general(title="Diamond Gold Bracelet")
    assert not _is_forced_general(title="Crystal Ring Set Sterling Silver")
    assert not _is_forced_general(title="Pearl Drop Earrings")
    assert not _is_forced_general(title="Rose Gold Necklace Pendant")


# === Multilingual coverage — function-driven wearables in DE/FR/ES/IT/NL ===
_MULTILINGUAL_GENERAL = [
    # Posture / support / brace (DE/FR/ES/IT)
    "Haltungskorrektur Rückenstütze",
    "Stützgürtel Rücken Damen",
    "Stützband Knie Sportbandage",
    "Rückenstütze Posture Brace",
    # Orthopedic insoles / toe spacers
    "Orthopädische Einlagen Damen",
    "Zehenspreizer Silikon Hallux Valgus",
    # Magnifying glass
    "Lupe mit Beleuchtung",
    "Lesehilfe Vergrößerungsglas",
    # Trekking pole / walking stick
    "Wanderstock aus Carbon",
    "Spazierstock klappbar",
    # Self defense
    "Selbstverteidigung Stock",
    # Smartwatch / fitness tracker
    "Smart Watch mit Herzfrequenz",
    "Fitness Tracker mit Pulsmesser",
    "Smart Ring Health Monitor",
    # Pet protective gear
    "Hundemantel wasserdicht",
    "Hundegeschirr verstellbar Größe L",
    "Maulkorb für große Hunde",
    "Hundeleine reflektierend",
    # Umbrella
    "Regenschirm automatisch",
    # Compression hosiery (medical)
    "Compression Sleeve Knee Support",
    "Kompressionsstrümpfe Klasse 2 Medizinisch",
]


@pytest.mark.parametrize("title", _MULTILINGUAL_GENERAL)
def test_multilingual_function_wearables_are_forced_general(title):
    assert _is_forced_general(title=title), (
        f"_is_forced_general missed multilingual {title!r}. The "
        "function-over-form rule requires this item to land on General."
    )


# === Precedence: when both FORCE_FASHION and FORCE_GENERAL match,
# === FORCE_GENERAL wins. The classic case is medical hosiery — bare
# === 'stockings' is fashion, 'compression stockings' is medical →
# === General. Same with 'orthopedic insoles' (insoles=medical even
# === though 'shoes' would push toward fashion).
_DOUBLE_MATCH_TITLES = [
    "Compression Stockings Medical Grade",  # 'stockings' + 'compression stocking'
    "Orthopedic Insoles for Shoes",         # 'shoes' + 'orthopedic insoles'
]


@pytest.mark.parametrize("title", _DOUBLE_MATCH_TITLES)
def test_force_general_wins_over_force_fashion(title):
    """When FORCE_FASHION and FORCE_GENERAL both fire, the call site
    in scraper.py checks FORCE_GENERAL first, so the row lands on
    General. Sanity-check that both regexes do match this title."""
    assert _is_forced_general(title=title), (
        f"FORCE_GENERAL must fire on {title!r} — "
        "medical insoles / compression hosiery are General."
    )
    assert _is_forced_fashion(title=title), (
        f"Sanity: {title!r} should also match FORCE_FASHION (stockings / "
        "shoes alone are fashion) — that's the precedence case."
    )


# === Negatives — STYLE-driven items must NOT be flagged as general.
# === Includes the user's "keep in fashion" list and broader style staples.
_LEGITIMATE_FASHION_TITLES = [
    # Style watches (vs smart watches)
    "Classic Mechanical Watch with Leather Band",
    "Vintage Quartz Wristwatch Stainless Steel",
    "Rolex Submariner Style Watch",
    # Decorative pet items (NOT protective gear)
    "Decorative Dog Bandana - Floral Pattern",
    "Fancy Pet Bowtie Set Wedding",
    "Small Dog Hat Sun Protection Pink",
    # Style bags (not medical / utility)
    "Leather Crossbody Handbag",
    "Make-up Bag Duo Travel",
    "Watch Holder Travel Case Leather 3 Watches",
    "Felt Bag Organiser Multi-Pocket",
    # Costumes (dress-up = fashion)
    "Realistic Superhero Costume for Kids & Adults",
    "Halloween Latex Mask Adult Full Face",
    "Dog Skeleton Bodysuit Costume",
    "Cosplay Wig Long Pink Anime",
    # Apparel staples
    "Cotton Pajamas Set Women",
    "Wool Cardigan Long Sleeve",
    "Floral Summer Maxi Dress",
    "Leather Jacket Black",
    # Intimates (shapewear is FASHION, not posture corrector)
    "Shapewear Bodysuit Tummy Control",
    "Silk Lingerie Set",
    "Seamless Underwear 5-pack",
    # Sunglasses / eyewear (style, not magnifying / smart)
    "Polarized Sunglasses Aviator",
    "Reading Glasses Tortoiseshell Frames",
    # Footwear (orthopedic SHOES are fashion; orthopedic INSOLES are general)
    "Orthopedic Slip-on Shoes Men",
    "Bequemer Orthoschuh Damen",
    "White Sneakers Leather",
]


@pytest.mark.parametrize("title", _LEGITIMATE_FASHION_TITLES)
def test_style_driven_titles_are_NOT_forced_general(title):
    assert not _is_forced_general(title=title), (
        f"_is_forced_general incorrectly flagged style-driven {title!r}. "
        "The pattern is too broad — fashion staples must not be demoted."
    )


# === The shapewear/posture-corrector boundary — both shape the body
# === but the rule splits on style vs medical. Test the exact line.
def test_shapewear_is_fashion_not_general():
    assert _is_forced_fashion(title="Tummy Control Shapewear Bodysuit")
    assert not _is_forced_general(title="Tummy Control Shapewear Bodysuit")


def test_posture_corrector_is_general_not_fashion():
    assert _is_forced_general(title="Posture Corrector for Neck and Back")
    # Note: 'corrector' alone might or might not match fashion; what
    # matters is FORCE_GENERAL fires and the precedence is correct.


# === URL/handle/image fields — the safety net must also fire on
# === Shopify slugs and CDN paths so disguised titles still demote.
def test_force_general_handles_url_and_image_fields():
    assert _is_forced_general(handle="lumaguard-led-dog-collar")
    assert _is_forced_general(handle="bowlift-posture-corrector-back-neck")
    assert _is_forced_general(handle="adjustable-trekking-poles")
    assert _is_forced_general(
        product_url="https://example.com/products/smartwatch-fitness-tracker",
    )
    assert _is_forced_general(
        image_url="https://cdn.shopify.com/files/orthopedic-toe-spacer.jpg",
    )
    # Negative: a Shopify URL for a regular dress shouldn't match.
    assert not _is_forced_general(
        handle="floral-summer-maxi-dress",
        product_url="https://example.com/products/floral-summer-maxi-dress",
    )


# === Subniche heuristics — when migrating, demoted rows should land
# === in a sensible General-feed bucket, not 'fashion' or empty.
@pytest.mark.parametrize("title,expected", [
    ("Senior Smartwatch with HR", "electronics"),
    ("Fitness Tracker Pulse", "electronics"),
    ("Posture Corrector Back Brace", "health"),
    ("Orthopedic Insoles Memory Foam", "health"),
    ("ToeGuard Toe Spacers Hallux", "health"),
    ("Compression Sleeve Knee Support", "health"),
    ("LumaGuard LED Dog Collar", "toys-books"),
    ("Waterproof Dog Raincoat", "toys-books"),
    ("Hundegeschirr verstellbar", "toys-books"),
    ("Hands-Free Magnifying Glass with Neck Strap", "other"),
    ("Adjustable Trekking Poles for Hiking", "other"),
    ("Self Defense Walking Stick", "other"),
    ("Snap-On Cosmetic Veneers", "beauty"),
    ("Face Massager Wand", "beauty"),
])
def test_classify_general_subniche_picks_sensible_bucket(title, expected):
    assert _classify_general_subniche(title) == expected


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


def _seed_fashion(db, store, handle, title, subniche):
    db.add(Product(
        store_id=store.id, shopify_id=handle, title=title, handle=handle,
        image_url="", price="", vendor="", product_type="", product_url="",
        current_position=1, previous_position=0, label="",
        ai_tags="", is_fashion=True, subniche=subniche,
        last_scraped=datetime.utcnow(),
    ))
    db.commit()


def test_migrate_demotes_user_listed_fashion_intruders(db, store):
    """All the items the user listed in the demote spec MUST flip
    is_fashion=False with a sensibly-bucketed subniche."""
    cases = [
        ("dog-raincoat",          "Waterproof Dog Raincoat", "fashion",     "toys-books"),
        ("led-dog-collar",        "LumaGuard LED Dog Collar", "fashion",    "toys-books"),
        ("posture-corrector",     "Posture Corrector Back",   "fashion",    "health"),
        ("hip-thigh-support",     "Hip and Thigh Support Band","fashion",   "health"),
        ("toe-spacers",           "ToeGuard Toe Spacers",     "fashion",    "health"),
        ("foot-pain-pads",        "ComfortStep Foot Pain Relief Pads", "fashion", "health"),
        ("magnifying-glass",      "Hands-Free Magnifying Glass with Neck Strap", "fashion", "other"),
        ("trekking-poles",        "Adjustable Trekking Poles", "fashion",    "other"),
        ("self-defense-stick-1",  "Self Defense Walking Stick", "fashion",   "other"),
        ("smartwatch-1",          "Smartwatch",               "fashion",    "electronics"),
        ("smartwatch-senior",     "Senior Smartwatch HR Tracking", "fashion", "electronics"),
        ("snap-veneers",          "Snap-On Cosmetic Veneers", "fashion",    "beauty"),
    ]
    for handle, title, sub, _ in cases:
        _seed_fashion(db, store, handle, title, sub)

    demoted = migrate_force_general_to_general(db)
    assert demoted == len(cases)

    for handle, _, _, expected_sub in cases:
        p = db.query(Product).filter(Product.shopify_id == handle).one()
        assert p.is_fashion is False, f"{handle} should be demoted"
        assert p.subniche == expected_sub, (
            f"{handle} subniche={p.subniche!r}, expected {expected_sub!r}"
        )


def test_migrate_keeps_legitimate_fashion(db, store):
    """Costumes, shapewear, mechanical watches, decorative pet bandanas,
    bag accessories must STAY on Fashion after the migration runs."""
    keep_cases = [
        ("classic-watch",     "Classic Mechanical Watch with Leather Band", "jewelry"),
        ("decorative-bandana","Decorative Dog Bandana Floral", "accessories"),
        ("superhero-costume", "Realistic Superhero Costume for Kids & Adults", "fashion"),
        ("halloween-mask",    "Halloween Latex Mask Adult Full Face", "fashion"),
        ("dog-costume",       "Large Dog Skeleton Bodysuit Costume", "fashion"),
        ("shapewear",         "Tummy Control Shapewear Bodysuit", "fashion"),
        ("makeup-bag",        "Make-up Bag Duo Travel", "bags"),
        ("watch-case",        "Watch Holder Travel Case Leather 3 Watches", "accessories"),
        ("ortho-shoes",       "Orthopedic Slip-on Shoes Men", "fashion"),
    ]
    for handle, title, sub in keep_cases:
        _seed_fashion(db, store, handle, title, sub)

    demoted = migrate_force_general_to_general(db)
    assert demoted == 0
    for handle, _, expected_sub in keep_cases:
        p = db.query(Product).filter(Product.shopify_id == handle).one()
        assert p.is_fashion is True, f"{handle} must stay on Fashion"
        assert p.subniche == expected_sub


def test_migrate_force_general_is_idempotent(db, store):
    _seed_fashion(db, store, "x-smartwatch", "Smart Watch HR", "fashion")
    assert migrate_force_general_to_general(db) == 1
    assert migrate_force_general_to_general(db) == 0


def test_per_scrape_sweep_demotes_fashion_gadgets(db, store):
    """update_products_in_db runs an inline force-general sweep on
    every scrape so dormant rows that match the wearable-gadget regex
    get pulled back to General even on partial scrapes."""
    _seed_fashion(db, store, "stuck-smartwatch",
                  "BowLift | Senior Smartwatch", "fashion")
    update_products_in_db(db, store, fashion_products=[], general_products=[])
    p = db.query(Product).filter(Product.shopify_id == "stuck-smartwatch").one()
    assert p.is_fashion is False
    assert p.subniche == "electronics"
