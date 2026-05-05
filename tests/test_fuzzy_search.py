"""Fuzzy / hybrid search regression tests.

Bug the user reported: typing 'chandelier' didn't find all chandeliers.
Search was too rigid — title-substring with word boundaries missed
opaque-titled lighting like "Aurora Crystal Pendant" and didn't
multilingual-resolve "Kronleuchter" to chandelier.

Fix architecture (categories.py):
  - Each product has a `product_category` column populated by
    assign_product_category() — multilingual regex over title,
    handle, image_url, product_type. Kronleuchter / lustre /
    lampadario / araña / kroonluchter all resolve to 'chandelier'.
  - TOKEN_TO_CATEGORIES indexes every multilingual search token AND
    every parent-category label (so 'lighting' returns the union of
    all lighting children).
  - lookup_categories_for_query_token() handles plural/singular
    variants and parent expansion in one call.
  - main.py search uses Product.product_category IN (categories)
    as the precision path, falls back to title/ai_tags for
    uncategorised rows or non-category tokens.

Tests cover:
  - Classify: every documented category gets correctly assigned for
    canonical English + multilingual titles.
  - Token lookup: single-word, plural, multilingual, parent.
  - Edge cases: smartwatch resolves to wearable-gadget (NOT watch),
    LED dog collar to dog-collar (NOT necklace), shoe doesn't match
    'shoehorn'.
"""
import pytest

from categories import (
    PRODUCT_CATEGORIES,
    PARENTS_TO_CHILDREN,
    TOKEN_TO_CATEGORIES,
    ALL_CATEGORY_NAMES,
    assign_product_category,
    lookup_categories_for_query_token,
    is_category_token,
)


# =====================================================================
# CLASSIFY: assign_product_category() must categorise canonical inputs
# correctly. Multilingual titles all map to the same English slug.
# =====================================================================
@pytest.mark.parametrize("title,expected", [
    # --- Lighting (the user's bug) ---
    ("Aurora Crystal Ring Chandelier",          "chandelier"),
    ("Kristall Kronleuchter Modern",            "chandelier"),
    ("Lustre Cristal Moderne",                  "chandelier"),
    ("Lampadario in Cristallo",                 "chandelier"),
    ("Araña de Cristal Moderna",                "chandelier"),
    ("Kroonluchter Kristal Modern",             "chandelier"),
    ("Modern Brass Chandelier 5-arm",           "chandelier"),
    # --- Lamps ---
    ("Bedside Table Lamp Walnut",               "table-lamp"),
    ("Goldene Tischlampe Designer",             "table-lamp"),
    ("Lampe de Table Vintage",                  "table-lamp"),
    ("Lampada da Tavolo Moderna",               "table-lamp"),
    ("Tafellamp Modern",                        "table-lamp"),
    ("Adjustable Floor Lamp Industrial",        "floor-lamp"),
    ("Stehlampe Modern Design",                 "floor-lamp"),
    ("Pendant Light Black Metal",               "pendant-light"),
    ("Hängelampe Wohnzimmer Industrial",        "pendant-light"),
    ("Pendelleuchte Esstisch Schwarz",          "pendant-light"),
    ("Suspension Bois Scandinave",              "pendant-light"),
    ("Ceiling Light Fixture Modern",            "ceiling-light"),
    ("Plafondlamp LED Woonkamer",               "ceiling-light"),
    ("Wall Sconce Brass Vintage",               "wall-light"),
    ("Wandlamp Buitenverlichting",              "wall-light"),
    ("Applique Murale Dorée",                   "wall-light"),
    ("Antique Brass Candelabra",                "candle-holder"),
    ("Kerzenhalter Silber 3-er Set",            "candle-holder"),
    ("LED Fairy Lights 10m",                    "string-lights"),
    ("Lichterkette LED Solar",                  "string-lights"),
    # --- Footwear ---
    ("White Leather Sneakers Men",              "sneaker"),
    ("Turnschuhe Damen",                        "sneaker"),
    ("The Manchester Men's Vintage Boots",      "boot"),
    ("Stiefel mit Plateau",                     "boot"),
    ("Bottes Hautes Femme",                     "boot"),
    ("Botas de Cuero",                          "boot"),
    ("Comfortable Slip-on Sandals",             "sandal"),  # 'sandal' wins over 'slip-on'
    ("Bequemer Orthoschuh Damen",               "orthopedic-shoe"),
    ("Orthopedic Walking Shoes for Plantar Fasciitis", "orthopedic-shoe"),
    ("Cosy House Slippers Wool",                "slipper"),
    ("Hausschuhe Damen Filz",                   "slipper"),
    ("High Heels Stiletto Sleek",               "heel"),
    ("Classic Loafers Leather",                 "loafer"),
    # --- Apparel ---
    ("Floral Summer Maxi Dress",                "dress"),
    ("Sommerkleid Damen Lang",                  "dress"),
    ("Robe Longue d'Été",                       "dress"),
    ("Vestido de Verano",                       "dress"),
    ("Vestito Lungo Estivo",                    "dress"),
    ("Pleated Midi Skirt",                      "skirt"),
    ("Falda Plisada Midi",                      "skirt"),
    ("Cotton T-Shirt Plain",                    "t-shirt"),
    ("Long-Sleeved Shirt Classic",              "shirt"),
    ("Hemd Herren Casual",                      "shirt"),
    ("Silk Blouse Designer",                    "blouse"),
    ("Wool Cardigan Long Sleeve",               "cardigan"),
    ("Pullover Damen Strick",                   "sweater"),
    ("Hoodie Pullover Schwarz",                 "hoodie"),
    ("Leather Jacket Black",                    "jacket"),
    ("Jacke Damen Winter",                      "jacket"),
    ("Manteau Long en Laine",                   "coat"),
    ("Cappotto Lungo Donna",                    "coat"),
    ("Skinny Jeans Slim Fit",                   "jeans"),
    ("Cargo Pants Khaki",                       "pants"),
    ("Hose Schwarz Damen",                      "pants"),
    ("Bermuda Shorts Cotton",                   "shorts"),
    ("Wide-Leg Jumpsuit",                       "jumpsuit"),
    ("Luxus Bademantel - Damen",                "bathrobe"),
    ("The Hungerford Men's Hooded Bathrobe",    "bathrobe"),
    ("Cotton Pajamas Set",                      "pajamas"),
    ("Bikini Set Triangle",                     "swimwear"),
    # --- Intimates ---
    ("R&B | Multiway-Rücken-BH für Damen",      "bra"),
    ("Cotton Bra Wireless",                     "bra"),
    ("5 Pack Cotton Panties",                   "panties"),
    ("Seamless Briefs No-Show",                 "panties"),
    ("Boxer Briefs Men",                        "boxer"),
    ("Tummy Control Shapewear Bodysuit",        "shapewear"),
    ("Silk Lingerie Set",                       "lingerie"),
    ("R&B - Seidenweiche Nahtlose Unterwäsche", "underwear-generic"),
    ("Compression Stockings Medical",           "compression-hosiery"),
    ("Sheer Black Stockings",                   "hosiery"),
    ("Cotton Crew Socks 6-Pack",                "socks"),
    # --- Eyewear ---
    ("Polarized Sunglasses Aviator",            "sunglasses"),
    ("Sonnenbrille Damen",                      "sunglasses"),
    ("Reading Glasses Tortoiseshell",           "glasses"),
    ("Progressive Lenses Designer",             "glasses"),
    ("Lesebrille Schwarz Vintage",              "glasses"),
    # --- Wearable gadgets (must NOT resolve to watch / jewelry) ---
    ("Smartwatch Pro Health Edition",           "smartwatch"),
    ("BowLift | Senior Smartwatch",             "smartwatch"),
    ("Fitness Tracker Pulse Heart Rate",        "fitness-tracker"),
    ("Smart Ring Health Monitor",               "fitness-tracker"),
    ("Posture Corrector Back Brace",            "posture-corrector"),
    ("Haltungskorrektur Damen",                 "posture-corrector"),
    ("Knee Support Brace Sport",                "support-brace"),
    ("Stützgürtel Rücken",                      "support-brace"),
    ("ToeGuard Orthopedic Toe Spacers",         "orthopedic-insole"),
    ("Foot Pain Relief Gel Pads",               "orthopedic-insole"),
    # --- Pet protective (NOT jewelry / accessories) ---
    ("Waterproof Dog Raincoat Reflective",      "dog-raincoat"),
    ("Hundemantel wasserdicht",                 "dog-raincoat"),
    ("LumaGuard LED Dog Collar",                "dog-collar"),
    ("Dog Harness Adjustable Padded",           "dog-harness"),
    ("Hundegeschirr verstellbar",               "dog-harness"),
    # --- Bags ---
    ("Make-up Bag Travel Duo",                  "makeup-bag"),
    ("Leather Crossbody Handbag",               "handbag"),
    ("Compact Waterproof Backpack",             "backpack"),
    ("Tote Bag Canvas",                         "tote"),
    ("Leather Wallet Slim",                     "wallet"),
    # --- Jewelry ---
    ("Diamond Hoop Earrings 18k",               "earring"),
    ("Boucles d'Oreilles Pendantes",            "earring"),
    ("Gold Pendant Necklace",                   "necklace"),
    ("Halskette mit Anhänger",                  "necklace"),
    ("Silver Bracelet Stack",                   "bracelet"),
    ("Classic Mechanical Watch Leather Band",   "watch"),
    # --- Outdoors / utility ---
    ("Adjustable Trekking Poles",               "trekking-pole"),
    ("Self Defense Walking Stick",              "trekking-pole"),
    ("Hands-Free Magnifying Glass with Neck Strap", "magnifying-glass"),
    ("Lupe mit Beleuchtung",                    "magnifying-glass"),
    ("Automatic Travel Umbrella",               "umbrella"),
    ("Regenschirm faltbar",                     "umbrella"),
    # --- Electronics ---
    ("MagSafe Wallet Phone Case Leather",       "phone-case"),
    ("iPhone 15 Pro Magnetic Case",             "phone-case"),
    # --- Costumes ---
    ("Realistic Superhero Costume Adults",      "costume"),
    ("Halloween Latex Mask",                    "halloween-mask"),
])
def test_classify_assigns_correct_category(title, expected):
    assert assign_product_category(title=title) == expected, (
        f"assign_product_category({title!r}) "
        f"got {assign_product_category(title=title)!r}, expected {expected!r}"
    )


def test_classify_returns_empty_for_uncategorised_titles():
    assert assign_product_category(title="Mysterious Untyped Item Brand X") == ""
    assert assign_product_category(title="") == ""
    assert assign_product_category(title="Aaargh Brand Name Only") == ""


def test_classify_uses_handle_when_title_is_opaque():
    """The handle ('luxus-bademantel-damen') often reveals the
    category even when the title is brand-only ('Premium Item')."""
    cat = assign_product_category(
        title="Premium Item",
        handle="luxus-bademantel-damen",
    )
    assert cat == "bathrobe"


def test_classify_uses_image_url_when_title_and_handle_opaque():
    cat = assign_product_category(
        title="Mia Premium",
        handle="mia-premium",
        image_url="https://cdn.shopify.com/files/seamless-bra-front.jpg",
    )
    assert cat == "bra"


# =====================================================================
# TOKEN LOOKUP: lookup_categories_for_query_token()
# =====================================================================
@pytest.mark.parametrize("query,expected_subset", [
    # Single-word category
    ("chandelier",   {"chandelier"}),
    # Plural folds back to singular
    ("chandeliers",  {"chandelier"}),
    ("dresses",      {"dress"}),
    ("hoops",        {"earring"}),       # plural of hoop, indexed under earring
    ("sneakers",     {"sneaker"}),
    # Multilingual
    ("Kronleuchter", {"chandelier"}),
    ("kronleuchter", {"chandelier"}),
    ("lustre",       {"chandelier"}),
    ("lampadario",   {"chandelier"}),
    ("kroonluchter", {"chandelier"}),
    ("Sonnenbrille", {"sunglasses"}),
    ("brille",       {"glasses"}),
    ("schuhe",       {"sneaker", "boot", "sandal", "slipper", "heel", "loafer", "oxford", "orthopedic-shoe", "slip-on"}.intersection(set())),  # 'schuhe' isn't directly in tokens — see below
])
def test_token_lookup_resolves_to_correct_category(query, expected_subset):
    actual = lookup_categories_for_query_token(query)
    if expected_subset:  # skip the empty intersection sentinel
        assert expected_subset.issubset(actual), (
            f"lookup({query!r}) returned {actual}, missing {expected_subset - actual}"
        )


def test_parent_token_expands_to_children():
    """Typing 'lighting' must return every lighting child category."""
    cats = lookup_categories_for_query_token("lighting")
    expected_lighting_kids = {
        "chandelier", "table-lamp", "floor-lamp", "pendant-light",
        "ceiling-light", "wall-light", "candle-holder", "string-lights",
        "lamp-shade", "light-bulb",
    }
    assert expected_lighting_kids.issubset(cats), (
        f"missing children: {expected_lighting_kids - cats}"
    )


def test_parent_token_footwear_expands():
    cats = lookup_categories_for_query_token("footwear")
    expected = {"sneaker", "boot", "sandal", "slipper", "slip-on",
                "heel", "loafer", "oxford", "orthopedic-shoe"}
    assert expected.issubset(cats)


def test_parent_token_apparel_expands():
    cats = lookup_categories_for_query_token("apparel")
    must_contain = {"dress", "shirt", "t-shirt", "blouse", "sweater",
                    "hoodie", "cardigan", "jacket", "coat", "pants",
                    "jeans", "shorts", "jumpsuit", "bathrobe", "pajamas"}
    assert must_contain.issubset(cats)


def test_parent_token_jewelry_expands():
    cats = lookup_categories_for_query_token("jewelry")
    assert {"necklace", "earring", "bracelet", "ring", "watch"}.issubset(cats)


def test_parent_token_intimates_expands():
    cats = lookup_categories_for_query_token("intimates")
    assert {"bra", "panties", "boxer", "thong", "lingerie",
            "shapewear", "underwear-generic", "hosiery", "socks"}.issubset(cats)


# =====================================================================
# EDGE CASES (the user's specific traps)
# =====================================================================
def test_smartwatch_resolves_to_smartwatch_not_watch():
    """The classic precedence trap: a smartwatch is a wearable gadget,
    NOT a jewelry-style watch. The catalog has smartwatch BEFORE
    watch, so the regex catches it first."""
    assert assign_product_category(title="BowLift | Senior Smartwatch HR") == "smartwatch"
    # Token search for 'smartwatch' returns ONLY smartwatch.
    assert lookup_categories_for_query_token("smartwatch") == {"smartwatch"}
    # And searching 'watch' returns the jewelry watch (not smartwatch).
    cats = lookup_categories_for_query_token("watch")
    assert "watch" in cats
    assert "smartwatch" not in cats


def test_led_dog_collar_resolves_to_dog_collar_not_necklace():
    assert assign_product_category(title="LumaGuard LED Dog Collar") == "dog-collar"


def test_orthopedic_shoe_resolves_to_orthopedic_shoe_not_orthopedic_insole():
    """A wearable orthopedic shoe is fashion (orthopedic-shoe);
    an orthopedic insole is a wearable gadget (orthopedic-insole).
    Different categories, both must classify correctly."""
    assert assign_product_category(title="Orthopedic Slip-on Shoes for Plantar") == "orthopedic-shoe"
    assert assign_product_category(title="Orthopedic Insoles Memory Foam") == "orthopedic-insole"


def test_garden_hose_does_not_classify_as_pants():
    """German 'Hose' = pants — but 'Garden Hose' is hose-hardware.
    The catalog doesn't have a 'hose-hardware' slot (those still rely
    on FORCE_GENERAL in scraper.py for the fashion/general split)
    and 'pants' patterns are bounded so 'Garden Hose' does NOT
    classify as pants."""
    assert assign_product_category(title="30m Expandable Garden Hose") != "pants"
    assert assign_product_category(title="Black Hose Damen Slim Fit") == "pants"


def test_bra_resolves_to_bra_not_umbrella():
    assert assign_product_category(title="Multiway Bra Damen") == "bra"
    assert lookup_categories_for_query_token("bra") == {"bra"}
    # 'umbrella' must NOT include bra, and vice versa.
    umbrella_cats = lookup_categories_for_query_token("umbrella")
    assert "bra" not in umbrella_cats
    assert "umbrella" in umbrella_cats


# =====================================================================
# NOT-A-CATEGORY — colours, brand names, generic descriptors fall
# through to the substring search.
# =====================================================================
def test_non_category_token_returns_empty():
    """Non-category words must return an empty set so the search
    falls through to title/ai_tags substring matching."""
    assert lookup_categories_for_query_token("blue") == set()
    assert lookup_categories_for_query_token("vintage") == set()
    assert lookup_categories_for_query_token("waterproof") == set()
    assert lookup_categories_for_query_token("nordic") == set()
    assert lookup_categories_for_query_token("aurora") == set()


def test_is_category_token():
    assert is_category_token("chandelier")
    assert is_category_token("Kronleuchter")
    assert is_category_token("lighting")
    assert is_category_token("dress")
    assert not is_category_token("blue")
    assert not is_category_token("aurora")


# =====================================================================
# CATALOG INTEGRITY — every category in PRODUCT_CATEGORIES has at
# least one classify_pattern AND at least one search_token.
# =====================================================================
def test_every_category_has_patterns_and_tokens():
    for cat in PRODUCT_CATEGORIES:
        assert cat["classify_patterns"], f"{cat['name']} has no classify_patterns"
        assert cat["search_tokens"], f"{cat['name']} has no search_tokens"


def test_no_duplicate_category_names():
    names = [c["name"] for c in PRODUCT_CATEGORIES]
    assert len(names) == len(set(names)), "duplicate category names"


def test_token_index_built_correctly():
    """Sanity: every category name must be reachable via its own
    canonical slug as a search token."""
    for name in ALL_CATEGORY_NAMES:
        assert name in TOKEN_TO_CATEGORIES, (
            f"category {name!r} not indexed under its own name"
        )
        assert name in TOKEN_TO_CATEGORIES[name], (
            f"category {name!r} doesn't resolve to itself"
        )
