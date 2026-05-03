"""Source-position regression tests.

User rule (now load-bearing across the whole pipeline): the `position`
each Spy Wizard product carries MUST equal its 1-indexed slot on the
merchant's /collections/all?sort_by=best-selling source HTML — NOT a
re-numbered fashion-only or general-only sequence.

Bug we're pinning against: previously, after junk filtering, the
remaining fashion items got renumbered 1..N. So a layout like

    [Junk@1, Cute Dress@2, Smartwatch@3, Sneakers@4]

would surface as "Cute Dress = #1" on the Fashion tab, even though
on the live competitor's HTML the dress sits at position #2.

These tests cover:
  - The user's exact example: junk + fashion + general + fashion
    must yield Fashion tab positions {2, 4} (NOT {1, 2}).
  - Two-page scrapes: source_position is a CROSS-page counter, so a
    fashion item on page 2 carries its absolute rank from page 1's
    last-product + N.
  - Junk-heavy top: [Junk@1, Junk@2, Fashion@3] → Fashion tab shows
    position 3 (not position 1).
  - General-heavy top: [General@1, Fashion@2] → Fashion tab shows
    position 2 (not position 1).
  - FORCE_GENERAL precedence still respected: a smartwatch matching
    both fashion + general regexes at source_position 5 lands on
    General with position 5 (not position 1).
  - target caps truncate to the FIRST N items by source order — they
    do NOT renumber.
"""
import pytest

from scraper import (
    _distribute_page_to_feeds,
    WEARABLE_SUBNICHES,
)


def _seed(title, *, fashion=False, subniche="", excluded=False):
    """Helper: synthesise a page_product dict mirroring what the
    scraper's _extract_products_from_html function returns, with the
    Gemini-classified is_fashion + subniche already attached."""
    return {
        "title": title, "handle": title.lower().replace(" ", "-"),
        "image_url": "", "product_type": "", "product_url": "",
        "is_fashion": fashion, "subniche": subniche,
        "_excluded": excluded,
    }


def _positions_by_title(items):
    return [(p["title"], p["position"]) for p in items]


# === The user's exact example ===
def test_user_example_junk_fashion_general_fashion():
    """User-stated bug: source list [Junk@1, Dress@2, Smartwatch@3,
    Sneakers@4] currently renders as Dress=#1 on Fashion. After the
    fix, Dress must keep position 2 (its literal source slot) and
    Sneakers must keep position 4. Smartwatch lands on General at
    position 3."""
    page = [
        _seed("Shipping Protection", excluded=True),       # source 1
        _seed("Cute Dress", fashion=True, subniche="fashion"),  # source 2
        _seed("Smartwatch", fashion=False, subniche="electronics"),  # source 3
        _seed("Sneakers", fashion=True, subniche="fashion"),  # source 4
    ]
    fashion, general = [], []
    next_pos = _distribute_page_to_feeds(page, fashion, general, 100, 100, 0)
    assert next_pos == 4
    assert _positions_by_title(fashion) == [("Cute Dress", 2), ("Sneakers", 4)]
    assert _positions_by_title(general) == [("Smartwatch", 3)]


def test_junk_at_top_does_not_renumber_fashion():
    """[Junk@1, Junk@2, Fashion@3] → Fashion tab shows position 3."""
    page = [
        _seed("Versandschutz", excluded=True),
        _seed("Geschenkkarte", excluded=True),
        _seed("Floral Maxi Dress", fashion=True, subniche="fashion"),
    ]
    fashion, general = [], []
    _distribute_page_to_feeds(page, fashion, general, 100, 100, 0)
    assert _positions_by_title(fashion) == [("Floral Maxi Dress", 3)]
    assert general == []


def test_general_at_top_does_not_renumber_fashion():
    """[General@1, Fashion@2] → Fashion tab position 2."""
    page = [
        _seed("Bauhaus Pendant Light", fashion=False, subniche="home"),
        _seed("Cotton T-Shirt", fashion=True, subniche="fashion"),
    ]
    fashion, general = [], []
    _distribute_page_to_feeds(page, fashion, general, 100, 100, 0)
    assert _positions_by_title(fashion) == [("Cotton T-Shirt", 2)]
    assert _positions_by_title(general) == [("Bauhaus Pendant Light", 1)]


# === Cross-page counter ===
def test_source_position_counter_carries_across_pages():
    """Page 1 ends at counter=24; page 2's first item is at source
    position 25. The Fashion tab must surface that absolute rank."""
    page1 = [_seed(f"Item {i}", fashion=False, subniche="home") for i in range(24)]
    page2 = [
        _seed("Page-2 Dress", fashion=True, subniche="fashion"),  # source 25
        _seed("Page-2 Lamp", fashion=False, subniche="home"),     # source 26
    ]
    fashion, general = [], []
    end_of_p1 = _distribute_page_to_feeds(page1, fashion, general, 100, 100, 0)
    assert end_of_p1 == 24
    end_of_p2 = _distribute_page_to_feeds(page2, fashion, general, 100, 100, end_of_p1)
    assert end_of_p2 == 26
    assert _positions_by_title(fashion) == [("Page-2 Dress", 25)]
    # Page-2 lamp lands at general position 26
    page2_lamp_pos = next(p["position"] for p in general if p["title"] == "Page-2 Lamp")
    assert page2_lamp_pos == 26


def test_excluded_items_consume_a_slot_in_source_position():
    """Junk we drop still increments the counter — that's how a
    Fashion item right after a block of junk shows its true rank."""
    page = [
        _seed("Shipping Protection", excluded=True),    # 1
        _seed("Versandschutz", excluded=True),          # 2
        _seed("Mystery Box", excluded=True),            # 3
        _seed("Real Dress", fashion=True, subniche="fashion"),  # 4
    ]
    fashion, general = [], []
    next_pos = _distribute_page_to_feeds(page, fashion, general, 100, 100, 0)
    assert next_pos == 4
    assert _positions_by_title(fashion) == [("Real Dress", 4)]


# === FORCE_GENERAL precedence respected with source positions ===
def test_smartwatch_lands_on_general_with_correct_source_position():
    """A smartwatch at source position 5 should land on General with
    position=5, NOT promoted to Fashion (FORCE_GENERAL beats
    FORCE_FASHION) and NOT renumbered."""
    page = [
        _seed("Floral Dress", fashion=True, subniche="fashion"),       # 1
        _seed("Cotton Hoodie", fashion=True, subniche="fashion"),      # 2
        _seed("Bauhaus Lamp", fashion=False, subniche="home"),         # 3
        _seed("Leather Wallet", fashion=True, subniche="fashion"),     # 4
        _seed("BowLift Senior Smartwatch", fashion=True, subniche="fashion"),  # 5
    ]
    fashion, general = [], []
    _distribute_page_to_feeds(page, fashion, general, 100, 100, 0)
    # Smartwatch demoted to General with position 5 preserved
    smartwatch = next(p for p in general if "Smartwatch" in p["title"])
    assert smartwatch["position"] == 5
    # Fashion list keeps positions 1, 2, 4 (with the gap at 3 = lamp,
    # 5 = smartwatch demoted)
    assert _positions_by_title(fashion) == [
        ("Floral Dress", 1),
        ("Cotton Hoodie", 2),
        ("Leather Wallet", 4),
    ]
    # General list keeps positions 3 + 5
    assert _positions_by_title(general) == [
        ("Bauhaus Lamp", 3),
        ("BowLift Senior Smartwatch", 5),
    ]


# === Target cap truncation ===
def test_fashion_target_cap_keeps_first_N_by_source_order():
    """When the fashion cap is 3, the first 3 fashion items by source
    order are kept. Their positions stay literal — not 1/2/3 if the
    source positions are 2/3/5."""
    page = [
        _seed("Junk", excluded=True),                                  # 1
        _seed("Dress A", fashion=True, subniche="fashion"),            # 2
        _seed("Dress B", fashion=True, subniche="fashion"),            # 3
        _seed("Lamp", fashion=False, subniche="home"),                 # 4
        _seed("Dress C", fashion=True, subniche="fashion"),            # 5
        _seed("Dress D", fashion=True, subniche="fashion"),            # 6 — over cap
    ]
    fashion, general = [], []
    _distribute_page_to_feeds(page, fashion, general, 3, 100, 0)
    assert len(fashion) == 3
    assert _positions_by_title(fashion) == [
        ("Dress A", 2), ("Dress B", 3), ("Dress C", 5),
    ]
    # Dress D is dropped, doesn't get a position-4 squash.


def test_general_target_cap_keeps_first_N_by_source_order():
    page = [
        _seed("Lamp A", fashion=False, subniche="home"),    # 1
        _seed("Dress A", fashion=True, subniche="fashion"), # 2
        _seed("Lamp B", fashion=False, subniche="home"),    # 3
        _seed("Lamp C", fashion=False, subniche="home"),    # 4 — over cap
    ]
    fashion, general = [], []
    _distribute_page_to_feeds(page, fashion, general, 100, 2, 0)
    assert _positions_by_title(general) == [("Lamp A", 1), ("Lamp B", 3)]
    # Lamp C dropped, NOT renumbered into position 3.


# === Wearable subniche reconciliation preserves source positions ===
def test_wearable_subniche_preserves_source_position():
    """Gemini sometimes returns subniche='jewelry' but is_fashion=
    false. The reconciliation flips is_fashion to True, but the
    position MUST still be the literal source position, not a
    re-numbered fashion-only sequence."""
    page = [
        _seed("Lamp", fashion=False, subniche="home"),  # 1
        _seed("Pearl Earrings", fashion=False, subniche="jewelry"),  # 2 — flips to fashion
        _seed("Lamp 2", fashion=False, subniche="home"),  # 3
        _seed("Dress", fashion=True, subniche="fashion"),  # 4
    ]
    fashion, general = [], []
    _distribute_page_to_feeds(page, fashion, general, 100, 100, 0)
    earring = next(p for p in fashion if "Earrings" in p["title"])
    assert earring["position"] == 2
    assert earring["subniche"] == "jewelry"  # subniche preserved
    assert _positions_by_title(fashion) == [("Pearl Earrings", 2), ("Dress", 4)]


def test_source_position_field_attached_even_to_excluded_items():
    """Junk we drop should still get a source_position stamp so
    downstream debugging / migrations can reconstruct the source list."""
    page = [
        _seed("Junk", excluded=True),
        _seed("Real Item", fashion=True, subniche="fashion"),
    ]
    fashion, general = [], []
    _distribute_page_to_feeds(page, fashion, general, 100, 100, 0)
    assert page[0]["source_position"] == 1
    assert page[1]["source_position"] == 2
    assert page[1]["position"] == 2  # mirrors source_position for kept items


def test_subniche_exclude_consumes_slot_too():
    """Items Gemini retroactively flagged as subniche='exclude' are
    skipped from both feeds (treated as junk) but their source_position
    slot is consumed so subsequent items keep their true position."""
    page = [
        _seed("Real Dress", fashion=True, subniche="fashion"),  # 1
        _seed("Disguised Gift Card", fashion=False, subniche="exclude"),  # 2
        _seed("Real Sneakers", fashion=True, subniche="fashion"),  # 3
    ]
    fashion, general = [], []
    _distribute_page_to_feeds(page, fashion, general, 100, 100, 0)
    assert _positions_by_title(fashion) == [
        ("Real Dress", 1), ("Real Sneakers", 3),
    ]
    assert general == []


# === User's exact wording: positions have GAPS, that's correct ===
def test_fashion_positions_have_gaps_when_general_items_interleave():
    """Per the user spec: the displayed list will have gaps. You might
    see #1, #2, #5, #7, #8 in the Fashion tab because items at
    positions 3, 4, 6 were General or excluded."""
    page = [
        _seed("F1", fashion=True, subniche="fashion"),       # 1
        _seed("F2", fashion=True, subniche="fashion"),       # 2
        _seed("Lamp1", fashion=False, subniche="home"),      # 3
        _seed("Junk1", excluded=True),                       # 4
        _seed("F5", fashion=True, subniche="fashion"),       # 5
        _seed("Lamp2", fashion=False, subniche="home"),      # 6
        _seed("F7", fashion=True, subniche="fashion"),       # 7
        _seed("F8", fashion=True, subniche="fashion"),       # 8
    ]
    fashion, general = [], []
    _distribute_page_to_feeds(page, fashion, general, 100, 100, 0)
    fashion_positions = [p["position"] for p in fashion]
    assert fashion_positions == [1, 2, 5, 7, 8]
    general_positions = [p["position"] for p in general]
    assert general_positions == [3, 6]
