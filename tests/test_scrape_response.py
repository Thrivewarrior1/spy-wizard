"""Per-store warning + response-shape regression tests.

User spec: the Scrape-All popup must report BOTH fashion AND general
counts per store. When a mixed-niche store ends up with one empty
feed (likely mis-routing or sparse catalog), the popup must surface a
warning row so the user catches it without manual inspection. Pure-
niche stores (Fashion-only / General-only) legitimately have one
empty feed and must NOT trigger the warning.

These tests pin the warning-decision helper centrally so the
decision logic can't drift between scrape_all_stores and the
/api/stats unhealthy_stores check.
"""
import pytest

from scraper import _per_store_warning


# --- Mixed-niche stores (warning expected when one feed is empty) ---
@pytest.mark.parametrize("niche", [
    "Fashion & General",
    "Fashion & HD",
    "Fashion & General Mix",
    "MultiMarket. Mostly USA",
    "Mixed-niche store",
    "fashion & general",  # case-insensitive
])
def test_mixed_niche_warns_when_general_empty(niche):
    msg = _per_store_warning(niche, fashion_n=187, general_n=0)
    assert msg is not None
    assert "general" in msg.lower()


@pytest.mark.parametrize("niche", [
    "Fashion & General",
    "Fashion & HD",
    "MultiMarket",
])
def test_mixed_niche_warns_when_fashion_empty(niche):
    msg = _per_store_warning(niche, fashion_n=0, general_n=145)
    assert msg is not None
    assert "fashion" in msg.lower()


def test_mixed_niche_no_warning_when_both_populated():
    assert _per_store_warning("Fashion & General", 200, 200) is None
    assert _per_store_warning("MultiMarket", 50, 30) is None


# --- Pure-niche stores (one empty feed is the EXPECTED state) ---
@pytest.mark.parametrize("niche,f,g", [
    ("Fashion", 200, 0),     # pure fashion store; 0 general expected
    ("General", 0, 200),     # pure general store; 0 fashion expected
    ("Fashion", 197, 1),     # near-empty general legitimate too
    ("General", 12, 188),
])
def test_pure_niche_does_NOT_warn_on_single_empty_feed(niche, f, g):
    assert _per_store_warning(niche, f, g) is None


# --- Dead scrape: both feeds empty, ALWAYS warns regardless of niche ---
@pytest.mark.parametrize("niche", [
    "Fashion", "General", "Fashion & General", "MultiMarket", "",
])
def test_dead_scrape_warns_for_any_niche(niche):
    msg = _per_store_warning(niche, fashion_n=0, general_n=0)
    assert msg is not None
    assert "dead" in msg.lower() or "empty" in msg.lower()


def test_empty_niche_string_treats_store_as_pure_niche():
    """A store created without a niche string should fall back to the
    pure-niche behavior (no warning when only one feed populates) so
    we don't false-alarm during seed migrations or before the user
    has set the niche field."""
    assert _per_store_warning("", 200, 0) is None
    assert _per_store_warning(None, 0, 200) is None
