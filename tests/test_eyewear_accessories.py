"""Eyewear-accessory regression tests — utility / consumables for
glasses must land on General, not on the Fashion eyewear bucket.

The user added Wild Eye Vision (an outdoor/sunglasses store) and
worried that the FORCE_FASHION eyewear pattern would sweep lens
cleaners, microfiber cloths, contact-lens solution, eye drops,
glasses repair kits and similar utility items into Fashion. The fix:
add explicit FORCE_GENERAL tokens for these accessory categories so
they win precedence over the bare 'glasses' / 'sunglasses' / 'eyewear'
patterns.

Hose-hardware regression: garden hoses kept getting promoted to Fashion
because the German FORCE_FASHION 'Hose' (= pants) pattern matched the
English noun. FORCE_GENERAL now has explicit 'garden hose' / 'fire
hose' / 'expandable hose' patterns that win precedence.
"""
import pytest

from scraper import _is_forced_fashion, _is_forced_general


# === Eyewear accessories — must demote to General ===
_EYEWEAR_ACCESSORY_TITLES = [
    # English
    "Lens Cleaner Spray for Glasses",
    "Microfiber Lens Cloth Pack of 5",
    "Microfibre Lens Cloth",
    "Lens Cleaning Wipes Disposable",
    "Lens Case for Contact Lenses",
    "Contact Lens Solution 360ml",
    "Contact Lens Cleaner Travel Set",
    "Eye Drops Lubricating Sterile",
    "Glasses Repair Kit with Screwdriver",
    "Eyeglass Repair Kit Mini Tools",
    "Sunglasses Case Hard Shell",
    "Glasses Pouch Velvet Soft",
    "Eyeglass Holder Stand Wood",
    "Anti Fog Spray for Glasses 30ml",
    "Lens Fog Spray Travel Size",
    "Optical Cleaner Spray Anti-Static",
    "Eyeglass Screwdriver Kit Precision",
    # German
    "Brillenreiniger Spray 100ml",
    "Brillentuch Mikrofaser 5er Pack",
    "Brillenetui Hartschale",
    "Augentropfen Befeuchtend",
]


@pytest.mark.parametrize("title", _EYEWEAR_ACCESSORY_TITLES)
def test_eyewear_accessory_is_forced_general(title):
    assert _is_forced_general(title=title), (
        f"_is_forced_general missed eyewear accessory {title!r}. The "
        "FORCE_FASHION eyewear pattern would otherwise misclassify lens "
        "cleaners / cloths / cases / drops / repair kits as fashion."
    )


# === Glasses chain / cord — borderline. The user said 'glasses chain'
# === could go either way; we route to General because it's a utility
# === strap, not jewelry. Pin the call so future bug reports have a
# === clear baseline.
def test_glasses_chain_and_cord_are_general():
    assert _is_forced_general(title="Glasses Chain Beaded Eyewear Holder")
    assert _is_forced_general(title="Glasses Cord Adjustable Sport")
    assert _is_forced_general(title="Eyeglasses Strap Neck Holder")


# === Real eyewear (style) MUST still match FORCE_FASHION ===
@pytest.mark.parametrize("title", [
    "Polarized Sunglasses Aviator",
    "Reading Glasses Tortoiseshell Frames",
    "Progressive Lenses Designer",
    "Designer Eyewear Round Frames",
    "Sonnenbrille Damen Polarisiert",
    "Lesebrille Schwarz Vintage",
])
def test_real_eyewear_still_forced_fashion(title):
    assert _is_forced_fashion(title=title)
    # And NOT demoted by the accessory regex.
    assert not _is_forced_general(title=title), (
        f"Real eyewear {title!r} got pulled into General — the eyewear-"
        "accessory pattern is too broad."
    )


# === Hose hardware — was getting promoted to Fashion via German Hose ===
@pytest.mark.parametrize("title", [
    "30m Expandable Garden Hose Anti-Kink",
    "Garden Hose 50ft Heavy Duty",
    "Fire Hose Replacement 100ft",
    "JetCrafter High-Pressure Garden Hose Attachment for Cleaning",
    "Hose Reel Wall Mount Auto-Rewind",
    "Hose Nozzle Brass 8 Pattern Sprayer",
    "Hose Spray Gun Garden Watering",
    "Hose Connector Quick Release",
    "Water Hose Drinking-Safe RV",
    "Spray Hose Kitchen Tap Replacement",
    "Hochdruck Gartenschlauch 25m",
])
def test_hose_hardware_is_forced_general(title):
    assert _is_forced_general(title=title), (
        f"_is_forced_general missed hose-hardware item {title!r}. The "
        "German FORCE_FASHION 'Hose' (pants) pattern would promote it to "
        "Fashion — FORCE_GENERAL must catch it first."
    )


# === Real German pants (Hose) must still match FORCE_FASHION ===
@pytest.mark.parametrize("title", [
    "Schwarze Hose Damen Slim Fit",
    "Jogginghose Herren Sport",
    "Lederhose Bayern Tracht",
])
def test_real_german_pants_still_forced_fashion(title):
    assert _is_forced_fashion(title=title)
    # And NOT swept into General by the hose-hardware regex.
    assert not _is_forced_general(title=title)
