"""Regression tests for the hard-coded non-product exclusion regex.

These items must NEVER reach the Fashion feed OR the General feed.
Multi-language because our 12 stores ship across DE/FR/ES/IT/NL.
"""
import pytest

from scraper import NON_PRODUCT_TITLE_RE, NON_PRODUCT_TYPE_RE


@pytest.mark.parametrize("title", [
    # === English ===
    "Shipping Protection",
    "Shipping Insurance",
    "Package Protection",
    "Route Protection",
    "Route Insurance",
    "Order Protection",
    "Delivery Insurance",
    "Delivery Protection",
    "100% Coverage",
    "Coverage Plan",
    "Protection Plan",
    "Extended Warranty",
    "Warranty Plan",
    "Service Plan",
    "Product Warranty",
    "Gift Card $50",
    "E-Gift Card",
    "Egift",
    "Gift Voucher",
    "Store Credit",
    "Carbon Offset",
    "Plant a Tree",
    "Slidecart",
    # === German ===
    "Versicherter Versand",
    "Versandschutz",
    "Versandversicherung",
    "Paketversicherung",
    "Geschenkkarte",
    "Geschenkgutschein",
    "Erweiterte Garantie",
    "Verlängerte Garantie",
    "Verlangerte Garantie",  # ASCII fallback
    "Garantieverlängerung",
    # === French ===
    "Carte Cadeau 50€",
    "Chèque Cadeau",
    "Cheque Cadeau",
    "Assurance Livraison",
    "Assurance Expédition",
    "Assurance Expedition",
    "Protection Expédition",
    "Garantie Étendue",
    "Garantie Etendue",
    "Garantie Prolongée",
    # === Spanish ===
    "Tarjeta Regalo",
    "Cheque Regalo",
    "Protección de Envío",
    "Proteccion de Envio",
    "Seguro de Envío",
    "Garantía Extendida",
    "Garantia Ampliada",
    # === Italian ===
    "Carta Regalo",
    "Buono Regalo",
    "Protezione Spedizione",
    "Assicurazione Spedizione",
    "Garanzia Estesa",
    # === Dutch ===
    "Verzendverzekering",
    "Bezorgverzekering",
    "Cadeaubon",
    "Cadeaukaart",
    "Uitgebreide Garantie",
])
def test_non_products_are_blocked(title):
    assert NON_PRODUCT_TITLE_RE.search(title), (
        f"Expected NON_PRODUCT_TITLE_RE to match {title!r} so it is dropped "
        "before storage. If this fails the title will leak into Fashion or "
        "General again."
    )


@pytest.mark.parametrize("title", [
    # Real fashion
    "Cotton T-Shirt",
    "Floral Summer Dress",
    "Bag with Strap",
    "Leather Wallet",
    "White Sneakers",
    "Wool Cardigan",
    "Sommerkleid",
    "Pullover Damen",
    "Robe d'été",
    "Vestido de Verano",
    # Real general (non-fashion but legitimate)
    "Lampe de Table",
    "Floor Lamp",
    "Phone Case",
    "Bluetooth Speaker",
    "Necklace",
    # Tricky strings that contain a service-like substring but are NOT add-ons.
    # The regex must be specific enough not to flag these.
    "Garantie de Qualité Premium Cotton",
    "Tip-Top Fedora Hat",
    "Tax Bracelet Silver",  # contains "tax" — regex should not match plain "tax"
    "Sample Sale Dress",   # contains "sample"
    "Insured Delivery Bag",  # contains "insured" but no shipping/route/etc prefix
    "Garantía Original",   # plain "garantia" is too generic to flag
    "Warranty Card Holder",  # contains "warranty" but as part of a real product name
])
def test_real_products_are_not_blocked(title):
    assert not NON_PRODUCT_TITLE_RE.search(title), (
        f"NON_PRODUCT_TITLE_RE incorrectly matched {title!r}. The pattern "
        "is too broad and would drop legitimate products."
    )


@pytest.mark.parametrize("ptype,should_block", [
    # Shopify product types that clearly indicate a checkout add-on
    ("Slidecart - Shipping Protection", True),
    ("Gift Card", True),
    ("Versandschutz", True),
    ("Carte Cadeau", True),
    ("Extended Warranty", True),
    # Real product types
    ("Women Blouse Seasonal", False),
    ("Men Winter Coats & Jackets", False),
    ("Women Sandals Evergreen", False),
    ("Home Decor Lamps", False),
    ("Electronics", False),
])
def test_type_exclusion(ptype, should_block):
    actual = bool(NON_PRODUCT_TYPE_RE.search(ptype))
    assert actual == should_block, (
        f"NON_PRODUCT_TYPE_RE on {ptype!r}: expected {should_block}, got {actual}"
    )
