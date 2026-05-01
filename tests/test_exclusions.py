"""Regression tests for the hard-coded non-product exclusion regex.

These items must NEVER reach the Fashion feed OR the General feed.
Multi-language because our 12 stores ship across DE/FR/ES/IT/NL.
"""
import pytest

from scraper import (
    NON_PRODUCT_TITLE_RE,
    NON_PRODUCT_TYPE_RE,
    NON_PRODUCT_URL_RE,
    _is_non_product,
)


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


@pytest.mark.parametrize("url", [
    # Real handles spotted leaking into General → "services" on prod
    "shipping-protection",
    "100-coverage",
    "https://gentsofbritain.com/products/100-coverage",
    "https://novigood.com/products/shipping-protection",
    "https://anciennemonde.fr/products/shipping-protection",
    "https://gentsofbritain.com/cdn/shop/files/shipping-protection.png?v=1749453624",
    "https://bondimode.com/cdn/shop/files/ymq-cart-shipping-protection.png",
    # Other disguise patterns
    "/products/route-protection",
    "/products/package-protection",
    "/products/gift-card-50",
    "/products/e-gift-card",
    "/products/extended-warranty",
    "/products/coverage-plan",
    "/products/protection-plan",
    "/products/versicherter-versand",
    "/products/versandschutz",
    "/products/geschenkkarte",
    "/products/carte-cadeau",
    "/products/tarjeta-regalo",
    "/products/carta-regalo",
    "/products/cadeaubon",
])
def test_non_product_urls_are_blocked(url):
    assert NON_PRODUCT_URL_RE.search(url), (
        f"NON_PRODUCT_URL_RE failed to match {url!r}. Disguise listings "
        "(cute title, telltale handle/image) will leak into General again."
    )


@pytest.mark.parametrize("url", [
    # Real product handles must NOT match the URL exclusion
    "/products/cotton-t-shirt",
    "/products/sun-protection-hat",
    "/products/uv400-sunglasses",
    "/products/leather-card-holder",
    "/products/mens-handmade-panama-hat",
    "https://store.com/cdn/shop/files/floral-summer-dress.jpg",
    "https://store.com/cdn/shop/files/UV-protection-sunglasses-front.jpg",
    "/products/giftbox-perfume-set",     # "giftbox" != "gift-card"
    "/products/protection-juridique-handbook",  # legal-protection, but not shipping-protection
])
def test_real_product_urls_not_blocked(url):
    assert not NON_PRODUCT_URL_RE.search(url), (
        f"NON_PRODUCT_URL_RE incorrectly matched {url!r}. The pattern is "
        "too broad and would drop legitimate product handles."
    )


def test_100_percent_coverage_disguise_is_blocked():
    """The Gents of Britain "100% Coverage" listing has a benign-looking
    title but its handle (/products/100-coverage) and image filename
    (shipping-protection.png) make it obvious. _is_non_product must catch
    it via the URL/image fields even though the title regex is borderline.
    """
    assert _is_non_product(
        title="100% Coverage",
        handle="100-coverage",
        product_url="https://gentsofbritain.com/products/100-coverage",
        image_url="https://gentsofbritain.com/cdn/shop/files/shipping-protection.png?v=1749453624",
    )
    # Even with a fully disguised generic title, the handle alone catches it
    assert _is_non_product(
        title="Premium Add-On",
        handle="shipping-protection",
        product_url="https://novigood.com/products/shipping-protection",
        image_url="https://bondimode.com/cdn/shop/files/ymq-cart-shipping-protection.png",
    )


def test_is_non_product_does_not_flag_real_product():
    assert not _is_non_product(
        title="Men's Handmade Panama Hat",
        handle="mens-handmade-panama-hat",
        product_url="https://hudsonclaye.com/products/mens-handmade-panama-hat",
        image_url="https://hudsonclaye.com/cdn/shop/files/Mens-Handmade-Panama-Hat.png",
    )
