"""Shopify bestseller scraper — fashion-only feed.

Strategy:
  1. Fetch /collections/<slug>?sort_by=best-selling HTML pages page by
     page. The Shopify JSON endpoint ignores sort_by; only the rendered
     HTML reflects true bestseller ranking. Default slug is 'all', but
     COLLECTION_OVERRIDES lets us point individual stores at a different
     collection when their /collections/all is misconfigured upstream.
  2. Extract products from the page. Primary source is the
     `web-pixels-manager` `collection_viewed` event JSON embedded in the
     HTML — it gives us real titles, real image URLs, and (critically)
     each product's Shopify `type` (e.g. "Slidecart - Shipping
     Protection", "Women Blouse Seasonal"). When that block is missing
     we fall back to walking <main> for product anchor tags.
  3. Pre-filter obvious non-fashion items by title/type regex (shipping
     protection, gift cards, route insurance, etc.) so Gemini's quota
     isn't burned on dead-certain rejects.
  4. Classify the rest with Gemini in batches. is_fashion=False items
     never enter the fashion feed; the 100-cap is POST-filter so junk
     near the top of the bestseller list does not reduce the feed below
     100 fashion products.
  5. Position assignment: rank 1 = first fashion product encountered in
     the HTML, rank 2 = second, etc. Non-fashion items are excluded
     entirely (NOT ranked-then-hidden).
  6. Classifier failures are surfaced — they do NOT silently drop
     products. The scrape returns a list of error strings the API can
     show.
  7. Hero/villain labels only assigned once a product has >= 1 prior
     PositionHistory row.
"""
import asyncio
import json
import httpx
import logging
import os
import random
import re
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from sqlalchemy.orm import Session
from sqlalchemy import func
from models import Store, Product, PositionHistory
from classifier import classify_products_batch

logger = logging.getLogger(__name__)

HISTORY_RETENTION_DAYS = 30
# Aggressive page cap so a store with lots of non-fashion mixed near the
# top of its best-seller list can still surface our fashion target.
# Loop also exits early when a fetched page yields zero new product links
# (catalog exhausted).
MAX_PAGES = 100
TARGET_FASHION = 300

# Subniches that belong on the Fashion tab. Fashion now spans
# clothing/shoes ('fashion'), bags, accessories (hats, scarves, belts,
# sunglasses, etc.), AND jewelry (necklaces, earrings, rings, etc.).
# Any product whose Gemini-assigned subniche is in this set is forced
# to is_fashion=True regardless of what Gemini said for is_fashion —
# wearables NEVER appear on the General tab.
WEARABLE_SUBNICHES = {"fashion", "bags", "accessories", "jewelry"}
# General-feed cap. 100 of each store's bestselling NON-fashion items
# (gadgets, home decor, beauty, services like shipping protection, etc.)
# get tracked separately on the General tab. Independent positions
# 1..100, independent hero/villain, independent retirement.
TARGET_GENERAL = 100

# Per-store override for the collection path used to find best-sellers.
# Some Shopify shops have a misconfigured /collections/all (e.g. Lumenrosa
# returns only ~11 products there even though they have 2000+ in damen +
# herren). Keyed by host (lowercase, no scheme/trailing-slash); each value
# is the collection slug to use in /collections/<slug>?sort_by=best-selling.
COLLECTION_OVERRIDES = {
    "www.lumenrosa.de": "damen",
    "lumenrosa.de": "damen",
}

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

# Match both bare /products/<handle> and the collection-prefixed
# /collections/<anything>/products/<handle> form (Shopify renders the latter
# when the link sits on a collection page, which is exactly where we scrape).
PRODUCT_HREF_RE = re.compile(
    r"(?:^|/)products/([a-zA-Z0-9][a-zA-Z0-9\-_]*?)(?:\?|#|/|$)"
)

# Hard-coded patterns for items that are NOT physical products at all —
# checkout add-ons like shipping insurance, warranties, gift cards.
# Matched items are DROPPED ENTIRELY: they never enter the Fashion feed
# OR the General feed, never get classified by Gemini, never count
# toward any cap. Multi-language because our stores ship across DE/FR/
# ES/IT/NL markets. Patterns are deliberately narrow (require the
# exact compound phrase, not just a single word) to avoid false
# positives like "Garantie Card Holder" or "Tip-Top Hat".
NON_PRODUCT_TITLE_RE = re.compile(
    r"(?:^|\b)(?:"
    # === English ===
    r"(?:shipping|package|order|delivery|route)\s*"
    r"(?:protection|insurance|insured|guarantee)"
    r"|(?:100\s*%\s*coverage|coverage\s*plan|protection\s*plan)"
    r"|(?:extended\s*warranty|service\s*plan|warranty\s*plan|product\s*warranty)"
    r"|(?:gift\s*card|e[\s\-]*gift(?:\s*card)?|gift\s*voucher|store\s*credit)"
    r"|(?:carbon\s*offset|plant\s*a\s*tree|round[\s\-]?up\s*donation)"
    r"|slidecart"
    # === German ===
    r"|(?:versicherter\s*versand|versandschutz|versandversicherung|paketversicherung)"
    r"|(?:geschenkkarte|geschenkgutschein|geschenkkarten)"
    r"|(?:erweiterte\s*garantie|verl[äa]ngerte\s*garantie|garantieverl[äa]ngerung)"
    # === French ===
    r"|(?:assurance\s*(?:exp[ée]dition|livraison|colis|envoi))"
    r"|(?:protection\s*(?:exp[ée]dition|livraison|colis))"
    r"|(?:carte\s*cadeau|ch[èe]que\s*cadeau)"
    r"|(?:garantie\s*(?:[ée]tendue|prolong[ée]e))"
    # === Spanish ===
    r"|(?:protecci[óo]n\s*(?:de\s*)?(?:env[íi]o|paquete))"
    r"|(?:seguro\s*(?:de\s*)?env[íi]o)"
    r"|(?:tarjeta\s*regalo|cheque\s*regalo)"
    r"|(?:garant[íi]a\s*(?:extendida|ampliada))"
    # === Italian ===
    r"|(?:protezione\s*(?:spedizione|consegna))"
    r"|(?:assicurazione\s*(?:spedizione|consegna|trasporto))"
    r"|(?:carta\s*regalo|buono\s*regalo)"
    r"|(?:garanzia\s*estesa)"
    # === Dutch ===
    r"|(?:verzendverzekering|bezorgverzekering|pakketverzekering)"
    r"|(?:cadeaubon|cadeaukaart)"
    r"|(?:uitgebreide\s*garantie)"
    # === Surprise / mystery boxes — these are NOT real trackable
    # === SKUs (the contents change every order) so we drop them
    # === from both feeds. Same family as gift cards: a slot the
    # === merchant fills with whatever, not a real product.
    # English
    r"|(?:surprise\s*(?:box|product|package|bag|item))"
    r"|(?:mystery\s*(?:box|product|package|bag|item))"
    r"|(?:blind\s*box|lucky\s*(?:bag|box)|grab\s*bag)"
    # German (compounds — Überraschungsprodukt, Mysterybox, etc.)
    r"|(?:[üu]berraschungs(?:produkt|paket|box|tasche|t[üu]te|artikel)?)"
    r"|(?:ueberraschungs(?:produkt|paket|box|tasche|tuete|artikel)?)"
    r"|(?:mystery[\s\-]?(?:box|paket|produkt))"
    r"|(?:wundert[üu]te|wundertuete|gl[üu]ckst[üu]te|gluecktuete)"
    # French
    r"|(?:bo[îi]te\s*(?:myst[èe]re|mystere|surprise))"
    r"|(?:produit\s*(?:myst[èe]re|mystere|surprise))"
    r"|(?:pochette\s*surprise|sac\s*(?:myst[èe]re|mystere|surprise))"
    # Spanish
    r"|(?:caja\s*(?:sorpresa|misteriosa|misterio))"
    r"|(?:producto\s*sorpresa|bolsa\s*sorpresa)"
    # Italian
    r"|(?:scatola\s*(?:sorpresa|mistero|misteriosa))"
    r"|(?:prodotto\s*sorpresa|busta\s*sorpresa)"
    # Dutch
    r"|(?:verrassings(?:product|doos|pakket|tas))"
    r"|(?:mysterie(?:doos|box|product|pakket))"
    r")(?=\b|$)",
    re.IGNORECASE,
)
NON_PRODUCT_TYPE_RE = re.compile(
    r"\b(?:"
    r"slidecart|"
    r"shipping[\s\-]*protection|package[\s\-]*protection|"
    r"route[\s\-]*(?:protection|insurance)|"
    r"gift[\s\-]*card|extended\s*warranty|service\s*plan|"
    r"versandschutz|versicherter\s*versand|geschenkkarte|"
    r"protection[\s\-]*exp[ée]dition|carte\s*cadeau|"
    r"protecci[óo]n[\s\-]*env[íi]o|tarjeta[\s\-]*regalo|"
    r"protezione[\s\-]*spedizione|carta[\s\-]*regalo|"
    r"verzendverzekering|cadeaubon|"
    # Surprise / mystery box product types
    r"surprise[\s\-]*(?:box|product|package|bag)|"
    r"mystery[\s\-]*(?:box|product|package|bag)|"
    r"blind[\s\-]*box|lucky[\s\-]*(?:bag|box)|grab[\s\-]*bag|"
    r"[üu]berraschungs(?:produkt|paket|box|tasche|t[üu]te|artikel)?|"
    r"ueberraschungs(?:produkt|paket|box|tasche|tuete|artikel)?|"
    r"wundert[üu]te|wundertuete|gl[üu]ckst[üu]te|gluecktuete|"
    r"bo[îi]te[\s\-]*(?:myst[èe]re|mystere|surprise)|"
    r"caja[\s\-]*(?:sorpresa|misteriosa|misterio)|"
    r"scatola[\s\-]*(?:sorpresa|mistero|misteriosa)|"
    r"verrassings(?:product|doos|pakket|tas)|"
    r"mysterie(?:doos|box|product|pakket)"
    r")\b",
    re.IGNORECASE,
)

# Some stores disguise checkout add-ons with cute marketing titles like
# "100% Coverage" or even leave the title generic so the title regex
# can't catch them — but the Shopify handle and image filename almost
# always reveal what it really is. Match the slug/path or the image
# basename (anywhere in the URL) so we drop these regardless of title.
NON_PRODUCT_URL_RE = re.compile(
    r"(?:^|[/\-_])(?:"
    r"shipping[\-_]protection|package[\-_]protection|order[\-_]protection|"
    r"delivery[\-_](?:protection|insurance)|route[\-_](?:protection|insurance)|"
    r"shipping[\-_]insurance|package[\-_]insurance|"
    r"100[\-_]?coverage|coverage[\-_]plan|protection[\-_]plan|"
    r"extended[\-_]warranty|warranty[\-_]plan|service[\-_]plan|product[\-_]warranty|"
    r"gift[\-_]card|e[\-_]?gift(?:[\-_]card)?|gift[\-_]voucher|store[\-_]credit|"
    r"carbon[\-_]offset|plant[\-_]a[\-_]tree|round[\-_]?up[\-_]donation|"
    r"slidecart|"
    r"versicherter[\-_]versand|versandschutz|versandversicherung|paketversicherung|"
    r"geschenkkarte|geschenkgutschein|"
    r"erweiterte[\-_]garantie|verlaengerte[\-_]garantie|garantieverlaengerung|"
    r"assurance[\-_](?:expedition|livraison|colis|envoi)|"
    r"protection[\-_](?:expedition|livraison|colis)|"
    r"carte[\-_]cadeau|cheque[\-_]cadeau|"
    r"garantie[\-_](?:etendue|prolongee)|"
    r"proteccion[\-_](?:de[\-_])?(?:envio|paquete)|"
    r"seguro[\-_](?:de[\-_])?envio|"
    r"tarjeta[\-_]regalo|cheque[\-_]regalo|"
    r"garantia[\-_](?:extendida|ampliada)|"
    r"protezione[\-_](?:spedizione|consegna)|"
    r"assicurazione[\-_](?:spedizione|consegna|trasporto)|"
    r"carta[\-_]regalo|buono[\-_]regalo|garanzia[\-_]estesa|"
    r"verzendverzekering|bezorgverzekering|pakketverzekering|"
    r"cadeaubon|cadeaukaart|uitgebreide[\-_]garantie|"
    # Surprise / mystery boxes (handles, image basenames)
    r"surprise[\-_](?:box|product|package|bag|item)|"
    r"mystery[\-_](?:box|product|package|bag|item)|"
    r"blind[\-_]box|lucky[\-_](?:bag|box)|grab[\-_]bag|"
    r"ueberraschungs(?:produkt|paket|box|tasche|tuete|artikel)?|"
    r"u[\-_]?berraschungs(?:produkt|paket|box|tasche|tuete|artikel)?|"
    r"wundertuete|gluecktuete|"
    r"boite[\-_](?:mystere|surprise)|"
    r"produit[\-_](?:mystere|surprise)|"
    r"pochette[\-_]surprise|sac[\-_](?:mystere|surprise)|"
    r"caja[\-_](?:sorpresa|misteriosa|misterio)|"
    r"producto[\-_]sorpresa|bolsa[\-_]sorpresa|"
    r"scatola[\-_](?:sorpresa|mistero|misteriosa)|"
    r"prodotto[\-_]sorpresa|busta[\-_]sorpresa|"
    r"verrassings(?:product|doos|pakket|tas)|"
    r"mysterie(?:doos|box|product|pakket)"
    r")(?=$|[/\-_.?#])",
    re.IGNORECASE,
)

# Backward-compat aliases — older imports/tests may still reference the
# legacy names; keep them pointing at the new patterns until call sites
# are migrated.
NON_FASHION_TITLE_RE = NON_PRODUCT_TITLE_RE
NON_FASHION_TYPE_RE = NON_PRODUCT_TYPE_RE


# Apparel / footwear / eyewear / intimates allowlist — multilingual safety
# net inverted from NON_PRODUCT_*_RE. Items matching any pattern are
# FORCED to is_fashion=True even if Gemini routed them to electronics/
# home/beauty/health/other. The user's explicit directive: false
# positives (a pet-goggle ending up on Fashion) are far less bad than
# false negatives (a Bademantel staying on General). All twelve stores
# ship across DE/FR/ES/IT/NL/UK markets so every category is covered
# in those languages too.
#
# Compound-friendly tokens (Bademantel, Schuh, Unterwäsche) use \w*
# rather than a trailing \b so they catch German compounds like
# "Schuhüberzug" / "Frottee-Bademantel" / "Strumpfhosen" in one shot.
_FORCE_FASHION_PATTERNS = [
    # === Apparel — English ===
    r"\bt[\s\-]?shirts?\b", r"\bblouses?\b", r"\bsweaters?\b",
    r"\bhoodies?\b", r"\bjackets?\b", r"\bcoats?\b",
    r"\bdress(?:es)?\b", r"\bskirts?\b",
    r"\bjeans\b", r"\btrousers\b", r"\bjumpsuits?\b",
    r"\brobes?\b", r"\bbathrobes?\b", r"\bbathoobe\b",
    r"\bpajamas?\b", r"\bpyjamas?\b", r"\bnightgowns?\b",
    r"\bsleepwear\b", r"\bloungewear\b", r"\bponcho\b",
    r"\bcardigans?\b", r"\bleggings?\b", r"\boveralls?\b",
    # === Apparel — German (compound-friendly; umlauts cover their
    # === ASCII transliterations too, since image URLs / Shopify
    # === handles routinely flatten ä→ae, ö→oe, ü→ue).
    r"\bhemden?\b", r"\bblusen?\b", r"\bpullover\w*",
    r"\bjacken?\b", r"\bjacke\b",
    r"\bm(?:ä|a|ae)ntel\w*", r"\bmantel\w*",
    r"\bkleid\w*", r"\br(?:ö|o|oe)ck\b", r"\br(?:ö|o|oe)cke\b",
    r"\br(?:ö|o|oe)cken\b", r"\br(?:ö|o|oe)ckes\b",
    r"\bhose\b", r"\bhosen\w*", r"\bjogginghose\w*",
    r"\bsakkos?\b",
    r"\bbademantel\w*", r"\bbademaentel\w*",
    r"\bnachthemd\w*", r"\bschlafanzug\w*",
    r"\bschlafanz(?:ü|u|ue)g\w*",
    # === Apparel — French ===
    r"\bchemises?\b", r"\bchemisiers?\b", r"\bblousons?\b",
    r"\bvestes?\b", r"\bmanteaux\b", r"\bmanteau\b",
    r"\bjupes?\b", r"\bpantalons?\b", r"\bpeignoirs?\b",
    # === Apparel — Spanish ===
    r"\bcamisas?\b", r"\bcamisetas?\b", r"\bblusas?\b",
    r"\bsu[ée]teres?\b", r"\bchaquetas?\b", r"\babrigos?\b",
    r"\bvestidos?\b", r"\bfaldas?\b", r"\bbatas?\b",
    r"\bpijamas?\b",
    # === Apparel — Italian ===
    r"\bcamicie\b", r"\bcamicia\b", r"\bcamicette\b",
    r"\bmaglioni?\b", r"\bgiacche?\b", r"\bcappotti?\b",
    r"\bcappotto\b", r"\bvestiti?\b", r"\bvestito\b",
    r"\bgonne?\b", r"\bpantaloni\b", r"\bvestaglie?\b",
    # === Apparel — Dutch ===
    r"\bjurken?\b", r"\bjurk\b", r"\brokken?\b", r"\bbroeken?\b",
    r"\btruien?\b", r"\bjassen?\b", r"\bbadjassen?\b", r"\bbadjas\b",
    # === Underwear / intimates — English ===
    r"\bunderwear\b", r"\bunderpants\b", r"\bpanty\b", r"\bpanties\b",
    r"\bbriefs?\b", r"\bboxers?\b", r"\bthongs?\b",
    r"\bbras?\b", r"\blingerie\b", r"\bshapewear\b",
    r"\bhosiery\b", r"\btights\b", r"\bstockings?\b", r"\bsocks?\b",
    # === Underwear — German (umlauts also accept ASCII ae/oe/ue) ===
    r"\bunterw(?:ä|a|ae)sche\w*", r"\bunterhose\w*",
    r"\bmiederwaren\w*", r"\bstr(?:ü|u|ue)mpfe\b",
    r"\bstr(?:ü|u|ue)mpfh(?:ö|o|oe)se\w*", r"\bsocken\b",
    # === Underwear — French ===
    r"\bsous[\s\-]?v[êe]tements?\b", r"\bculottes?\b",
    r"\bsoutien[\s\-]?gorges?\b", r"\bcollants\b",
    r"\bchaussettes\b",
    # === Underwear — Spanish ===
    r"\bropa[\s\-]?interior\b", r"\bbragas?\b", r"\bbraguitas?\b",
    r"\bsost[ée]nes?\b", r"\bsost[ée]n\b", r"\bmedias\b",
    r"\bcalcetines\b",
    # === Underwear — Italian ===
    r"\bintimo\b", r"\bbiancheria[\s\-]?intima\b",
    r"\breggisen[oi]\b", r"\bcalze\b",
    # === Underwear — Dutch ===
    r"\bondergoed\b", r"\bonderbroeken?\b", r"\bbehas?\b",
    r"\bkousen\b", r"\bsokken\b",
    # === Footwear — English ===
    r"\bshoes?\b", r"\bsneakers?\b", r"\bboots?\b",
    r"\bsandals?\b", r"\bslippers?\b", r"\bslip[\s\-]?ons?\b",
    r"\bheels?\b", r"\bstilettos?\b", r"\bloafers?\b",
    r"\bflats\b", r"\boxfords?\b",
    # === Footwear — German (compound-friendly; umlauts accept ASCII too) ===
    r"\borthoschuh\w*", r"\borthop(?:ä|a|ae)disch\w*",
    r"\bschuh\w*", r"\bstiefel\w*", r"\bsandalen\b",
    r"\bhausschuh\w*", r"\bhalbschuh\w*",
    # === Footwear — French ===
    r"\bchaussures?\b", r"\bbottes?\b", r"\bsandales?\b",
    r"\bbaskets?\b", r"\bescarpins?\b",
    # === Footwear — Spanish ===
    r"\bzapatos?\b", r"\bbotas?\b", r"\bsandalias?\b",
    r"\bzapatillas?\b",
    # === Footwear — Italian ===
    r"\bscarpe\b", r"\bstivali\b", r"\bsandali\b",
    # === Footwear — Dutch ===
    r"\bschoenen\b", r"\blaarzen\b",
    # === Eyewear ===
    r"\bsunglasses\b", r"\bgoggles\b", r"\bglasses\b",
    r"\beyewear\b",
    r"\breading[\s\-]glasses\b", r"\bprogressive[\s\-]glasses\b",
    r"\bbrillen?\b", r"\bsonnenbrille\w*", r"\blesebrille\w*",
    r"\blunettes\b",
    r"\bgafas\b",
    r"\bocchiali\b",
    r"\bbril\b",
    # === Swimwear ===
    r"\bbikinis?\b", r"\bswimsuits?\b", r"\bswimwear\b",
    r"\bboard[\s\-]shorts?\b",
    r"\bbademode\w*", r"\bbadeanz(?:ü|u|ue)g\w*",
    r"\bba[ñn]ador\w*",
    r"\bcostumi?[\s\-]da[\s\-]bagno\b",
    r"\bbadpak\w*",
    # === Wedding apparel ===
    r"\bhochzeit\w*", r"\bbrautkleid\w*", r"\bbrautjungfern\w*",
    r"\bwedding[\s\-]?(?:dress|gown|guest)\w*",
    r"\brobe[\s\-]de[\s\-]mari[ée]\w*",
    r"\bvestido[\s\-]de[\s\-]novia\b",
    # === Strong fashion signals ===
    r"\bnahtlos\w*", r"\bseamless\b",
    # === Brand tokens (truncated titles per user spec) ===
    r"\bsalkin\b", r"\bsakin\b",
]

FORCE_FASHION_TITLE_RE = re.compile(
    "|".join(_FORCE_FASHION_PATTERNS),
    re.IGNORECASE,
)

# German "BH" / "B.H." abbreviation. The 2-letter form is too short to
# live among the alternations safely (alternation order would matter),
# so we keep it in its own regex and OR-merge in `_is_forced_fashion`.
FORCE_FASHION_BH_RE = re.compile(r"\bbh\b|\bb\.h\.?", re.IGNORECASE)


def _is_forced_fashion(
    title: str = "",
    product_type: str = "",
    handle: str = "",
    product_url: str = "",
    image_url: str = "",
) -> bool:
    """True iff any field looks like apparel/footwear/eyewear/intimates.
    Used as a safety net AFTER Gemini classification — a match here
    forces is_fashion=True regardless of what Gemini said. Same field
    surface as `_is_non_product`, since handle / product_url / image
    paths often reveal a category Gemini missed (e.g. handle
    'luxus-bademantel-damen' on a title that didn't include the word).
    """
    for field in (title, product_type, handle, product_url, image_url):
        if not field:
            continue
        if FORCE_FASHION_TITLE_RE.search(field):
            return True
        if FORCE_FASHION_BH_RE.search(field):
            return True
    return False


def _is_non_product(
    title: str = "",
    product_type: str = "",
    handle: str = "",
    product_url: str = "",
    image_url: str = "",
) -> bool:
    """Return True iff any of the supplied product fields look like a
    checkout add-on (shipping insurance, gift card, warranty, slidecart
    upsell, etc.). Centralised so scrape-time filtering, the per-scrape
    sweep, and the startup DB cleanup all apply identical rules.
    """
    if title and NON_PRODUCT_TITLE_RE.search(title):
        return True
    if product_type and NON_PRODUCT_TYPE_RE.search(product_type):
        return True
    for field in (handle, product_url, image_url):
        if field and NON_PRODUCT_URL_RE.search(field):
            return True
    return False


def _build_headers() -> dict:
    # IMPORTANT: do NOT include "br" in Accept-Encoding unless the brotli
    # Python package is installed. httpx's automatic decompression covers
    # gzip/deflate by default; if we advertise "br" without brotli, the
    # server returns Brotli-encoded bytes and resp.text becomes garbage,
    # which is exactly the silent-fail symptom we hit in production.
    return {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Cache-Control": "no-cache",
    }


async def _fetch_with_retry(client, url, max_retries=3):
    """Fetch URL with retry on 429 rate limiting."""
    resp = None
    for attempt in range(max_retries):
        resp = await client.get(url)
        if resp.status_code == 429:
            wait = 15 + (attempt * 15) + random.uniform(0, 5)
            logger.warning(
                f"Rate limited (429) on {url}, waiting {wait:.0f}s "
                f"(attempt {attempt+1}/{max_retries})"
            )
            await asyncio.sleep(wait)
            continue
        return resp
    return resp


def _normalize_image_url(url: str) -> str:
    if not url:
        return ""
    url = url.strip()
    if url.startswith("//"):
        return "https:" + url
    if url.startswith("http") or url.startswith("/"):
        return url
    return ""


def _extract_image_url(img) -> str:
    if not img:
        return ""
    for attr in ("src", "data-src", "data-original", "data-srcset", "srcset"):
        val = img.get(attr)
        if not val:
            continue
        if attr in ("srcset", "data-srcset"):
            first = val.split(",")[0].strip().split(" ")[0]
            normalized = _normalize_image_url(first)
        else:
            normalized = _normalize_image_url(val)
        if normalized:
            return normalized
    return ""


def _walk_to_card(a_tag, max_steps=6):
    card = a_tag
    for _ in range(max_steps):
        parent = card.parent
        if not parent:
            break
        card = parent
        if card.name in ("li", "article"):
            return card
        if card.name in ("div", "section"):
            cls = " ".join(card.get("class") or [])
            if re.search(r"product|grid__item|card|tile|item", cls, re.I):
                return card
            if card.find("img") and card.find(class_=re.compile(r"price|money|amount|title", re.I)):
                return card
    return card


_VARIANT_JSON_RE = re.compile(r'^\s*[\[{].*"id"\s*:\s*\d', re.S)

# Sale-badge / category-label text that some Shopify themes render as
# the *only* text inside the image-wrapping anchor. If we accept it as
# the title we end up with a card titled "Reduziert" or "Sale" with a
# bogus discount-percent price. Title extraction skips these and tries
# other sources within the card.
_BADGE_TEXT_RE = re.compile(
    r"^\s*(reduziert|reduced|sale|im\s*sale|on\s*sale|discount|clearance|"
    r"neu(?:heit)?|new|nouveau|nouveauté|featured|bestseller|best[\s\-]*seller|"
    r"top|hot|trending|out\s*of\s*stock|sold\s*out|"
    r"-?\d+\s*%|save\s*-?\d+\s*%?|spare\s*-?\d+\s*%?)\s*$",
    re.I,
)


def _clean_title(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    if _VARIANT_JSON_RE.match(text):
        return ""
    if text.startswith("[{") or text.startswith("{") or '"id":' in text:
        return ""
    if _BADGE_TEXT_RE.match(text):
        return ""
    text = re.sub(r"\s+", " ", text)
    if len(text) > 300:
        return ""
    return text


def _extract_title(card, a_tag, handle: str) -> str:
    text = _clean_title(a_tag.get_text(" ", strip=True))
    if text and len(text) >= 2:
        return text
    for link in card.find_all("a", href=True):
        href_path = link["href"].split("?", 1)[0].split("#", 1)[0]
        if not PRODUCT_HREF_RE.search(href_path):
            continue
        text = _clean_title(link.get_text(" ", strip=True))
        if text and len(text) >= 2:
            return text
    for el in card.find_all(class_=re.compile(r"title|product-?name|product-?heading", re.I)):
        text = _clean_title(el.get_text(" ", strip=True))
        if text and len(text) >= 2:
            return text
    img = card.find("img")
    if img:
        alt = _clean_title(img.get("alt") or "")
        if alt and len(alt) >= 2:
            return alt
    return handle.replace("-", " ").title()


def _extract_price(card) -> str:
    """Pull the displayed price from the card. Skips elements whose
    class hints they hold a discount percentage or sale badge — those
    show up as "-30%" or "Save 49%" and would otherwise be picked up
    as a 30 or 49 price (which the frontend then renders as €0.30 /
    €0.49).
    """
    for el in card.find_all(class_=re.compile(r"price|money|amount", re.I)):
        cls = " ".join(el.get("class") or [])
        if re.search(r"badge|label|percent|saving|save|discount|sale-", cls, re.I):
            continue
        text = el.get_text(" ", strip=True)
        if "%" in text:
            continue
        # Match a plausible currency-style number. Reject standalone
        # one- or two-digit ints that look like discount counters.
        match = re.search(r"\d{1,3}(?:[.,]\d{2})", text) or re.search(r"\d{2,}", text)
        if match:
            return match.group()
    return ""


def _find_events_string(html_text: str) -> str | None:
    """Locate the JSON-string-encoded `events` payload that Shopify's
    web-pixels-manager script embeds in every collection page. Returns
    the decoded events JSON text (still a JSON string of the events
    array), or None when the page doesn't ship the block.
    """
    needle = '"events":"'
    idx = html_text.find(needle)
    if idx == -1:
        return None
    start = idx + len(needle)
    # Walk forward, respecting backslash escapes, until the matching
    # unescaped closing quote of the events string.
    i = start
    n = len(html_text)
    while i < n:
        c = html_text[i]
        if c == "\\":
            i += 2  # skip escaped char (covers \", \\, \/, \uXXXX, etc.)
            continue
        if c == '"':
            try:
                # Wrapping back in quotes lets json.loads handle the
                # escape sequences cleanly.
                return json.loads('"' + html_text[start:i] + '"')
            except json.JSONDecodeError:
                return None
        i += 1
    return None


def _extract_products_from_events(html_text: str, base_url: str, seen: set) -> list:
    """Parse the `collection_viewed` event embedded in the page to recover
    title, image, type, vendor and price for each product on this page.

    This is the preferred extraction path because Shopify's own script
    serialises the data we need (image.src is always the real CDN URL,
    product.type tells us when something is shipping protection, etc.).
    Returns an empty list when the event block isn't present so the
    caller can fall back to the anchor-walk parser.
    """
    events_text = _find_events_string(html_text)
    if not events_text:
        return []
    try:
        events = json.loads(events_text)
    except json.JSONDecodeError:
        return []

    variants = []
    for ev in events or []:
        # Each event is [name, payload]; we want the collection_viewed one.
        if not isinstance(ev, list) or len(ev) < 2:
            continue
        if ev[0] != "collection_viewed":
            continue
        payload = ev[1] or {}
        coll = payload.get("collection") or {}
        v = coll.get("productVariants") or []
        if isinstance(v, list):
            variants.extend(v)

    if not variants:
        return []

    products = []
    seen_pid = set()
    for variant in variants:
        prod = (variant or {}).get("product") or {}
        pid = prod.get("id")
        if not pid or pid in seen_pid:
            continue
        seen_pid.add(pid)

        url_path = (prod.get("url") or "").split("?", 1)[0]
        m = PRODUCT_HREF_RE.search(url_path)
        if not m:
            continue
        handle = m.group(1).lower()
        if handle in seen:
            continue
        seen.add(handle)

        title = (prod.get("untranslatedTitle") or prod.get("title") or "").strip()
        if not title:
            title = handle.replace("-", " ").title()

        img = (variant.get("image") or {}).get("src") or ""
        if img.startswith("//"):
            img = "https:" + img
        elif img.startswith("/"):
            img = base_url + img

        price = ""
        p = variant.get("price") or {}
        amt = p.get("amount")
        if isinstance(amt, (int, float)):
            price = f"{amt:.2f}"

        products.append({
            "shopify_id": handle,
            "title": title,
            "handle": handle,
            "image_url": img,
            "price": price,
            "vendor": (prod.get("vendor") or "").strip(),
            "product_type": (prod.get("type") or "").strip(),
            "product_url": f"{base_url}/products/{handle}",
        })

    return products


def _extract_products_from_html(soup: BeautifulSoup, base_url: str, seen: set, html_text: str = "") -> list:
    """Parse a Shopify collection HTML page. Prefers the embedded
    web-pixels-manager events JSON for accuracy; falls back to anchor-
    tag walking when the events block isn't present.

    Both paths share the cross-page `seen` set so a product is never
    emitted twice.
    """
    if html_text:
        products = _extract_products_from_events(html_text, base_url, seen)
        if products:
            return products

    main = soup.find("main") or soup.find(id="MainContent") or soup.body or soup
    products = []

    for a_tag in main.find_all("a", href=True):
        path = a_tag["href"].split("?", 1)[0].split("#", 1)[0]
        match = PRODUCT_HREF_RE.search(path)
        if not match:
            continue
        handle = match.group(1).lower()
        if handle in seen:
            continue
        seen.add(handle)

        card = _walk_to_card(a_tag)
        title = _extract_title(card, a_tag, handle)
        image_url = _extract_image_url(card.find("img"))
        if not image_url:
            # Some Shopify themes lazy-load via <picture><source srcset>
            # or wider data-* attrs that aren't on the immediate <img>.
            for img in card.find_all("img"):
                image_url = _extract_image_url(img)
                if image_url:
                    break
            if not image_url:
                for src in card.find_all("source"):
                    image_url = _extract_image_url(src)
                    if image_url:
                        break
        price = _extract_price(card)

        products.append({
            "shopify_id": handle,
            "title": title,
            "handle": handle,
            "image_url": image_url,
            "price": price,
            "vendor": "",
            "product_type": "",
            "product_url": f"{base_url}/products/{handle}",
        })

    return products


async def scrape_store_bestsellers(
    store_url: str,
    target_fashion: int = TARGET_FASHION,
    target_general: int = TARGET_GENERAL,
):
    """Scrape a store's best-selling collection into TWO ranked lists:
    fashion-only (up to `target_fashion`) and everything-else
    (up to `target_general`). Both come from the same HTML pages.

    Returns (fashion_products, general_products, errors):
      - fashion_products: up to `target_fashion` items with sequential
        positions 1..N reflecting fashion-only bestseller order.
      - general_products: up to `target_general` items with sequential
        positions 1..N reflecting general bestseller order. Each carries
        a `subniche` label (jewelry/electronics/home/beauty/services/...)
        from Gemini for the General tab's filter pills.
      - errors: list of human-readable strings describing any non-fatal
        problems. Empty when everything was clean.
    """
    base_url = store_url.rstrip("/")
    # Pick the collection slug to scrape. Defaults to the universal
    # /collections/all, but stores with a broken /collections/all (e.g.
    # Lumenrosa returns only 11 even though the catalog has 2000+) have
    # a per-host override so we land on a collection that actually
    # holds their full assortment ranked by best-selling.
    host = re.sub(r"^https?://", "", base_url, flags=re.I).split("/", 1)[0].lower()
    collection_slug = COLLECTION_OVERRIDES.get(host, "all")
    fashion: list = []
    general: list = []
    seen: set = set()
    errors: list = []
    has_gemini = bool(os.getenv("GEMINI_API_KEY"))
    if not has_gemini:
        # Degraded mode: without Gemini we cannot split fashion vs general,
        # so everything goes to the fashion feed unclassified. Loud warning.
        errors.append(
            "GEMINI_API_KEY is not set on the server — fashion/general "
            "classification is DISABLED. Set GEMINI_API_KEY in Railway env "
            "vars to enable strict filtering."
        )

    try:
        async with httpx.AsyncClient(
            timeout=30.0, follow_redirects=True, headers=_build_headers()
        ) as client:
            page = 1
            while (
                (len(fashion) < target_fashion or len(general) < target_general)
                and page <= MAX_PAGES
            ):
                url = f"{base_url}/collections/{collection_slug}?sort_by=best-selling&page={page}"
                try:
                    resp = await _fetch_with_retry(client, url)
                except httpx.HTTPError as e:
                    errors.append(f"page {page} HTTP error: {e}")
                    break

                if resp is None or resp.status_code != 200:
                    status = resp.status_code if resp is not None else "no response"
                    errors.append(f"page {page} non-200 ({status})")
                    break

                soup = BeautifulSoup(resp.text, "html.parser")
                page_products = _extract_products_from_html(
                    soup, base_url, seen, html_text=resp.text
                )
                logger.info(
                    f"{base_url} page {page}: parsed {len(page_products)} new products "
                    f"(fashion={len(fashion)} general={len(general)})"
                )

                if not page_products:
                    break

                # Hard exclusion pass — items matching NON_PRODUCT_*_RE
                # are checkout add-ons (shipping insurance, warranties,
                # gift cards, slidecart upsells, donations, etc.). Drop
                # them ENTIRELY before they hit Gemini: they never enter
                # the Fashion feed OR the General feed and don't count
                # toward either cap, so the loop keeps paginating to
                # backfill real products in their place.
                gemini_input = []
                for p in page_products:
                    if _is_non_product(
                        title=p.get("title", ""),
                        product_type=p.get("product_type", ""),
                        handle=p.get("handle", ""),
                        product_url=p.get("product_url", ""),
                        image_url=p.get("image_url", ""),
                    ):
                        p["_excluded"] = True
                        continue
                    gemini_input.append(p)

                if has_gemini and gemini_input:
                    ok, classifier_errors = await _classify_or_fail(gemini_input)
                    errors.extend(classifier_errors)
                    if not ok:
                        # Hard fail — without classification we cannot
                        # split fashion vs general reliably, so stop
                        # rather than corrupt the feeds.
                        break
                elif not has_gemini:
                    # Degraded path: route everything that survived the
                    # hard-exclude pass into fashion so the main feed
                    # populates.
                    for p in gemini_input:
                        p["is_fashion"] = True
                        p["subniche"] = "fashion"
                        p["ai_tags"] = ""

                # Distribute classified products into the two ranked
                # lists. Each list is independently positioned 1..N in
                # the order it encounters its members on the bestseller
                # pages, so per-feed hero/villain math stays meaningful.
                # Anything Gemini flagged as 'exclude' (a non-product
                # the regex missed) is dropped here too.
                #
                # Reconciliation: a wearable subniche
                # (fashion/bags/accessories/jewelry) ALWAYS implies the
                # Fashion feed even if Gemini accidentally returned
                # is_fashion=false. Catches the case where Gemini
                # classifies an earring as 'jewelry' but forgets to
                # flip is_fashion. The user explicitly wants jewelry
                # and accessories on Fashion, never on General.
                for p in page_products:
                    if p.get("_excluded"):
                        continue
                    sub = (p.get("subniche") or "").strip().lower()
                    if sub == "exclude":
                        continue
                    if sub in WEARABLE_SUBNICHES:
                        p["is_fashion"] = True
                    # Apparel / footwear / eyewear / intimates safety net.
                    # Gemini occasionally routes Bademantel to 'home',
                    # Unterwäsche to 'beauty', orthopedic shoes to
                    # 'health', wedding dresses to 'other', etc. The
                    # multilingual allowlist forces is_fashion=True for
                    # any wearable category Gemini missed and rewrites
                    # the subniche to 'fashion' so the upsert doesn't
                    # later squash it back to a General-feed bucket.
                    if _is_forced_fashion(
                        title=p.get("title", ""),
                        product_type=p.get("product_type", ""),
                        handle=p.get("handle", ""),
                        product_url=p.get("product_url", ""),
                        image_url=p.get("image_url", ""),
                    ):
                        p["is_fashion"] = True
                        if sub not in WEARABLE_SUBNICHES:
                            p["subniche"] = "fashion"
                            sub = "fashion"
                    if p.get("is_fashion"):
                        if len(fashion) < target_fashion:
                            p["position"] = len(fashion) + 1
                            fashion.append(p)
                    else:
                        if len(general) < target_general:
                            p["position"] = len(general) + 1
                            general.append(p)

                if len(fashion) >= target_fashion and len(general) >= target_general:
                    break

                page += 1
                await asyncio.sleep(2 + random.uniform(0, 2))
    except Exception as e:
        errors.append(f"unexpected error: {e}")
        logger.exception(f"Error scraping {base_url}")
        return fashion, general, errors

    if not fashion and not general:
        if not errors:
            errors.append("no products parsed from any page")

    logger.info(
        f"{base_url}: returning {len(fashion)} fashion + {len(general)} general "
        f"(errors={len(errors)})"
    )
    return fashion, general, errors


async def _classify_or_fail(batch: list):
    """Run Gemini classification on a batch. Returns (ok, errors).

    Errors include the Gemini HTTP body / exception text propagated up from
    classifier.py so we can see the real cause (auth, quota, model-not-
    found, schema rejection) instead of a generic "did not classify"
    summary that hides the root failure.
    """
    if not batch:
        return True, []
    # Pre-set defaults so we can detect items Gemini didn't return for.
    for p in batch:
        p.setdefault("is_fashion", None)
        p.setdefault("ai_tags", "")
    try:
        classifier_errors = await classify_products_batch(batch)
    except Exception as e:
        return False, [f"Gemini exception: {type(e).__name__}: {e}"]

    errors: list = []
    if classifier_errors:
        # Cap to first 3 unique messages so the API response stays readable.
        seen = set()
        for msg in classifier_errors:
            if msg in seen:
                continue
            seen.add(msg)
            errors.append(f"Gemini error: {msg}")
            if len(errors) >= 3:
                break

    missing = [p["handle"] for p in batch if p.get("is_fashion") is None]
    if missing:
        sample = ", ".join(missing[:5])
        more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        errors.append(f"Gemini did not classify {len(missing)} items: {sample}{more}")
        return False, errors

    return True, errors


async def debug_fetch(store_url: str) -> dict:
    """Diagnostic helper: report what the server actually receives when
    fetching the store's best-seller page. Useful when production says
    'no products parsed' but local testing succeeds — usually means the
    upstream is serving a different page (Cloudflare challenge, etc.)."""
    base_url = store_url.rstrip("/")
    host = re.sub(r"^https?://", "", base_url, flags=re.I).split("/", 1)[0].lower()
    collection_slug = COLLECTION_OVERRIDES.get(host, "all")
    url = f"{base_url}/collections/{collection_slug}?sort_by=best-selling&page=1"
    try:
        async with httpx.AsyncClient(
            timeout=30.0, follow_redirects=True, headers=_build_headers()
        ) as client:
            resp = await client.get(url)
    except Exception as e:
        return {"url": url, "error": str(e)}

    body = resp.text
    soup = BeautifulSoup(body, "html.parser")
    main = soup.find("main") or soup.find(id="MainContent")
    sample_links = []
    seen = set()
    container = main or soup.body or soup
    for a in container.find_all("a", href=True):
        path = a["href"].split("?", 1)[0].split("#", 1)[0]
        m = PRODUCT_HREF_RE.search(path)
        if not m:
            continue
        h = m.group(1).lower()
        if h in seen:
            continue
        seen.add(h)
        sample_links.append({"href": a["href"], "handle": h})
        if len(sample_links) >= 10:
            break

    raw_product_match_count = len(PRODUCT_HREF_RE.findall(body))

    return {
        "url": url,
        "status": resp.status_code,
        "final_url": str(resp.url),
        "content_length": len(body),
        "main_found": main is not None,
        "sample_text": body[:1000],
        "raw_product_match_count": raw_product_match_count,
        "unique_product_links_in_main": len(seen),
        "sample_product_links": sample_links,
    }


def _upsert_one(
    db: Session,
    store: Store,
    existing_products: dict,
    product_data: dict,
    *,
    is_fashion: bool,
    subniche: str,
    now: datetime,
):
    """Upsert one product row and write a PositionHistory snapshot.

    Note: hero/villain `label` and `previous_position` are NO LONGER
    written here. Both are computed at READ time from PositionHistory
    in main.py's _compute_label_map / _prior_position_subquery, so they
    always reflect a real day-over-day delta against the most recent
    snapshot dated < today (UTC). Storing them at scrape time made
    labels go stale during structural changes (e.g. the 100→300 target
    bump generated 73 spurious 'villain' rows even though no product
    had actually moved).
    """
    shopify_id = product_data["shopify_id"]
    new_position = product_data["position"]

    if shopify_id in existing_products:
        product = existing_products[shopify_id]
        product.current_position = new_position
        product.title = product_data["title"]
        product.image_url = product_data["image_url"]
        product.price = product_data["price"]
        product.product_url = product_data["product_url"]
        product.vendor = product_data.get("vendor", "")
        product.product_type = product_data.get("product_type", "")
        product.ai_tags = product_data.get("ai_tags", "")
        product.is_fashion = is_fashion
        product.subniche = subniche
        product.last_scraped = now
    else:
        product = Product(
            store_id=store.id,
            shopify_id=shopify_id,
            title=product_data["title"],
            handle=product_data["handle"],
            image_url=product_data["image_url"],
            price=product_data["price"],
            vendor=product_data.get("vendor", ""),
            product_type=product_data.get("product_type", ""),
            product_url=product_data["product_url"],
            current_position=new_position,
            previous_position=0,
            label="",
            ai_tags=product_data.get("ai_tags", ""),
            is_fashion=is_fashion,
            subniche=subniche,
            last_scraped=now,
        )
        db.add(product)
        db.flush()

    db.add(PositionHistory(
        product_id=product.id,
        position=new_position,
        date=now,
    ))


def update_products_in_db(
    db: Session,
    store: Store,
    fashion_products: list,
    general_products: list | None = None,
):
    """Persist this scrape's fashion AND general lists for `store`.

    Each list has its own position numbering 1..N. Retirement is also
    per-feed:
      - any existing fashion row whose shopify_id isn't in this scrape's
        fashion list flips is_fashion=False (drops out of Fashion tab),
      - any existing non-fashion row whose shopify_id isn't in the
        general list has its subniche cleared (drops out of General tab).

    Both retirements only run when the corresponding list is at least
    half-full so a partial scrape can't wipe legitimate data.
    """
    general_products = general_products or []
    existing_products = {p.shopify_id: p for p in store.products if p.shopify_id}
    now = datetime.utcnow()
    fashion_ids = {p["shopify_id"] for p in fashion_products}
    general_ids = {p["shopify_id"] for p in general_products}

    # Allowed subniches per feed. Fashion now spans clothing/shoes
    # (subniche='fashion'), bags, accessories, AND jewelry — all
    # 'wearable' categories. Anything else Gemini emits collapses to
    # the safe default for that feed.
    FASHION_SUBNICHES = {"fashion", "bags", "accessories", "jewelry"}
    GENERAL_SUBNICHES = {
        "electronics", "home", "beauty", "health", "food", "toys-books", "other",
    }

    for product_data in fashion_products:
        sub = (product_data.get("subniche") or "fashion").strip().lower()
        if sub not in FASHION_SUBNICHES:
            # Gemini said this was fashion but tagged it with a
            # non-fashion subniche — fall back to the generic
            # 'fashion' label rather than mixing General categories
            # into the Fashion feed.
            sub = "fashion"
        _upsert_one(
            db, store, existing_products, product_data,
            is_fashion=True, subniche=sub, now=now,
        )

    for product_data in general_products:
        sub = (product_data.get("subniche") or "other").strip().lower()
        if sub not in GENERAL_SUBNICHES:
            # Defensive: Gemini routed a wearable-category subniche
            # (fashion/bags/accessories/jewelry) into the general
            # bucket. The is_fashion flag is the source of truth, but
            # the subniche needs sanitising so the General feed query
            # (subniche != '') doesn't surface it under a wearable
            # label. Bucket as 'other'.
            sub = "other"
        _upsert_one(
            db, store, existing_products, product_data,
            is_fashion=False, subniche=sub, now=now,
        )

    # Unconditional junk sweep — any existing product whose title,
    # product_type, handle, product_url, or image_url NOW looks like
    # a checkout add-on is purged from BOTH feeds, even if the rest
    # of the per-feed retirement is skipped because the current
    # scrape is partial. This catches legacy "services" entries left
    # over from before the regex was tightened, plus disguised
    # listings like "100% Coverage" whose title alone looks innocent
    # but whose handle/image clearly say shipping-protection.
    junk_purged = 0
    for shopify_id, product in existing_products.items():
        if not (product.is_fashion or product.subniche):
            continue
        if _is_non_product(
            title=product.title or "",
            product_type=product.product_type or "",
            handle=product.handle or "",
            product_url=product.product_url or "",
            image_url=product.image_url or "",
        ):
            product.is_fashion = False
            product.subniche = ""
            junk_purged += 1

    # Apparel / footwear / eyewear / intimates sweep. Promotes any
    # is_fashion=False row that NOW matches the multilingual fashion
    # allowlist (Bademantel, Unterwäsche, Orthoschuh, Brille, etc.)
    # to is_fashion=True. Coerces the subniche to 'fashion' if it was
    # previously a General bucket so downstream queries don't surface
    # it under a non-wearable label.
    forced_promoted = 0
    for shopify_id, product in existing_products.items():
        if product.is_fashion:
            continue
        if not _is_forced_fashion(
            title=product.title or "",
            product_type=product.product_type or "",
            handle=product.handle or "",
            product_url=product.product_url or "",
            image_url=product.image_url or "",
        ):
            continue
        product.is_fashion = True
        sub = (product.subniche or "").strip().lower()
        if sub not in WEARABLE_SUBNICHES:
            product.subniche = "fashion"
        forced_promoted += 1

    # Per-feed retirement. We never look at the OTHER feed's IDs when
    # deciding whether to retire — a product that moved feeds is already
    # represented in its new feed's list.
    fashion_retired = 0
    general_retired = 0
    if len(fashion_products) >= max(30, TARGET_FASHION // 2):
        for shopify_id, product in existing_products.items():
            if (
                product.is_fashion
                and shopify_id not in fashion_ids
                and shopify_id not in general_ids
            ):
                product.is_fashion = False
                product.subniche = ""
                fashion_retired += 1
    if len(general_products) >= max(15, TARGET_GENERAL // 2):
        for shopify_id, product in existing_products.items():
            if (
                not product.is_fashion
                and product.subniche
                and shopify_id not in general_ids
                and shopify_id not in fashion_ids
            ):
                product.subniche = ""
                general_retired += 1

    db.commit()
    logger.info(
        f"Updated {store.name}: {len(fashion_products)} fashion + "
        f"{len(general_products)} general"
        + (f" (retired f={fashion_retired} g={general_retired})" if fashion_retired or general_retired else "")
        + (f" (purged {junk_purged} junk)" if junk_purged else "")
        + (f" (promoted {forced_promoted} apparel)" if forced_promoted else "")
    )


def migrate_wearables_to_fashion(db: Session) -> int:
    """One-shot DB migration: any existing row stored as
    is_fashion=False with a wearable subniche (jewelry, accessories,
    bags) belongs on the Fashion tab under the new classification.
    Flip is_fashion=True so the change takes effect immediately on
    startup rather than waiting for the next scrape to re-classify
    every store. Idempotent — safe to call on every restart.
    """
    rows = db.query(Product).filter(
        Product.is_fashion == False,
        Product.subniche.in_(list(WEARABLE_SUBNICHES - {"fashion"})),
    ).all()
    for p in rows:
        p.is_fashion = True
    if rows:
        db.commit()
        logger.info(
            f"migrate_wearables_to_fashion: promoted {len(rows)} "
            f"jewelry/accessories/bags rows to the Fashion feed"
        )
    return len(rows)


def migrate_apparel_to_fashion(db: Session) -> int:
    """One-shot DB migration: any existing General-tab row whose title,
    product_type, handle, product_url, or image_url matches the
    multilingual apparel/footwear/eyewear/intimates allowlist gets
    promoted to is_fashion=True with subniche='fashion' (unless it
    already had a wearable subniche, in which case the subniche is
    preserved). Sister to migrate_wearables_to_fashion — this one
    handles items Gemini routed to electronics/home/beauty/health/
    other that are clearly clothing/shoes/glasses/underwear.
    Idempotent — safe to call on every restart.
    """
    rows = db.query(Product).filter(Product.is_fashion == False).all()
    promoted = 0
    for product in rows:
        if not _is_forced_fashion(
            title=product.title or "",
            product_type=product.product_type or "",
            handle=product.handle or "",
            product_url=product.product_url or "",
            image_url=product.image_url or "",
        ):
            continue
        product.is_fashion = True
        sub = (product.subniche or "").strip().lower()
        if sub not in WEARABLE_SUBNICHES:
            product.subniche = "fashion"
        promoted += 1
    if promoted:
        db.commit()
        logger.info(
            f"migrate_apparel_to_fashion: promoted {promoted} "
            f"apparel/footwear/eyewear/intimate rows to the Fashion feed"
        )
    return promoted


def cleanup_non_product_rows(db: Session) -> int:
    """One-shot DB-wide sweep: any product whose title, product_type,
    handle, product_url, or image_url indicates a checkout add-on
    gets is_fashion=False and subniche="". Idempotent — safe to call
    on every startup. Catches legacy junk that existed before the
    regex was tightened, including disguised titles like "100% Coverage"
    whose handle (/products/100-coverage) and image (shipping-protection.png)
    give them away.
    """
    rows = db.query(Product).filter(
        (Product.is_fashion == True) | (Product.subniche != "")
    ).all()
    purged = 0
    for product in rows:
        if _is_non_product(
            title=product.title or "",
            product_type=product.product_type or "",
            handle=product.handle or "",
            product_url=product.product_url or "",
            image_url=product.image_url or "",
        ):
            product.is_fashion = False
            product.subniche = ""
            purged += 1
    if purged:
        db.commit()
        logger.info(f"cleanup_non_product_rows: purged {purged} legacy junk rows")
    return purged


def cleanup_old_history(db: Session, retention_days: int = HISTORY_RETENTION_DAYS) -> int:
    cutoff = datetime.utcnow() - timedelta(days=retention_days)
    deleted = (
        db.query(PositionHistory)
        .filter(PositionHistory.date < cutoff)
        .delete(synchronize_session=False)
    )
    db.commit()
    logger.info(f"Pruned {deleted} position_history rows older than {retention_days} days")
    return deleted


def reset_all_labels(db: Session) -> int:
    count = (
        db.query(Product)
        .update(
            {Product.label: "normal", Product.previous_position: 0},
            synchronize_session=False,
        )
    )
    db.commit()
    logger.info(f"Reset labels and previous_position for {count} products")
    return count


async def scrape_all_stores(db: Session) -> dict:
    """Scrape all stores. Returns a per-store summary so the API can
    surface real success/failure (counts, errors) to the user."""
    stores = db.query(Store).all()
    logger.info(f"Starting scrape of {len(stores)} stores...")

    results = {
        "stores": [],
        "total_products": 0,
        "total_general": 0,
        "stores_with_products": 0,
        "stores_failed": 0,
    }

    for store in stores:
        store_result = {
            "id": store.id, "name": store.name,
            "products": 0, "general": 0, "errors": [],
        }
        logger.info(f"Scraping {store.name} ({store.url})...")
        try:
            fashion, general, errors = await scrape_store_bestsellers(store.url)
            store_result["errors"] = errors
            if fashion or general:
                update_products_in_db(db, store, fashion, general)
                store_result["products"] = len(fashion)
                store_result["general"] = len(general)
                results["total_products"] += len(fashion)
                results["total_general"] += len(general)
                results["stores_with_products"] += 1
                logger.info(
                    f"  ✓ {len(fashion)} fashion + {len(general)} general "
                    f"for {store.name}"
                )
            else:
                results["stores_failed"] += 1
                logger.warning(
                    f"  ✗ No products for {store.name} — errors: {errors}"
                )
        except Exception as e:
            store_result["errors"].append(f"unhandled: {e}")
            results["stores_failed"] += 1
            logger.exception(f"Failed to scrape {store.name}")

        results["stores"].append(store_result)
        await asyncio.sleep(5 + random.uniform(0, 3))

    try:
        cleanup_old_history(db)
    except Exception as e:
        logger.error(f"History cleanup failed: {e}")

    logger.info(
        f"Scrape complete: {results['total_products']} fashion + "
        f"{results['total_general']} general across "
        f"{results['stores_with_products']}/{len(stores)} stores"
    )
    return results
