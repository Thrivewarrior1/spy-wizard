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

# Hard-coded non-fashion patterns — items where title/type leave no doubt
# they are not clothing/shoes/bags. Caught BEFORE Gemini so the model
# can't misclassify them and so we don't burn quota on dead-certain
# rejects. Matching is case-insensitive against title + product_type.
NON_FASHION_TITLE_RE = re.compile(
    r"shipping[\s\-]*protection|package[\s\-]*protection|"
    r"route[\s\-]*(?:insurance|protect)|delivery[\s\-]*(?:guarantee|protect)|"
    r"gift[\s\-]*card|e[\s\-]*gift|store[\s\-]*credit|"
    r"\b100%\s*coverage\b|coverage[\s\-]*plan|"
    r"slidecart|order[\s\-]*protection|carbon[\s\-]*offset|"
    r"plant[\s\-]*a[\s\-]*tree|donation\b",
    re.I,
)
NON_FASHION_TYPE_RE = re.compile(
    r"shipping|protection|insurance|gift\s*card|slidecart|donation|"
    r"e[\s\-]*gift|service|warranty",
    re.I,
)


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

                # Pre-classify obvious services (shipping protection, gift
                # cards, slidecart upsells) so Gemini doesn't waste a slot
                # on them. They go straight to the General feed under the
                # 'services' subniche.
                gemini_input = []
                for p in page_products:
                    title = p.get("title", "")
                    ptype = p.get("product_type", "")
                    if (
                        NON_FASHION_TITLE_RE.search(title)
                        or (ptype and NON_FASHION_TYPE_RE.search(ptype))
                    ):
                        p["is_fashion"] = False
                        p["subniche"] = "services"
                        p["ai_tags"] = ""
                    else:
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
                    # Degraded path: route everything not blacklisted into
                    # fashion so the main feed populates.
                    for p in gemini_input:
                        p["is_fashion"] = True
                        p["subniche"] = "fashion"
                        p["ai_tags"] = ""

                # Distribute classified products into the two ranked
                # lists. Each list is independently positioned 1..N in
                # the order it encounters its members on the bestseller
                # pages, so per-feed hero/villain math stays meaningful.
                for p in page_products:
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
    Hero/villain is computed against the previous current_position only
    when the product is staying in the SAME feed (fashion↔fashion or
    general↔general); a feed flip resets the label to 'normal' so a
    product hopping between tabs doesn't get a misleading direction.
    """
    shopify_id = product_data["shopify_id"]
    new_position = product_data["position"]

    if shopify_id in existing_products:
        product = existing_products[shopify_id]
        old_position = product.current_position or 0
        was_fashion = bool(product.is_fashion)
        feed_changed = was_fashion != is_fashion

        history_count = (
            db.query(func.count(PositionHistory.id))
            .filter(PositionHistory.product_id == product.id)
            .scalar()
        ) or 0

        if feed_changed:
            # Switching feeds — old position isn't comparable anymore.
            product.previous_position = 0
            product.label = "normal"
        else:
            product.previous_position = old_position
            if history_count >= 1 and old_position > 0:
                if new_position < old_position:
                    product.label = "hero"
                elif new_position > old_position:
                    product.label = "villain"
                else:
                    product.label = "normal"
            else:
                product.label = "normal"

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
            label="normal",
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

    for product_data in fashion_products:
        _upsert_one(
            db, store, existing_products, product_data,
            is_fashion=True, subniche="fashion", now=now,
        )

    for product_data in general_products:
        sub = (product_data.get("subniche") or "other").strip().lower()
        if sub == "fashion":
            # Defensive: a misclassification slipped through. Treat as
            # 'other' on the General feed rather than mixing labels.
            sub = "other"
        _upsert_one(
            db, store, existing_products, product_data,
            is_fashion=False, subniche=sub, now=now,
        )

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
    )


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
