"""Shopify bestseller scraper.

Strategy:
  1. Fetch /collections/all?sort_by=best-selling HTML pages (the JSON API
     ignores sort_by and returns alphabetical order — only the rendered
     HTML reflects true bestseller ranking).
  2. Group product links by handle. Position = DOM order of the FIRST link
     to each handle inside <main>, so position 1 = the first product card
     visible on the page = the true #1 best seller.
  3. For each handle, gather title/image/price from the BEST data across
     all card links pointing to it (image-only link + title-bearing link
     are typical Shopify card markup).
  4. Gemini classification is BEST-EFFORT only. Missing key, API failures,
     or per-product errors do NOT drop products from the feed. Products
     keep their true position even when Gemini is unavailable.
  5. Hero/villain labels are only assigned once a product has >= 2 prior
     PositionHistory rows.
"""
import asyncio
import httpx
import logging
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
MAX_PAGES = 10
DEFAULT_MAX_PRODUCTS = 100

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


async def _fetch_with_retry(client, url, headers=None, max_retries=3):
    """Fetch URL with retry on 429 rate limiting."""
    resp = None
    for attempt in range(max_retries):
        resp = await client.get(url, headers=headers) if headers is not None else await client.get(url)
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
    """Pull the best-available image URL off a Shopify product card <img>."""
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
    """Walk up from a product link to its enclosing card container."""
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


def _clean_title(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    # Some Shopify themes render variant JSON inside <select> options that
    # land in the card. Drop anything that looks like a JSON dump.
    if _VARIANT_JSON_RE.match(text):
        return ""
    if text.startswith("[{") or text.startswith("{") or '"id":' in text:
        return ""
    text = re.sub(r"\s+", " ", text)
    if len(text) > 300:
        return ""
    return text


def _extract_title(card, a_tag, handle: str) -> str:
    """Find the cleanest title for a product card.

    Tries (in order): the link's own text, any other product-link text in
    the card, a title-class element, the image alt, then handle.
    """
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
        alt = _clean_title((img.get("alt") or ""))
        if alt and len(alt) >= 2:
            return alt

    return handle.replace("-", " ").title()


def _extract_price(card) -> str:
    """Pull a numeric price from any price-like element in the card."""
    for el in card.find_all(class_=re.compile(r"price|money|amount", re.I)):
        text = el.get_text(" ", strip=True)
        match = re.search(r"\d+(?:[.,]\d+)?", text)
        if match:
            return match.group()
    return ""


def _extract_products_from_html(soup: BeautifulSoup, base_url: str, seen: set) -> list:
    """Parse a Shopify collection HTML page. Returns products in display
    order, deduped by handle. Position assignment happens at scrape level.
    """
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


async def scrape_store_bestsellers(store_url: str, max_products: int = DEFAULT_MAX_PRODUCTS) -> list:
    """Scrape bestsellers from a Shopify store via the HTML bestseller page.

    Returns up to `max_products` products in true bestseller display order,
    each with sequential `position` 1..N reflecting their actual rank on
    the live page. Gemini is best-effort: when it succeeds it adds ai_tags
    and may mark items non-fashion, but the FEED is never starved by a
    Gemini failure — products default to is_fashion=True.
    """
    base_url = store_url.rstrip("/")
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Cache-Control": "no-cache",
    }

    raw_products: list = []
    seen: set = set()

    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True, headers=headers) as client:
            page = 1
            while len(raw_products) < max_products and page <= MAX_PAGES:
                url = f"{base_url}/collections/all?sort_by=best-selling&page={page}"
                try:
                    resp = await _fetch_with_retry(client, url)
                except httpx.HTTPError as e:
                    logger.warning(f"HTTP error for {url}: {e}")
                    break

                if resp is None or resp.status_code != 200:
                    status = resp.status_code if resp is not None else "no response"
                    logger.warning(f"Non-200 response for {base_url} page={page}: {status}")
                    break

                soup = BeautifulSoup(resp.text, "html.parser")
                page_products = _extract_products_from_html(soup, base_url, seen)

                if not page_products:
                    break

                raw_products.extend(page_products)
                page += 1
                await asyncio.sleep(2 + random.uniform(0, 2))

    except Exception as e:
        logger.error(f"Error scraping {base_url}: {e}")
        return []

    if not raw_products:
        logger.warning(f"No products parsed from {base_url}")
        return []

    raw_products = raw_products[:max_products]

    # True bestseller rank from the rendered HTML — assigned BEFORE Gemini
    # so it can never be perturbed by classification.
    for idx, p in enumerate(raw_products, start=1):
        p["position"] = idx
        p["is_fashion"] = True
        p["ai_tags"] = ""

    # Gemini is enrichment, not a gate. It may set is_fashion=False on
    # obvious non-clothing items (gift cards, shipping protection) and add
    # search tags. If it fails or no key is set, products stay is_fashion=True.
    try:
        classify_products_batch(raw_products)
    except Exception as e:
        logger.warning(f"Gemini classification failed for {base_url}: {e}")

    fashion_count = sum(1 for p in raw_products if p.get("is_fashion"))
    logger.info(
        f"{base_url}: scraped {len(raw_products)} products "
        f"(positions 1..{len(raw_products)}, fashion={fashion_count})"
    )
    return raw_products


def update_products_in_db(db: Session, store: Store, scraped_products: list):
    """Update products and assign hero/villain/normal based on rank delta.

    Label rule: hero/villain only when the product already has >= 2 prior
    PositionHistory rows. The first two scrapes always produce 'normal'.
    """
    existing_products = {p.shopify_id: p for p in store.products if p.shopify_id}

    now = datetime.utcnow()

    for product_data in scraped_products:
        shopify_id = product_data["shopify_id"]
        new_position = product_data["position"]

        if shopify_id in existing_products:
            product = existing_products[shopify_id]
            old_position = product.current_position or 0

            history_count = (
                db.query(func.count(PositionHistory.id))
                .filter(PositionHistory.product_id == product.id)
                .scalar()
            ) or 0

            product.previous_position = old_position
            product.current_position = new_position

            product.title = product_data["title"]
            product.image_url = product_data["image_url"]
            product.price = product_data["price"]
            product.product_url = product_data["product_url"]
            product.vendor = product_data.get("vendor", "")
            product.product_type = product_data.get("product_type", "")
            product.ai_tags = product_data.get("ai_tags", "")
            product.is_fashion = bool(product_data.get("is_fashion", True))
            product.last_scraped = now

            if history_count >= 2 and old_position > 0:
                if new_position < old_position:
                    product.label = "hero"
                elif new_position > old_position:
                    product.label = "villain"
                else:
                    product.label = "normal"
            else:
                product.label = "normal"

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
                is_fashion=bool(product_data.get("is_fashion", True)),
                last_scraped=now,
            )
            db.add(product)
            db.flush()

        history = PositionHistory(
            product_id=product.id,
            position=new_position,
            date=now,
        )
        db.add(history)

    db.commit()
    logger.info(f"Updated {len(scraped_products)} products for {store.name}")


def cleanup_old_history(db: Session, retention_days: int = HISTORY_RETENTION_DAYS) -> int:
    """Delete PositionHistory rows older than `retention_days`. Returns rows removed."""
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
    """Reset every product's label to 'normal' and clear previous_position tracking."""
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
    """Scrape all stores, update DB, and prune old history.

    Returns a dict with per-store counts so callers (API endpoint, scheduler)
    can surface real success/failure to the user.
    """
    stores = db.query(Store).all()
    logger.info(f"Starting scrape of {len(stores)} stores...")

    results = {"stores": [], "total_products": 0, "stores_with_products": 0, "stores_failed": 0}

    for store in stores:
        store_result = {"id": store.id, "name": store.name, "products": 0, "error": None}
        logger.info(f"Scraping {store.name} ({store.url})...")
        try:
            products = await scrape_store_bestsellers(store.url)
            if products:
                update_products_in_db(db, store, products)
                store_result["products"] = len(products)
                results["total_products"] += len(products)
                results["stores_with_products"] += 1
                logger.info(f"  ✓ {len(products)} products for {store.name}")
            else:
                store_result["error"] = "no products parsed"
                results["stores_failed"] += 1
                logger.warning(f"  ✗ No products found for {store.name}")
        except Exception as e:
            store_result["error"] = str(e)
            results["stores_failed"] += 1
            logger.error(f"  ✗ Failed to scrape {store.name}: {e}")

        results["stores"].append(store_result)
        await asyncio.sleep(5 + random.uniform(0, 3))

    try:
        cleanup_old_history(db)
    except Exception as e:
        logger.error(f"History cleanup failed: {e}")

    logger.info(
        f"Scrape complete: {results['total_products']} products across "
        f"{results['stores_with_products']}/{len(stores)} stores"
    )
    return results
