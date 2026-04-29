"""Shopify bestseller scraper.

Strategy:
  1. Fetch /collections/all?sort_by=best-selling HTML pages (the JSON API
     ignores sort_by and returns alphabetical order — only the rendered
     HTML reflects true bestseller ranking).
  2. Extract ALL product info (title, handle, image, price) directly from
     the HTML. ZERO individual /products/{handle}.json calls — Shopify
     blocks the high-volume per-product fetches with 429.
  3. Send EVERY parsed product to Gemini for fashion classification
     (clothing, shoes, bags only — see classifier.py).
  4. Keep only is_fashion=True products. They get sequential positions
     1..N in their original display order. Non-fashion products are
     dropped entirely and never enter the database.
  5. Hero/villain labels are only assigned once a product has >= 2 prior
     PositionHistory rows (i.e. it has been observed at least twice before
     this scrape).
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

PRODUCT_HREF_RE = re.compile(r"^/?products/([a-zA-Z0-9][a-zA-Z0-9\-_]*?)(?:\?|#|/|$)")


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


def _extract_image_url(img) -> str:
    """Pull the best-available image URL off a Shopify product card <img>."""
    if not img:
        return ""

    candidates = [
        img.get("src"),
        img.get("data-src"),
        img.get("data-original"),
    ]

    srcset = img.get("srcset") or img.get("data-srcset") or ""
    if srcset:
        first = srcset.split(",")[0].strip().split(" ")[0]
        candidates.append(first)

    for c in candidates:
        if c:
            url = c.strip()
            if url.startswith("//"):
                url = "https:" + url
            if url.startswith("http") or url.startswith("/"):
                return url

    return ""


def _extract_products_from_html(soup: BeautifulSoup, base_url: str, seen: set) -> list:
    """Parse a Shopify collection HTML page and return product dicts in
    display order. Pulls title / handle / image / price directly from the
    rendered card markup — no follow-up API calls.
    """
    products = []

    main = soup.find("main") or soup.find(id="MainContent") or soup.body or soup

    for a_tag in main.find_all("a", href=True):
        href = a_tag["href"]
        path = href.split("?", 1)[0].split("#", 1)[0]
        match = PRODUCT_HREF_RE.match(path)
        if not match:
            continue

        handle = match.group(1).lower()
        if handle in seen:
            continue
        seen.add(handle)

        # Walk up to find the card container that holds the link plus
        # other product info (image / price / title).
        card = a_tag
        for _ in range(5):
            parent = card.parent
            if not parent:
                break
            card = parent
            if parent.name in ("div", "li", "article", "section"):
                if card.find("img") or card.find(class_=re.compile(r"price|title|name", re.I)):
                    break

        # Title: prefer a labeled element, fall back to the link text.
        title = ""
        title_el = card.find(
            ["h2", "h3", "h4", "h5", "span", "p", "div"],
            class_=re.compile(r"title|name|product", re.I),
        )
        if title_el:
            title = title_el.get_text(strip=True)
        if not title:
            title = a_tag.get_text(strip=True)
        if not title:
            title = handle.replace("-", " ").title()

        # Image
        image_url = _extract_image_url(card.find("img"))

        # Price
        price = ""
        price_el = card.find(class_=re.compile(r"price|money|amount", re.I))
        if price_el:
            price_text = price_el.get_text(" ", strip=True)
            price_match = re.search(r"[\d]+(?:[.,]\d+)?", price_text)
            if price_match:
                price = price_match.group()

        if not title or len(title) < 2:
            continue

        products.append({
            "shopify_id": handle,  # No JSON fetch, so handle doubles as the stable ID.
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

    Issues only 1 HTTP request per collection page (typically 3-5 pages to
    reach 100 products). No per-product API calls.

    Returns a list of fashion-only products (clothing/shoes/bags) in their
    bestseller display order, each with sequential `position` 1..N.
    Non-fashion items are dropped before positions are assigned.
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

    # Gemini classifies every product. Only is_fashion=True items survive.
    classify_products_batch(raw_products)

    fashion_products = []
    for details in raw_products:
        if not details.get("is_fashion"):
            continue
        details["position"] = len(fashion_products) + 1
        fashion_products.append(details)

    logger.info(
        f"{base_url}: scraped {len(raw_products)} products, "
        f"{len(fashion_products)} are fashion (positions 1..{len(fashion_products)})"
    )
    return fashion_products


def update_products_in_db(db: Session, store: Store, scraped_products: list):
    """Update fashion products and assign hero/villain/normal based on position.

    Label rule: hero/villain only when the product already has >= 2 prior
    PositionHistory rows (i.e. it has been observed in at least two earlier
    scrapes). The first two scrapes of a product always produce 'normal'.

    Products from `scraped_products` are already filtered to is_fashion=True
    by the scraper, so non-fashion items never reach the database.
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
            product.is_fashion = True
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
                is_fashion=True,
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
    logger.info(f"Updated {len(scraped_products)} fashion products for {store.name}")


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


async def scrape_all_stores(db: Session):
    """Scrape all stores, update DB, and prune old history."""
    stores = db.query(Store).all()
    logger.info(f"Starting scrape of {len(stores)} stores...")

    for store in stores:
        logger.info(f"Scraping {store.name} ({store.url})...")
        try:
            products = await scrape_store_bestsellers(store.url)
            if products:
                update_products_in_db(db, store, products)
                logger.info(f"  ✓ {len(products)} fashion products for {store.name}")
            else:
                logger.warning(f"  ✗ No fashion products found for {store.name}")
        except Exception as e:
            logger.error(f"  ✗ Failed to scrape {store.name}: {e}")

        await asyncio.sleep(5 + random.uniform(0, 3))

    try:
        cleanup_old_history(db)
    except Exception as e:
        logger.error(f"History cleanup failed: {e}")

    logger.info("Scrape complete.")
