import asyncio
import httpx
import json
import logging
import re
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from sqlalchemy.orm import Session
from models import Store, Product, PositionHistory
from classifier import classify_products_batch

logger = logging.getLogger(__name__)

HISTORY_RETENTION_DAYS = 30
MAX_PAGES = 10
PRODUCT_FETCH_CONCURRENCY = 8

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

PRODUCT_HREF_RE = re.compile(r"^/products/([a-zA-Z0-9][a-zA-Z0-9\-_]*)")


def _extract_handles_from_html(html: str) -> list:
    """Parse a Shopify collection HTML page and return product handles in
    the exact order they first appear in <a href="/products/..."> links.
    Duplicates are removed while preserving first-seen order.
    """
    soup = BeautifulSoup(html, "html.parser")
    handles = []
    seen = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Strip query string / fragment so /products/foo?variant=... still matches.
        path = href.split("?", 1)[0].split("#", 1)[0]
        match = PRODUCT_HREF_RE.match(path)
        if not match:
            continue
        handle = match.group(1)
        if handle in seen:
            continue
        seen.add(handle)
        handles.append(handle)

    return handles


async def _fetch_product_json(
    client: httpx.AsyncClient,
    base_url: str,
    handle: str,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    """Fetch a single product's full details from /products/{handle}.json."""
    url = f"{base_url}/products/{handle}.json"
    async with semaphore:
        try:
            response = await client.get(url)
            if response.status_code != 200:
                logger.warning(
                    f"Non-200 fetching product {handle} from {base_url}: {response.status_code}"
                )
                return None
            data = response.json()
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to fetch product {handle} from {base_url}: {e}")
            return None

    product = data.get("product")
    if not product:
        return None

    image_url = ""
    images = product.get("images") or []
    if images:
        image_url = images[0].get("src", "") or ""
    if not image_url and product.get("image"):
        image_url = product["image"].get("src", "") or ""

    price = ""
    variants = product.get("variants") or []
    if variants:
        price = variants[0].get("price", "") or ""

    return {
        "shopify_id": str(product.get("id", "")),
        "title": product.get("title", "Unknown"),
        "handle": product.get("handle", handle),
        "image_url": image_url,
        "price": price,
        "vendor": product.get("vendor", "") or "",
        "product_type": product.get("product_type", "") or "",
        "product_url": f"{base_url}/products/{handle}",
    }


async def scrape_store_bestsellers(store_url: str, max_pages: int = MAX_PAGES) -> list:
    """Scrape bestsellers from a Shopify store by parsing the actual HTML
    bestseller page (the JSON API ignores `sort_by` and returns alphabetical
    order — only the rendered HTML reflects true bestseller ranking).

    Walks /collections/all?sort_by=best-selling&page={N} from page 1 until an
    empty page is hit or `max_pages` is reached. Position is the global order
    that handles first appear in the HTML across pages (page 1 first link =
    position 1). Then fetches /products/{handle}.json for each handle to get
    titles, images, prices, etc.
    """
    base_url = store_url.rstrip("/")
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    ordered_handles: list = []
    seen_handles: set = set()

    try:
        async with httpx.AsyncClient(timeout=45.0, follow_redirects=True, headers=headers) as client:
            for page in range(1, max_pages + 1):
                url = f"{base_url}/collections/all?sort_by=best-selling&page={page}"
                try:
                    response = await client.get(url)
                except httpx.HTTPError as e:
                    logger.warning(f"HTTP error for {url}: {e}")
                    break

                if response.status_code != 200:
                    logger.warning(
                        f"Non-200 response for {base_url} page={page}: {response.status_code}"
                    )
                    break

                page_handles = _extract_handles_from_html(response.text)
                if not page_handles:
                    break

                new_count = 0
                for handle in page_handles:
                    if handle in seen_handles:
                        continue
                    seen_handles.add(handle)
                    ordered_handles.append(handle)
                    new_count += 1

                # Shopify returns the same products on overflow pages — stop
                # if a page contributed nothing new.
                if new_count == 0:
                    break

            if not ordered_handles:
                logger.warning(f"No product handles parsed from {base_url}")
                return []

            json_headers = {"Accept": "application/json"}
            client.headers.update(json_headers)

            semaphore = asyncio.Semaphore(PRODUCT_FETCH_CONCURRENCY)
            tasks = [
                _fetch_product_json(client, base_url, handle, semaphore)
                for handle in ordered_handles
            ]
            fetched = await asyncio.gather(*tasks)

    except Exception as e:
        logger.error(f"Error scraping {base_url}: {e}")
        return []

    products = []
    for handle, details in zip(ordered_handles, fetched):
        if not details or not details.get("shopify_id"):
            continue
        details["position"] = len(products) + 1
        products.append(details)

    return products


def update_products_in_db(db: Session, store: Store, scraped_products: list):
    """Update products and assign hero/villain/normal based on position movement.

    Label rule: hero/villain only when we have 2+ prior data points
    (the existing previous_position > 0 AND the existing current_position > 0).
    First scrape and second scrape always produce "normal".
    """
    existing_products = {p.shopify_id: p for p in store.products if p.shopify_id}

    now = datetime.utcnow()

    classify_products_batch(scraped_products)

    for product_data in scraped_products:
        shopify_id = product_data["shopify_id"]
        new_position = product_data["position"]

        if shopify_id in existing_products:
            product = existing_products[shopify_id]
            prev_position_before = product.previous_position or 0
            old_position = product.current_position or 0

            product.previous_position = old_position
            product.current_position = new_position

            product.title = product_data["title"]
            product.image_url = product_data["image_url"]
            product.price = product_data["price"]
            product.product_url = product_data["product_url"]
            product.vendor = product_data.get("vendor", "")
            product.product_type = product_data.get("product_type", "")
            product.ai_tags = product_data.get("ai_tags", "")
            product.is_fashion = product_data.get("is_fashion", True)
            product.last_scraped = now

            if prev_position_before > 0 and old_position > 0:
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
                is_fashion=product_data.get("is_fashion", True),
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
    """Reset every product's label to 'normal' and clear previous_position tracking.

    After reset, the next two scrapes will produce only 'normal' labels (since
    the 2+ data point rule needs to rebuild). Returns number of products reset.
    """
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
                logger.info(f"  ✓ {len(products)} products for {store.name}")
            else:
                logger.warning(f"  ✗ No products found for {store.name}")
        except Exception as e:
            logger.error(f"  ✗ Failed to scrape {store.name}: {e}")

    try:
        cleanup_old_history(db)
    except Exception as e:
        logger.error(f"History cleanup failed: {e}")

    logger.info("Scrape complete.")
