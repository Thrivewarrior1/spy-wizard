import httpx
import json
import logging
import re
from datetime import datetime
from bs4 import BeautifulSoup
from sqlalchemy.orm import Session
from models import Store, Product, PositionHistory
from classifier import classify_products_batch

logger = logging.getLogger(__name__)


async def scrape_store_bestsellers(store_url: str, limit: int = 250) -> list:
    """Scrape bestsellers by reading the actual HTML page at /collections/all?sort_by=best-selling.

    The Shopify JSON API (/products.json) IGNORES sort_by parameter and returns alphabetical order.
    So we MUST scrape the HTML page to get the TRUE bestseller order.

    Strategy:
    1. Fetch HTML pages of /collections/all?sort_by=best-selling (paginated)
    2. Extract product handles in the EXACT order they appear on the page
    3. Fetch full product details from /products/{handle}.json for each
    """
    products = []
    base_url = store_url.rstrip("/")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml",
    }

    try:
        async with httpx.AsyncClient(timeout=45.0, follow_redirects=True) as client:
            page = 1
            seen_handles = set()

            while len(products) < limit:
                # Fetch the actual bestseller HTML page
                url = f"{base_url}/collections/all?sort_by=best-selling&page={page}"
                response = await client.get(url, headers=headers)

                if response.status_code != 200:
                    logger.warning(f"HTTP {response.status_code} for {url}")
                    break

                html = response.text
                soup = BeautifulSoup(html, "html.parser")

                # Extract product handles from links in the order they appear
                page_handles = []
                for a_tag in soup.find_all("a", href=True):
                    href = a_tag["href"]
                    match = re.match(r"^/products/([a-zA-Z0-9\-_]+)", href)
                    if match:
                        handle = match.group(1)
                        if handle not in seen_handles:
                            seen_handles.add(handle)
                            page_handles.append(handle)

                if not page_handles:
                    # No more products on this page
                    break

                # Fetch details for each product via individual JSON endpoint
                for handle in page_handles:
                    if len(products) >= limit:
                        break

                    position = len(products) + 1  # Position = order on bestseller page

                    try:
                        # Get product details from JSON
                        detail_url = f"{base_url}/products/{handle}.json"
                        detail_resp = await client.get(detail_url, headers={
                            "User-Agent": headers["User-Agent"],
                            "Accept": "application/json"
                        })

                        if detail_resp.status_code == 200:
                            pdata = detail_resp.json().get("product", {})

                            image_url = ""
                            if pdata.get("images"):
                                image_url = pdata["images"][0].get("src", "")

                            price = ""
                            if pdata.get("variants"):
                                price = pdata["variants"][0].get("price", "")

                            products.append({
                                "shopify_id": str(pdata.get("id", "")),
                                "title": pdata.get("title", handle),
                                "handle": handle,
                                "image_url": image_url,
                                "price": price,
                                "vendor": pdata.get("vendor", ""),
                                "product_type": pdata.get("product_type", ""),
                                "product_url": f"{base_url}/products/{handle}",
                                "position": position,
                            })
                        else:
                            # If individual JSON fails, still record with handle as title
                            products.append({
                                "shopify_id": "",
                                "title": handle.replace("-", " ").title(),
                                "handle": handle,
                                "image_url": "",
                                "price": "",
                                "vendor": "",
                                "product_type": "",
                                "product_url": f"{base_url}/products/{handle}",
                                "position": position,
                            })

                    except Exception as e:
                        logger.debug(f"Failed to fetch {handle}: {e}")
                        continue

                page += 1

                # Safety: don't fetch more than 10 pages
                if page > 10:
                    break

            if products:
                logger.info(f"Scraped {len(products)} bestsellers from {base_url} (HTML method)")

    except Exception as e:
        logger.error(f"Error scraping {base_url}: {e}")

    return products


def update_products_in_db(db: Session, store: Store, scraped_products: list):
    """Update products in database and track position changes for Hero/Villain/Normal."""
    existing_products = {}
    for p in store.products:
        if p.shopify_id:
            existing_products[p.shopify_id] = p

    now = datetime.utcnow()

    for product_data in scraped_products:
        shopify_id = product_data["shopify_id"]
        new_position = product_data["position"]

        if shopify_id and shopify_id in existing_products:
            product = existing_products[shopify_id]
            old_position = product.current_position

            # Save snapshot of previous state BEFORE updating
            prev_prev = product.previous_position

            product.previous_position = old_position
            product.current_position = new_position

            product.title = product_data["title"]
            product.image_url = product_data["image_url"]
            product.price = product_data["price"]
            product.product_url = product_data["product_url"]
            product.vendor = product_data.get("vendor", "")
            product.product_type = product_data.get("product_type", "")
            product.ai_tags = product_data.get("ai_tags", product.ai_tags or "")
            product.is_fashion = product_data.get("is_fashion", product.is_fashion)
            product.last_scraped = now

            # Hero/Villain ONLY when we have reliable previous data
            # old_position > 0 means the product had a position from a prior scrape
            # prev_prev > 0 means there was even a scrape before THAT one
            if old_position > 0 and prev_prev > 0:
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

        # Record position history for 30-day tracking
        history = PositionHistory(
            product_id=product.id,
            position=new_position,
            date=now,
        )
        db.add(history)

    db.commit()
    logger.info(f"Updated {len(scraped_products)} products for {store.name}")


def cleanup_old_history(db: Session, days: int = 30):
    """Remove position history older than N days."""
    from datetime import timedelta
    cutoff = datetime.utcnow() - timedelta(days=days)
    deleted = db.query(PositionHistory).filter(PositionHistory.date < cutoff).delete()
    db.commit()
    if deleted:
        logger.info(f"Cleaned up {deleted} old history records (>{days} days)")


def reset_all_labels(db: Session):
    """Reset all product labels to 'normal'."""
    count = db.query(Product).update({"label": "normal", "previous_position": 0})
    db.commit()
    logger.info(f"Reset {count} product labels to 'normal'")


async def scrape_all_stores(db: Session):
    """Scrape all stores and update the database."""
    stores = db.query(Store).all()
    logger.info(f"Starting scrape of {len(stores)} stores...")

    cleanup_old_history(db, days=30)

    for store in stores:
        logger.info(f"Scraping {store.name} ({store.url})...")
        try:
            products = await scrape_store_bestsellers(store.url)
            if products:
                products = await classify_products_batch(products)
                update_products_in_db(db, store, products)
                fashion_count = sum(1 for p in products if p.get("is_fashion", True))
                logger.info(f"  ✓ {len(products)} products ({fashion_count} fashion) for {store.name}")
            else:
                logger.warning(f"  ✗ No products found for {store.name}")
        except Exception as e:
            logger.error(f"  ✗ Failed to scrape {store.name}: {e}")

    logger.info("Scrape complete.")
