import httpx
import json
import logging
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from models import Store, Product, PositionHistory
from classifier import classify_products_batch

logger = logging.getLogger(__name__)

HISTORY_RETENTION_DAYS = 30
SHOPIFY_PAGE_SIZE = 250


async def scrape_store_bestsellers(store_url: str, limit: int = 250) -> list:
    """Scrape top bestsellers from a Shopify store via paginated JSON API.

    Uses /collections/all/products.json?sort_by=best-selling and walks pages
    until `limit` is reached or the store returns an empty page. Position is
    the global bestseller rank (page 1 first item = position 1).
    """
    products = []
    base_url = store_url.rstrip("/")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=45.0, follow_redirects=True) as client:
            page = 1
            while len(products) < limit:
                url = (
                    f"{base_url}/collections/all/products.json"
                    f"?sort_by=best-selling&limit={SHOPIFY_PAGE_SIZE}&page={page}"
                )
                response = await client.get(url, headers=headers)
                if response.status_code != 200:
                    logger.warning(
                        f"Non-200 response for {base_url} page={page}: {response.status_code}"
                    )
                    break

                try:
                    data = response.json()
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parse error for {base_url} page={page}: {e}")
                    break

                page_products = data.get("products", [])
                if not page_products:
                    break

                for product in page_products:
                    if len(products) >= limit:
                        break
                    image_url = ""
                    if product.get("images"):
                        image_url = product["images"][0].get("src", "")
                    price = ""
                    if product.get("variants"):
                        price = product["variants"][0].get("price", "")
                    products.append({
                        "shopify_id": str(product.get("id", "")),
                        "title": product.get("title", "Unknown"),
                        "handle": product.get("handle", ""),
                        "image_url": image_url,
                        "price": price,
                        "vendor": product.get("vendor", ""),
                        "product_type": product.get("product_type", ""),
                        "product_url": f"{base_url}/products/{product.get('handle', '')}",
                        "position": len(products) + 1,
                    })

                if len(page_products) < SHOPIFY_PAGE_SIZE:
                    break
                page += 1

    except Exception as e:
        logger.error(f"Error scraping {base_url}: {e}")

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
            # Snapshot prior state BEFORE we mutate it.
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
