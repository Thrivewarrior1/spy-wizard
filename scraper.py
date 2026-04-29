import httpx
import json
import logging
from datetime import datetime
from sqlalchemy.orm import Session
from models import Store, Product, PositionHistory
from classifier import classify_products_batch

logger = logging.getLogger(__name__)

async def scrape_store_bestsellers(store_url: str, limit: int = 30) -> list:
    """Scrape top bestsellers from a Shopify store using their JSON API."""
    products = []
    base_url = store_url.rstrip("/")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            # Try JSON API first
            url = f"{base_url}/collections/all/products.json?sort_by=best-selling&limit={limit}"
            response = await client.get(url, headers=headers)

            if response.status_code == 200:
                try:
                    data = response.json()
                    for idx, product in enumerate(data.get("products", [])[:limit]):
                        image_url = ""
                        if product.get("images") and len(product["images"]) > 0:
                            image_url = product["images"][0].get("src", "")
                        price = ""
                        if product.get("variants") and len(product["variants"]) > 0:
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
                            "position": idx + 1,
                        })
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"JSON parse error for {base_url}: {e}")

            if not products:
                # Fallback: try page-based JSON
                url2 = f"{base_url}/products.json?limit={limit}"
                resp2 = await client.get(url2, headers=headers)
                if resp2.status_code == 200:
                    try:
                        data = resp2.json()
                        for idx, product in enumerate(data.get("products", [])[:limit]):
                            image_url = product.get("images", [{}])[0].get("src", "") if product.get("images") else ""
                            price = product.get("variants", [{}])[0].get("price", "") if product.get("variants") else ""
                            products.append({
                                "shopify_id": str(product.get("id", "")),
                                "title": product.get("title", "Unknown"),
                                "handle": product.get("handle", ""),
                                "image_url": image_url,
                                "price": price,
                                "vendor": product.get("vendor", ""),
                                "product_type": product.get("product_type", ""),
                                "product_url": f"{base_url}/products/{product.get('handle', '')}",
                                "position": idx + 1,
                            })
                    except (json.JSONDecodeError, KeyError):
                        pass

    except Exception as e:
        logger.error(f"Error scraping {base_url}: {e}")

    return products


def update_products_in_db(db: Session, store: Store, scraped_products: list):
    """Update products in database and CORRECTLY calculate position changes (Hero/Villain/Normal)."""
    # Get existing products indexed by shopify_id
    existing_products = {}
    for p in store.products:
        if p.shopify_id:
            existing_products[p.shopify_id] = p

    now = datetime.utcnow()

    # Classify products with AI (or keyword fallback) before persisting
    classify_products_batch(scraped_products)

    for product_data in scraped_products:
        shopify_id = product_data["shopify_id"]
        new_position = product_data["position"]

        if shopify_id in existing_products:
            product = existing_products[shopify_id]
            old_position = product.current_position

            # Store old position as previous
            product.previous_position = old_position
            # Update to new position
            product.current_position = new_position

            # Update other fields
            product.title = product_data["title"]
            product.image_url = product_data["image_url"]
            product.price = product_data["price"]
            product.product_url = product_data["product_url"]
            product.vendor = product_data.get("vendor", "")
            product.product_type = product_data.get("product_type", "")
            product.ai_tags = product_data.get("ai_tags", "")
            product.is_fashion = product_data.get("is_fashion", True)
            product.last_scraped = now

            # CORRECTLY determine label:
            # Hero = moved UP in bestseller list (lower position number = higher rank)
            # Villain = moved DOWN in bestseller list (higher position number = lower rank)
            # Normal = same position
            if old_position > 0:  # Only compare if we have a previous position
                if new_position < old_position:
                    product.label = "hero"  # Moved UP (e.g., from #10 to #5)
                elif new_position > old_position:
                    product.label = "villain"  # Moved DOWN (e.g., from #5 to #10)
                else:
                    product.label = "normal"  # Same position
            # If first time tracking (old_position == 0), keep as normal

        else:
            # Brand new product - mark as normal
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

        # Record position history
        history = PositionHistory(
            product_id=product.id,
            position=new_position,
            date=now,
        )
        db.add(history)

    db.commit()
    logger.info(f"Updated {len(scraped_products)} products for {store.name}")


async def scrape_all_stores(db: Session):
    """Scrape all stores and update the database."""
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

    logger.info("Scrape complete.")
