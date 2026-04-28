import httpx
import json
import logging
from datetime import datetime
from sqlalchemy.orm import Session
from models import Store, Product, PositionHistory, LabelEnum

logger = logging.getLogger(__name__)

async def scrape_store_bestsellers(store_url: str, limit: int = 30) -> list:
    """Scrape top bestsellers from a Shopify store using their JSON API."""
    products = []
    base_url = store_url.rstrip("/")

    try:
        # Shopify stores expose products as JSON
        url = f"{base_url}/collections/all/products.json?sort_by=best-selling&limit={limit}"

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json",
        }

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(url, headers=headers)

            if response.status_code == 200:
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
            else:
                logger.warning(f"Failed to scrape {base_url}: HTTP {response.status_code}")
                # Fallback: try scraping the HTML page
                products = await scrape_store_html(base_url, limit, client, headers)

    except Exception as e:
        logger.error(f"Error scraping {base_url}: {e}")

    return products


async def scrape_store_html(base_url: str, limit: int, client: httpx.AsyncClient, headers: dict) -> list:
    """Fallback: scrape bestsellers from HTML if JSON API fails."""
    products = []
    try:
        url = f"{base_url}/collections/all?sort_by=best-selling"
        response = await client.get(url, headers=headers)

        if response.status_code == 200:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")

            # Try to find product data in script tags
            for script in soup.find_all("script", type="application/json"):
                try:
                    data = json.loads(script.string)
                    # Look for product arrays in various Shopify theme formats
                    if isinstance(data, dict):
                        for key, val in data.items():
                            if isinstance(val, list) and len(val) > 0:
                                if isinstance(val[0], dict) and "title" in val[0]:
                                    for idx, p in enumerate(val[:limit]):
                                        products.append({
                                            "shopify_id": str(p.get("id", "")),
                                            "title": p.get("title", "Unknown"),
                                            "handle": p.get("handle", ""),
                                            "image_url": p.get("featured_image", ""),
                                            "price": str(p.get("price", "")),
                                            "vendor": p.get("vendor", ""),
                                            "product_type": p.get("type", ""),
                                            "product_url": f"{base_url}/products/{p.get('handle', '')}",
                                            "position": idx + 1,
                                        })
                                    if products:
                                        return products[:limit]
                except (json.JSONDecodeError, TypeError):
                    continue
    except Exception as e:
        logger.error(f"HTML scrape fallback failed for {base_url}: {e}")

    return products


def update_products_in_db(db: Session, store: Store, scraped_products: list):
    """Update products in database and calculate position changes."""
    existing_products = {p.shopify_id: p for p in store.products if p.shopify_id}

    # Track which products we've seen
    seen_ids = set()

    for product_data in scraped_products:
        shopify_id = product_data["shopify_id"]
        seen_ids.add(shopify_id)

        if shopify_id in existing_products:
            # Update existing product
            product = existing_products[shopify_id]
            product.previous_position = product.current_position
            product.current_position = product_data["position"]
            product.title = product_data["title"]
            product.image_url = product_data["image_url"]
            product.price = product_data["price"]
            product.product_url = product_data["product_url"]
            product.last_scraped = datetime.utcnow()

            # Calculate label
            if product.previous_position > 0:
                if product.current_position < product.previous_position:
                    product.label = LabelEnum.HERO.value
                elif product.current_position > product.previous_position:
                    product.label = LabelEnum.VILLAIN.value
                else:
                    product.label = LabelEnum.NORMAL.value

        else:
            # New product
            product = Product(
                store_id=store.id,
                shopify_id=shopify_id,
                title=product_data["title"],
                handle=product_data["handle"],
                image_url=product_data["image_url"],
                price=product_data["price"],
                vendor=product_data["vendor"],
                product_type=product_data["product_type"],
                product_url=product_data["product_url"],
                current_position=product_data["position"],
                previous_position=0,
                label=LabelEnum.NORMAL.value,
                last_scraped=datetime.utcnow(),
            )
            db.add(product)
            db.flush()

        # Add position history
        history = PositionHistory(
            product_id=product.id,
            position=product_data["position"],
            date=datetime.utcnow(),
        )
        db.add(history)

    db.commit()


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
                logger.info(f"  Updated {len(products)} products for {store.name}")
            else:
                logger.warning(f"  No products found for {store.name}")
        except Exception as e:
            logger.error(f"  Failed to scrape {store.name}: {e}")

    logger.info("Scrape complete.")
