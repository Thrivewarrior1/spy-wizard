"""Shopify bestseller scraper — fashion-only feed.

Strategy:
  1. Fetch /collections/all?sort_by=best-selling HTML pages page by page.
     The Shopify JSON endpoint ignores sort_by; only rendered HTML reflects
     true bestseller ranking.
  2. Group product links inside <main> by handle. The DOM-order index of
     the FIRST link to a handle is its raw bestseller rank on that page.
  3. Classify each newly-seen product with Gemini in batches:
       - is_fashion=True  → keep, append to the fashion feed
       - is_fashion=False → skip (gift cards, shipping protection, jewelry,
         home decor, supplements, etc. — never enters the feed)
  4. Keep paginating until we have 100 confirmed fashion products or we run
     out of pages. The 100 cap is POST-filter, so junk near the top of a
     bestseller list does not reduce the feed below 100.
  5. Position assignment: rank 1 = first fashion product encountered in
     the HTML, rank 2 = second, etc. Non-fashion items are excluded
     entirely (NOT ranked-then-hidden), so the displayed rank equals the
     fashion-only ordering of that store.
  6. Classifier failures are surfaced — they do NOT silently drop products.
     The scrape returns a list of error strings the API can show.
  7. Hero/villain labels only assigned once a product has >= 2 prior
     PositionHistory rows, so the first two scrapes always produce 'normal'.
"""
import asyncio
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
MAX_PAGES = 12
TARGET_FASHION = 100

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


def _build_headers() -> dict:
    return {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
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


def _clean_title(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    if _VARIANT_JSON_RE.match(text):
        return ""
    if text.startswith("[{") or text.startswith("{") or '"id":' in text:
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
    for el in card.find_all(class_=re.compile(r"price|money|amount", re.I)):
        text = el.get_text(" ", strip=True)
        match = re.search(r"\d+(?:[.,]\d+)?", text)
        if match:
            return match.group()
    return ""


def _extract_products_from_html(soup: BeautifulSoup, base_url: str, seen: set) -> list:
    """Parse a Shopify collection HTML page. Yields products in DOM order,
    deduped by handle (across pages too — `seen` is shared)."""
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


async def scrape_store_bestsellers(store_url: str, target_fashion: int = TARGET_FASHION):
    """Scrape a store's best-selling collection until we have `target_fashion`
    confirmed-fashion products, or we run out of pages.

    Returns (fashion_products, errors):
      - fashion_products: up to `target_fashion` products with sequential
        positions 1..N reflecting their fashion-only bestseller order.
      - errors: list of human-readable strings describing any non-fatal
        problems (Gemini failures, missing API key, HTTP errors per page).
        Empty list if everything was clean.
    """
    base_url = store_url.rstrip("/")
    fashion: list = []
    seen: set = set()
    errors: list = []

    if not os.getenv("GEMINI_API_KEY"):
        errors.append(
            "GEMINI_API_KEY is not set on the server — fashion classification "
            "is required for the feed but cannot run. Set the env var in Railway."
        )
        return fashion, errors

    try:
        async with httpx.AsyncClient(
            timeout=30.0, follow_redirects=True, headers=_build_headers()
        ) as client:
            page = 1
            while len(fashion) < target_fashion and page <= MAX_PAGES:
                url = f"{base_url}/collections/all?sort_by=best-selling&page={page}"
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
                page_products = _extract_products_from_html(soup, base_url, seen)
                logger.info(
                    f"{base_url} page {page}: parsed {len(page_products)} new products "
                    f"(running fashion total {len(fashion)})"
                )

                if not page_products:
                    break

                # Classify this page's batch with Gemini. Failures bubble up
                # through `errors` rather than being silently swallowed.
                ok, classifier_errors = _classify_or_fail(page_products)
                errors.extend(classifier_errors)
                if not ok:
                    # Hard fail — without classification we cannot meet the
                    # fashion-only requirement, so stop rather than corrupt
                    # the feed with unclassified items.
                    break

                for p in page_products:
                    if not p.get("is_fashion"):
                        continue
                    p["position"] = len(fashion) + 1
                    fashion.append(p)
                    if len(fashion) >= target_fashion:
                        break

                if len(fashion) >= target_fashion:
                    break

                page += 1
                await asyncio.sleep(2 + random.uniform(0, 2))
    except Exception as e:
        errors.append(f"unexpected error: {e}")
        logger.exception(f"Error scraping {base_url}")
        return fashion, errors

    if not fashion:
        # If we have errors, the caller surfaces those. Otherwise note that
        # the page yielded no fashion products at all (rare but possible).
        if not errors:
            errors.append("no fashion products parsed from any page")

    logger.info(
        f"{base_url}: returning {len(fashion)} fashion products "
        f"(positions 1..{len(fashion)}, errors={len(errors)})"
    )
    return fashion, errors


def _classify_or_fail(batch: list):
    """Run Gemini classification on a batch. Returns (ok, errors)."""
    if not batch:
        return True, []
    try:
        before_fashion = sum(1 for p in batch if p.get("is_fashion"))
        # Pre-set defaults so we can detect items Gemini didn't return for.
        for p in batch:
            p.setdefault("is_fashion", None)
            p.setdefault("ai_tags", "")
        classify_products_batch(batch)
    except Exception as e:
        return False, [f"Gemini exception: {e}"]

    missing = [p["handle"] for p in batch if p.get("is_fashion") is None]
    if missing:
        # Surface what Gemini didn't classify rather than silently dropping.
        sample = ", ".join(missing[:5])
        more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        return False, [f"Gemini did not classify {len(missing)} items: {sample}{more}"]
    _ = before_fashion
    return True, []


async def debug_fetch(store_url: str) -> dict:
    """Diagnostic helper: report what the server actually receives when
    fetching the store's best-seller page. Useful when production says
    'no products parsed' but local testing succeeds — usually means the
    upstream is serving a different page (Cloudflare challenge, etc.)."""
    base_url = store_url.rstrip("/")
    url = f"{base_url}/collections/all?sort_by=best-selling&page=1"
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


def update_products_in_db(db: Session, store: Store, scraped_products: list):
    """Persist scraped fashion products. All inputs are already is_fashion=True
    (the scraper filters before this). Hero/villain assignment requires a
    product to have >= 2 prior PositionHistory rows."""
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
        "stores_with_products": 0,
        "stores_failed": 0,
    }

    for store in stores:
        store_result = {"id": store.id, "name": store.name, "products": 0, "errors": []}
        logger.info(f"Scraping {store.name} ({store.url})...")
        try:
            products, errors = await scrape_store_bestsellers(store.url)
            store_result["errors"] = errors
            if products:
                update_products_in_db(db, store, products)
                store_result["products"] = len(products)
                results["total_products"] += len(products)
                results["stores_with_products"] += 1
                logger.info(f"  ✓ {len(products)} fashion products for {store.name}")
            else:
                results["stores_failed"] += 1
                logger.warning(
                    f"  ✗ No fashion products for {store.name} — errors: {errors}"
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
        f"Scrape complete: {results['total_products']} products across "
        f"{results['stores_with_products']}/{len(stores)} stores"
    )
    return results
