"""Gemini-powered fashion classifier (BEST-EFFORT enrichment, NOT a feed gate).

Gemini reads a product title/vendor and returns:
  - is_fashion: bool (clothing/shoes/bags vs. obvious junk like gift cards
                 and shipping protection)
  - ai_tags: short English keyword list for multi-language search

This module is purely additive. Products arrive with is_fashion=True already
set by the scraper. We only FLIP to False on items Gemini explicitly marks
non-fashion. Missing API key, batch failure, missing item in the response —
all of those leave is_fashion alone, so a Gemini outage never empties the
bestseller feed.
"""
import os
import json
import logging
import random
import asyncio
import httpx

logger = logging.getLogger(__name__)

# gemini-2.0-flash was retired for new API keys ("no longer available to new
# users"). gemini-2.5-flash is the current-generation drop-in replacement and
# supports the same responseSchema / responseMimeType structured-output flags.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

BATCH_SIZE = 20
# Transient-error HTTP statuses we should retry. 503 is the common
# "model overloaded" path; 429 is quota burst; 500/502/504 are upstream
# blips. 4xx other than 429 means a permanent fault we shouldn't retry.
RETRY_STATUSES = (429, 500, 502, 503, 504)
MAX_RETRIES = 5


async def _classify_batch_with_gemini(batch: list, api_key: str):
    """Classify a single batch in place.

    Async so retry sleeps don't block FastAPI's event loop. Returns
    (ok, error_message). On success error_message is None. On failure
    error_message is a short string describing the underlying cause (HTTP
    status + Gemini error body, JSON parse failure, etc.) so the scraper can
    surface it to the API consumer instead of silently dropping products.
    """
    items = [
        {
            "index": idx,
            "title": p.get("title", ""),
            "vendor": p.get("vendor", ""),
            "product_type": p.get("product_type", ""),
        }
        for idx, p in enumerate(batch)
    ]

    prompt = (
        "You are a STRICT fashion classifier for a clothing bestseller tracker.\n\n"
        "For EACH product below, decide if it is FASHION.\n\n"
        "FASHION = clothing, shoes, or bags ONLY. This includes:\n"
        "  - Clothing: shirts, t-shirts, blouses, tops, sweaters, hoodies, jackets, "
        "coats, dresses, skirts, pants, jeans, shorts, leggings, jumpsuits, "
        "swimwear, lingerie, sleepwear, activewear.\n"
        "  - Shoes: sneakers, boots, heels, sandals, flats, loafers, slippers.\n"
        "  - Bags: handbags, backpacks, totes, clutches, wallets, crossbody bags.\n\n"
        "NOT FASHION (is_fashion=false) — be strict, when in doubt say false:\n"
        "  - Jewelry of any kind (necklaces, earrings, rings, bracelets, watches).\n"
        "  - Accessories that are not clothing/shoes/bags (hats, scarves, belts, "
        "sunglasses, gloves, ties, hair accessories).\n"
        "  - Shipping protection, package protection, route insurance, "
        "delivery guarantees.\n"
        "  - Gift cards, donations, tips, samples.\n"
        "  - Electronics, phone cases, chargers, headphones.\n"
        "  - Home decor, candles, furniture, kitchenware, bedding, rugs, art.\n"
        "  - Beauty, skincare, makeup, perfume, supplements, vitamins, food.\n"
        "  - Toys, books, stationery, pet products.\n\n"
        "Also produce 2-6 short English keyword tags per product describing it "
        "(e.g. 'summer,floral,maxi,dress'), comma-separated, all lowercase. "
        "Translate non-English titles into English keywords.\n\n"
        f"Products (JSON):\n{json.dumps(items, ensure_ascii=False)}\n\n"
        "Respond with a JSON array, one object per product, matching the input indices."
    )

    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "index": {"type": "integer"},
                "is_fashion": {"type": "boolean"},
                "tags": {"type": "string"},
            },
            "required": ["index", "is_fashion", "tags"],
        },
    }

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
            "responseSchema": schema,
        },
    }

    try:
        last_status = None
        last_body = ""
        async with httpx.AsyncClient(timeout=60.0) as client:
            for attempt in range(MAX_RETRIES):
                resp = await client.post(
                    GEMINI_URL,
                    params={"key": api_key},
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                if resp.status_code == 200:
                    break
                last_status = resp.status_code
                last_body = (resp.text or "")[:400].replace("\n", " ")
                if resp.status_code not in RETRY_STATUSES or attempt == MAX_RETRIES - 1:
                    msg = f"HTTP {resp.status_code}: {last_body}"
                    logger.warning("Gemini batch failed — %s", msg)
                    return False, msg
                # Exponential backoff with jitter — Gemini's 503 spikes are
                # usually short, so 4s -> 8s -> 16s -> 32s gives ~60s total
                # which keeps a single page's wall-clock under control.
                wait = (2 ** (attempt + 1)) + random.uniform(0, 2)
                logger.warning(
                    "Gemini %s on attempt %d/%d, retrying in %.1fs",
                    resp.status_code, attempt + 1, MAX_RETRIES, wait,
                )
                await asyncio.sleep(wait)
            else:
                msg = f"HTTP {last_status} after {MAX_RETRIES} retries: {last_body}"
                logger.warning("Gemini batch failed — %s", msg)
                return False, msg
            data = resp.json()

        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as e:
            preview = json.dumps(data)[:400]
            msg = f"unexpected response shape ({e}): {preview}"
            logger.warning("Gemini batch failed — %s", msg)
            return False, msg

        try:
            results = json.loads(text)
        except json.JSONDecodeError as e:
            preview = text[:300].replace("\n", " ")
            msg = f"response not valid JSON ({e}): {preview}"
            logger.warning("Gemini batch failed — %s", msg)
            return False, msg

        for r in results:
            i = r.get("index")
            if not (isinstance(i, int) and 0 <= i < len(batch)):
                continue
            # Only flip is_fashion when Gemini gives us a real boolean.
            if "is_fashion" in r:
                batch[i]["is_fashion"] = bool(r.get("is_fashion"))
            tags = (r.get("tags") or "").strip().lower()
            if tags:
                batch[i]["ai_tags"] = tags

        return True, None

    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        logger.warning("Gemini batch failed — %s", msg)
        return False, msg


async def classify_products_batch(products: list):
    """Enrich products in place with ai_tags and (optionally) is_fashion.

    Async so the per-batch network IO and any retry sleeps cooperate with
    FastAPI's event loop instead of stalling it for tens of seconds at a
    time. Returns a list of error strings — empty when every batch
    succeeded. Errors include the Gemini HTTP body / exception message so
    the caller can surface the real cause (auth, quota, model-not-found)
    rather than a generic "did not classify" message.
    """
    if not products:
        return []

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.info(
            "GEMINI_API_KEY not set — skipping classification for %d products",
            len(products),
        )
        return ["GEMINI_API_KEY not set on server"]

    total = len(products)
    ok = 0
    failed = 0
    errors: list = []

    for start in range(0, total, BATCH_SIZE):
        batch = products[start:start + BATCH_SIZE]
        success, err = await _classify_batch_with_gemini(batch, api_key)
        if success:
            ok += len(batch)
        else:
            failed += len(batch)
            if err:
                errors.append(err)

    fashion_count = sum(1 for p in products if p.get("is_fashion"))
    logger.info(
        "Classified %d products (ok=%d, failed=%d, fashion=%d)",
        total, ok, failed, fashion_count,
    )
    return errors
