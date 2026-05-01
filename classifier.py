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
# When 2.5-flash is overloaded (frequent 503s in 2026) we fall back to the
# lite tier, which sees less traffic and is more than adequate for the
# yes/no fashion classification task we use it for.
# gemini-2.5-flash-lite is the lite tier — higher RPM quota, far less
# 503-overload contention than gemini-2.5-flash, and more than capable of
# the yes/no fashion classification task we use it for. Make it the
# primary so a typical scrape completes in minutes, not 30+ minutes.
GEMINI_PRIMARY = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
GEMINI_FALLBACKS = ["gemini-2.5-flash", "gemini-flash-latest"]


def _gemini_url(model: str) -> str:
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


# 50 products per Gemini call — well under the model's input-token limit
# for short product titles, and roughly 5x fewer requests than BATCH_SIZE=20
# so a full 12-store scrape stays under per-minute RPM quotas.
BATCH_SIZE = 50
# Transient-error HTTP statuses we should retry. 503 is the common
# "model overloaded" path; 429 is quota burst; 500/502/504 are upstream
# blips. 4xx other than 429 means a permanent fault we shouldn't retry.
RETRY_STATUSES = (429, 500, 502, 503, 504)
MAX_RETRIES = 4
# Inter-batch throttle: gentle gap between Gemini calls within a single
# scrape page so we don't burst over the per-minute quota that frequently
# manifests as 503 "model overloaded" on the free tier.
INTER_BATCH_DELAY = 1.5


async def _classify_batch_with_gemini(batch: list, api_key: str, model: str):
    """Classify a single batch in place against `model`.

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
        "You are a STRICT product classifier for a Shopify bestseller tracker.\n"
        "Two parallel feeds use your output:\n"
        "  - Fashion feed (is_fashion=true): clothing, shoes, bags ONLY.\n"
        "  - General feed (is_fashion=false): everything else, grouped by subniche.\n\n"
        "For EACH product below, decide:\n"
        "  1. is_fashion: true ONLY for clothing/shoes/bags. False for jewelry, "
        "accessories that are not clothing/shoes/bags, electronics, home decor, "
        "beauty, supplements, services, gift cards, etc.\n"
        "  2. subniche: ONE label from this fixed list:\n"
        "     - fashion          (clothing/shoes/bags — only when is_fashion=true)\n"
        "     - jewelry          (necklaces, earrings, rings, bracelets, watches)\n"
        "     - accessories      (hats, scarves, belts, sunglasses, gloves, ties)\n"
        "     - electronics      (gadgets, phones, cases, chargers, headphones)\n"
        "     - home             (lamps, candles, furniture, kitchenware, bedding, decor)\n"
        "     - beauty           (skincare, makeup, perfume, fragrances)\n"
        "     - health           (supplements, vitamins, wellness)\n"
        "     - food             (snacks, drinks, edibles)\n"
        "     - toys-books       (toys, games, books, stationery, pet products)\n"
        "     - services         (shipping protection, route insurance, gift cards, donations, "
        "warranties, slidecart upsells, '100% coverage' add-ons)\n"
        "     - other            (anything that doesn't fit above)\n"
        "  3. tags: 2-6 short English keyword tags (comma-separated, lowercase) "
        "describing the product so we can find it from a multi-language search. "
        "Translate non-English titles into English keywords.\n\n"
        "FASHION = clothing, shoes, or bags ONLY:\n"
        "  - Clothing: shirts, t-shirts, blouses, tops, sweaters, hoodies, jackets, "
        "coats, dresses, skirts, pants, jeans, shorts, leggings, jumpsuits, "
        "swimwear, lingerie, sleepwear, activewear.\n"
        "  - Shoes: sneakers, boots, heels, sandals, flats, loafers, slippers.\n"
        "  - Bags: handbags, backpacks, totes, clutches, wallets, crossbody bags.\n\n"
        "When is_fashion=true you MUST set subniche='fashion'. When is_fashion=false "
        "you MUST set subniche to one of the other labels. Be strict.\n\n"
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
                "subniche": {"type": "string"},
                "tags": {"type": "string"},
            },
            "required": ["index", "is_fashion", "subniche", "tags"],
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

    url = _gemini_url(model)
    try:
        last_status = None
        last_body = ""
        async with httpx.AsyncClient(timeout=60.0) as client:
            for attempt in range(MAX_RETRIES):
                resp = await client.post(
                    url,
                    params={"key": api_key},
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                if resp.status_code == 200:
                    break
                last_status = resp.status_code
                last_body = (resp.text or "")[:400].replace("\n", " ")
                if resp.status_code not in RETRY_STATUSES or attempt == MAX_RETRIES - 1:
                    msg = f"[{model}] HTTP {resp.status_code}: {last_body}"
                    logger.warning("Gemini batch failed — %s", msg)
                    return False, msg
                # Exponential backoff with jitter — Gemini's 503 spikes are
                # usually short, so 4s -> 8s -> 16s gives ~30s before we
                # surrender this model and try a fallback.
                wait = (2 ** (attempt + 1)) + random.uniform(0, 2)
                logger.warning(
                    "Gemini[%s] %s on attempt %d/%d, retrying in %.1fs",
                    model, resp.status_code, attempt + 1, MAX_RETRIES, wait,
                )
                await asyncio.sleep(wait)
            else:
                msg = f"[{model}] HTTP {last_status} after {MAX_RETRIES} retries: {last_body}"
                logger.warning("Gemini batch failed — %s", msg)
                return False, msg
            data = resp.json()

        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as e:
            preview = json.dumps(data)[:400]
            msg = f"[{model}] unexpected response shape ({e}): {preview}"
            logger.warning("Gemini batch failed — %s", msg)
            return False, msg

        try:
            results = json.loads(text)
        except json.JSONDecodeError as e:
            preview = text[:300].replace("\n", " ")
            msg = f"[{model}] response not valid JSON ({e}): {preview}"
            logger.warning("Gemini batch failed — %s", msg)
            return False, msg

        for r in results:
            i = r.get("index")
            if not (isinstance(i, int) and 0 <= i < len(batch)):
                continue
            # Only flip is_fashion when Gemini gives us a real boolean.
            if "is_fashion" in r:
                batch[i]["is_fashion"] = bool(r.get("is_fashion"))
            subniche = (r.get("subniche") or "").strip().lower()
            if subniche:
                batch[i]["subniche"] = subniche
            tags = (r.get("tags") or "").strip().lower()
            if tags:
                batch[i]["ai_tags"] = tags

        return True, None

    except Exception as e:
        msg = f"[{model}] {type(e).__name__}: {e}"
        logger.warning("Gemini batch failed — %s", msg)
        return False, msg


async def _classify_batch_with_fallback(batch: list, api_key: str):
    """Try the primary model, fall back through GEMINI_FALLBACKS on
    persistent failure. Returns (ok, error_message). The error message
    of the FIRST failed model is preserved for diagnostics; only fully
    exhausting the chain marks the batch as failed."""
    models = [GEMINI_PRIMARY] + [m for m in GEMINI_FALLBACKS if m != GEMINI_PRIMARY]
    first_error = None
    for i, model in enumerate(models):
        ok, err = await _classify_batch_with_gemini(batch, api_key, model)
        if ok:
            if i > 0:
                logger.info("Gemini batch succeeded on fallback model %s", model)
            return True, None
        if first_error is None:
            first_error = err
        # Reset is_fashion=None for items so the next model can try fresh.
        for p in batch:
            p["is_fashion"] = None
    return False, first_error


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

    for idx, start in enumerate(range(0, total, BATCH_SIZE)):
        batch = products[start:start + BATCH_SIZE]
        if idx > 0:
            # Gentle inter-batch throttle so we don't burst over the
            # per-minute quota (RPM) which Gemini reports as 503 on the
            # free tier when the model is otherwise healthy.
            await asyncio.sleep(INTER_BATCH_DELAY)
        success, err = await _classify_batch_with_fallback(batch, api_key)
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
