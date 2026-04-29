"""Gemini-powered strict fashion classifier.

Each product is tagged is_fashion=True ONLY if it is clothing, shoes, or bags.
Everything else (jewelry, watches, accessories, shipping protection, gift cards,
electronics, home decor, candles, beauty, supplements, etc.) is is_fashion=False.

Gemini is the single source of truth — there is no keyword fallback. If Gemini
is unavailable or fails for a batch, every product in that batch is marked
is_fashion=False so it never enters the bestseller feed.
"""
import os
import json
import logging
import httpx

logger = logging.getLogger(__name__)

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

BATCH_SIZE = 20


def _classify_batch_with_gemini(batch: list, api_key: str) -> bool:
    """Classify a single batch in place. Returns True on success, False on failure.

    On success each product gets:
      - is_fashion: bool (clothing/shoes/bags only)
      - ai_tags:    str (comma-separated English keywords)
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
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                GEMINI_URL,
                params={"key": api_key},
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()

        text = data["candidates"][0]["content"]["parts"][0]["text"]
        results = json.loads(text)

        for r in results:
            i = r.get("index")
            if isinstance(i, int) and 0 <= i < len(batch):
                batch[i]["is_fashion"] = bool(r.get("is_fashion", False))
                batch[i]["ai_tags"] = (r.get("tags") or "").strip().lower()

        # Anything Gemini didn't return for is treated as non-fashion.
        for p in batch:
            p.setdefault("is_fashion", False)
            p.setdefault("ai_tags", "")

        return True

    except Exception as e:
        logger.warning("Gemini batch failed (%s)", e)
        return False


def _mark_all_non_fashion(batch: list):
    for p in batch:
        p["is_fashion"] = False
        p["ai_tags"] = ""


def classify_products_batch(products: list) -> list:
    """Classify products in place in chunks of BATCH_SIZE.

    Adds 'ai_tags' (str) and 'is_fashion' (bool) to each product. Gemini is the
    only classifier — if the API key is missing or a batch fails, those products
    are marked is_fashion=False so they are excluded from the feed.
    """
    if not products:
        return products

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning(
            "GEMINI_API_KEY not set — marking all %d products non-fashion", len(products)
        )
        _mark_all_non_fashion(products)
        return products

    total = len(products)
    gemini_ok = 0
    failed = 0

    for start in range(0, total, BATCH_SIZE):
        batch = products[start:start + BATCH_SIZE]
        if _classify_batch_with_gemini(batch, api_key):
            gemini_ok += len(batch)
        else:
            _mark_all_non_fashion(batch)
            failed += len(batch)

    fashion_count = sum(1 for p in products if p.get("is_fashion"))
    logger.info(
        "Classified %d products (gemini_ok=%d, failed=%d, fashion=%d, non_fashion=%d)",
        total, gemini_ok, failed, fashion_count, total - fashion_count,
    )
    return products
