"""Gemini-powered product classifier with keyword fallback."""
import os
import json
import logging
import httpx

logger = logging.getLogger(__name__)

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

FASHION_KEYWORDS = {
    "dress", "shirt", "t-shirt", "tshirt", "tee", "pants", "trousers", "jacket",
    "coat", "skirt", "sweater", "shoes", "boots", "bag", "jeans", "blouse",
    "top", "hoodie", "cardigan", "vest", "shorts", "blazer", "polo", "sneakers",
    "sandals", "heels", "hat", "cap", "scarf", "belt", "sunglasses", "lingerie",
    "swimwear", "bikini", "tracksuit", "accessories", "leggings", "kleid",
    "hemd", "hose", "rock", "schuhe", "tasche", "mantel", "jacke", "pullover",
    "jeans", "stiefel", "robe", "chemise", "pantalon", "veste", "manteau",
    "jupe", "chaussures", "sac", "vestido", "camisa", "pantalones", "chaqueta",
    "abrigo", "falda", "zapatos", "bolso", "botas",
}


def _keyword_fallback(products: list) -> list:
    for p in products:
        title_lower = (p.get("title") or "").lower()
        ptype_lower = (p.get("product_type") or "").lower()
        haystack = f"{title_lower} {ptype_lower}"
        matched = [kw for kw in FASHION_KEYWORDS if kw in haystack]
        if matched:
            p["is_fashion"] = True
            p["ai_tags"] = ",".join(sorted(set(matched)))
        else:
            p["is_fashion"] = False
            p["ai_tags"] = ""
    return products


def classify_products_batch(products: list) -> list:
    """Classify a batch of product dicts in place. Adds 'ai_tags' (str) and 'is_fashion' (bool)."""
    if not products:
        return products

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.info("GEMINI_API_KEY not set, using keyword fallback for %d products", len(products))
        return _keyword_fallback(products)

    items = []
    for idx, p in enumerate(products):
        items.append({
            "index": idx,
            "title": p.get("title", ""),
            "vendor": p.get("vendor", ""),
            "product_type": p.get("product_type", ""),
        })

    prompt = (
        "You are a product classifier for a fashion bestseller tracker. "
        "For EACH product below, return:\n"
        "- is_fashion: true if it's apparel, footwear, bags, or fashion accessories; "
        "false for things like gift cards, candles, home decor, electronics, food.\n"
        "- tags: 2-6 concise English keywords describing the product "
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
            "temperature": 0.1,
            "responseMimeType": "application/json",
            "responseSchema": schema,
        },
    }

    try:
        with httpx.Client(timeout=30.0) as client:
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
            if isinstance(i, int) and 0 <= i < len(products):
                products[i]["is_fashion"] = bool(r.get("is_fashion", True))
                products[i]["ai_tags"] = (r.get("tags") or "").strip().lower()

        for p in products:
            p.setdefault("is_fashion", True)
            p.setdefault("ai_tags", "")

        logger.info("Gemini classified %d products", len(products))
        return products

    except Exception as e:
        logger.warning("Gemini classification failed (%s), using keyword fallback", e)
        return _keyword_fallback(products)
