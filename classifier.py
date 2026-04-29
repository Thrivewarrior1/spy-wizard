"""Gemini-powered product classifier with batching and keyword fallback."""
import os
import json
import logging
import httpx

logger = logging.getLogger(__name__)

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

BATCH_SIZE = 25

# Comprehensive fashion / apparel / accessory keywords across EN/DE/FR/ES.
FASHION_KEYWORDS = {
    # tops
    "shirt", "t-shirt", "tshirt", "tee", "blouse", "top", "tank", "camisole",
    "polo", "sweater", "pullover", "hoodie", "sweatshirt", "cardigan", "vest",
    "blazer", "jacket", "coat", "parka", "trench", "windbreaker", "puffer",
    "bomber", "anorak",
    # bottoms
    "pants", "trousers", "jeans", "leggings", "shorts", "skirt", "joggers",
    "sweatpants", "chinos", "slacks", "culottes", "bermudas",
    # dresses & one-pieces
    "dress", "gown", "jumpsuit", "romper", "playsuit", "kaftan", "tunic",
    # underwear / loungewear / swim
    "lingerie", "bra", "panties", "underwear", "boxer", "briefs", "thong",
    "pajamas", "pyjama", "loungewear", "robe", "nightgown", "sleepwear",
    "swimwear", "swimsuit", "bikini", "trunks", "rashguard",
    # shoes
    "shoes", "sneakers", "trainers", "boots", "ankle-boots", "heels", "pumps",
    "sandals", "flats", "loafers", "mules", "espadrilles", "slippers", "clogs",
    "oxfords", "flip-flops",
    # bags
    "bag", "handbag", "purse", "tote", "backpack", "rucksack", "satchel",
    "clutch", "wallet", "crossbody", "messenger-bag", "duffel",
    # accessories
    "hat", "cap", "beanie", "fedora", "scarf", "shawl", "belt", "tie",
    "bowtie", "gloves", "mittens", "sunglasses", "watch", "jewelry",
    "necklace", "earrings", "bracelet", "ring", "anklet", "brooch", "cufflinks",
    "umbrella", "headband",
    # generic
    "outfit", "apparel", "clothing", "fashion", "accessories", "tracksuit",
    "activewear", "sportswear", "athleisure", "kidswear", "menswear",
    "womenswear", "uniform",
    # German
    "kleid", "hemd", "hose", "rock", "schuhe", "tasche", "mantel", "jacke",
    "stiefel", "stiefelette", "muetze", "mütze", "schal", "guertel", "gürtel",
    "handschuhe", "sonnenbrille", "unterwäsche", "badeanzug", "anzug",
    "trikot", "trainingsanzug", "strickjacke",
    # French
    "robe", "chemise", "pantalon", "veste", "manteau", "jupe", "chaussures",
    "sac", "ceinture", "chapeau", "écharpe", "echarpe", "lunettes", "bottes",
    "sandales", "baskets", "maillot", "sous-vetements", "pyjama",
    # Spanish
    "vestido", "camisa", "pantalones", "chaqueta", "abrigo", "falda",
    "zapatos", "bolso", "botas", "cinturon", "cinturón", "sombrero", "bufanda",
    "gafas", "traje", "calcetines", "pijama", "bañador", "banador",
}


def _keyword_fallback(products: list) -> list:
    """Tag each product using FASHION_KEYWORDS as a backup when Gemini is unavailable."""
    for p in products:
        title_lower = (p.get("title") or "").lower()
        ptype_lower = (p.get("product_type") or "").lower()
        vendor_lower = (p.get("vendor") or "").lower()
        haystack = f"{title_lower} {ptype_lower} {vendor_lower}"
        matched = [kw for kw in FASHION_KEYWORDS if kw in haystack]
        if matched:
            p["is_fashion"] = True
            p["ai_tags"] = ",".join(sorted(set(matched))[:6])
        else:
            p["is_fashion"] = False
            p["ai_tags"] = ""
    return products


def _classify_batch_with_gemini(batch: list, api_key: str) -> bool:
    """Classify a single batch in place. Returns True on success, False on failure."""
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
            "temperature": 0.05,
            "responseMimeType": "application/json",
            "responseSchema": schema,
        },
    }

    try:
        with httpx.Client(timeout=45.0) as client:
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
                batch[i]["is_fashion"] = bool(r.get("is_fashion", True))
                batch[i]["ai_tags"] = (r.get("tags") or "").strip().lower()

        for p in batch:
            p.setdefault("is_fashion", True)
            p.setdefault("ai_tags", "")

        return True

    except Exception as e:
        logger.warning("Gemini batch failed (%s)", e)
        return False


def classify_products_batch(products: list) -> list:
    """Classify products in place in chunks of BATCH_SIZE.

    Adds 'ai_tags' (str) and 'is_fashion' (bool) to each product. If Gemini fails
    for a given batch, that batch falls back to keyword classification while other
    batches continue with Gemini.
    """
    if not products:
        return products

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.info("GEMINI_API_KEY not set, using keyword fallback for %d products", len(products))
        return _keyword_fallback(products)

    total = len(products)
    gemini_ok = 0
    fallback_used = 0

    for start in range(0, total, BATCH_SIZE):
        batch = products[start:start + BATCH_SIZE]
        if _classify_batch_with_gemini(batch, api_key):
            gemini_ok += len(batch)
        else:
            _keyword_fallback(batch)
            fallback_used += len(batch)

    logger.info(
        "Classified %d products (gemini=%d, keyword_fallback=%d, batch_size=%d)",
        total, gemini_ok, fallback_used, BATCH_SIZE,
    )
    return products
