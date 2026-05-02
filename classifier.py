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
        "Two parallel feeds use your output, plus an 'exclude' bucket:\n"
        "  - Fashion feed (is_fashion=true): EVERYTHING people wear or carry on\n"
        "    their person. Clothing, footwear, intimates/underwear, eyewear,\n"
        "    swimwear, sleepwear, bags, jewelry, watches, accessories. If a\n"
        "    HUMAN can wear it, it is fashion — this includes orthopedic shoes,\n"
        "    bathrobes, lingerie, bras, panties, socks, tights, sunglasses,\n"
        "    reading glasses, progressive lenses, beach sandals, slippers,\n"
        "    wedding-guest dresses. Apparel-shaped therapy products (e.g.\n"
        "    'magnetic massage underwear', 'shapewear', compression hosiery)\n"
        "    are STILL fashion — the form factor wins.\n"
        "  - General feed (is_fashion=false, subniche != exclude): non-wearable\n"
        "    physical products (electronics, home, beauty, health, food, toys-books,\n"
        "    other), grouped by subniche.\n"
        "  - Excluded (subniche='exclude'): items that are NOT physical products\n"
        "    at all — these are dropped entirely by the caller.\n\n"
        "For EACH product below, decide:\n"
        "  1. is_fashion: true for ANY apparel/footwear/intimates/eyewear/bag/\n"
        "                 jewelry/accessory category listed below. False for\n"
        "                 electronics/home/beauty/health/food/toys-books/other.\n"
        "                 False for excluded checkout add-ons.\n"
        "  2. subniche: ONE label from this fixed list. Set the most specific one\n"
        "               that fits the product, REGARDLESS of is_fashion:\n"
        "     - fashion          (clothing, footwear, intimates, eyewear, swimwear\n"
        "                         — anything wearable that isn't a bag/jewelry/\n"
        "                         accessory — is_fashion=true)\n"
        "     - bags             (handbags, backpacks, totes, clutches, wallets,\n"
        "                         crossbody bags, pouches — is_fashion=true)\n"
        "     - jewelry          (necklaces, earrings, rings, bracelets, watches,\n"
        "                         pendants, anklets, brooches, chokers — is_fashion=true)\n"
        "     - accessories      (hats, caps, beanies, scarves, belts, gloves,\n"
        "                         ties, umbrellas — is_fashion=true. Eyewear\n"
        "                         goes under 'fashion', not 'accessories'.)\n"
        "     - electronics      (gadgets, phones, cases, chargers, headphones)\n"
        "     - home             (lamps, candles, furniture, kitchenware, bedding, decor)\n"
        "     - beauty           (skincare, makeup, perfume, fragrances, hair tools)\n"
        "     - health           (supplements, vitamins, wellness, medical aids,\n"
        "                         braces — but NOT orthopedic shoes, those are fashion)\n"
        "     - food             (snacks, drinks, edibles)\n"
        "     - toys-books       (toys, games, books, stationery, pet products,\n"
        "                         children's toys — pet costumes/sunglasses/\n"
        "                         goggles also belong here, NOT fashion)\n"
        "     - exclude          (NOT a physical product — shipping insurance,\n"
        "                         shipping protection, package protection, route\n"
        "                         protection, warranties, extended warranties,\n"
        "                         service plans, gift cards, e-gift cards, store\n"
        "                         credit, tips, donations, deposits, taxes, duties,\n"
        "                         carbon offsets, slidecart upsells, '100% coverage'\n"
        "                         add-ons, ANYTHING that's a checkout-time add-on\n"
        "                         rather than a real item the merchant ships)\n"
        "     - other            (a real physical product that doesn't fit above)\n"
        "  3. tags: 2-6 short English keyword tags (comma-separated, lowercase) "
        "describing the product so we can find it from a multi-language search. "
        "Translate non-English titles into English keywords.\n\n"
        "FULL FASHION SCOPE — these ALL get is_fashion=true:\n"
        "  - Apparel (subniche='fashion'): tops, t-shirts, shirts, blouses,\n"
        "    sweaters, hoodies, cardigans, jackets, coats, blazers, suits,\n"
        "    dresses, skirts, pants, jeans, shorts, leggings, jumpsuits,\n"
        "    overalls, robes, BATHROBES, pajamas, nightgowns, sleepwear,\n"
        "    loungewear, ponchos.\n"
        "  - Underwear & intimates (subniche='fashion'): underwear, panties,\n"
        "    briefs, boxers, thongs, BRAS, lingerie, shapewear, hosiery,\n"
        "    tights, stockings, socks, slips, undershirts.\n"
        "  - Footwear (subniche='fashion'): shoes, sneakers, boots, sandals,\n"
        "    slippers, slip-ons, heels, flats, loafers, ORTHOPEDIC SHOES,\n"
        "    diabetic shoes, ortho slip-ons, shoe covers (when wearable).\n"
        "  - Eyewear (subniche='fashion'): sunglasses, GLASSES, reading\n"
        "    glasses, progressive glasses, bifocals, frames sold as fashion.\n"
        "    NOT magnifying glasses, drinking glasses, hearing aids.\n"
        "  - Swimwear (subniche='fashion'): bikinis, swimsuits, board shorts,\n"
        "    one-pieces, rashguards.\n"
        "  - Bags (subniche='bags'): handbags, backpacks, totes, clutches,\n"
        "    wallets, crossbody bags, pouches, satchels.\n"
        "  - Jewelry (subniche='jewelry'): necklaces, earrings, rings,\n"
        "    bracelets, pendants, watches, anklets, brooches, chokers.\n"
        "  - Accessories (subniche='accessories'): hats, caps, beanies, scarves,\n"
        "    belts, gloves, ties, umbrellas.\n\n"
        "MULTILINGUAL FASHION KEYWORDS — when ANY of these appear in the\n"
        "title or product type, the answer is almost always is_fashion=true:\n"
        "  - German: Bademantel, Bademäntel, Unterwäsche, Unterhose, BH, Bh,\n"
        "    B.H., Schuh, Schuhe, Stiefel, Sandalen, Hausschuh, Halbschuh,\n"
        "    Brille, Brillen, Sonnenbrille, Lesebrille, Kleid, Rock, Hose,\n"
        "    Sakko, Hemd, Bluse, Pullover, Jacke, Mantel, Slip, Strümpfe,\n"
        "    Strumpfhose, Socken, Bademode, Badeanzug, Pyjama, Nachthemd,\n"
        "    Schlafanzug, Hochzeit, Brautkleid, Orthoschuh, orthopädisch.\n"
        "  - French: chemise, chemisier, blouse, pull, pullover, veste,\n"
        "    blouson, manteau, robe, jupe, pantalon, jeans, short, costume,\n"
        "    peignoir, pyjama, sous-vêtements, culotte, soutien-gorge,\n"
        "    collants, chaussettes, chaussures, bottes, sandales, baskets,\n"
        "    escarpins, lunettes, maillot, bikini.\n"
        "  - Spanish: camisa, blusa, suéter, chaqueta, abrigo, vestido, falda,\n"
        "    pantalones, traje, bata, pijama, ropa interior, bragas, sostén,\n"
        "    medias, calcetines, zapatos, botas, sandalias, zapatillas, gafas,\n"
        "    bañador.\n"
        "  - Italian: camicia, camicetta, maglione, giacca, cappotto, vestito,\n"
        "    gonna, pantaloni, vestaglia, intimo, biancheria intima, reggiseno,\n"
        "    calze, scarpe, stivali, sandali, occhiali, costume da bagno.\n"
        "  - Dutch: jurk, rok, broek, trui, jas, badjas, ondergoed, onderbroek,\n"
        "    beha, kousen, sokken, schoenen, laarzen, bril, badpak.\n\n"
        "EDGE CASES (be EXPLICIT):\n"
        "  - 'Bademantel' / 'bathrobe' / 'spa robe' / 'Hooded fleece bathrobe'\n"
        "    → is_fashion=true, subniche='fashion'. NOT home decor.\n"
        "  - 'Orthoschuh' / 'orthopedic shoes' / 'plantar fasciitis shoes'\n"
        "    → is_fashion=true. NOT health.\n"
        "  - 'Unterwäsche', 'Bra', 'BH', 'panty', 'shapewear', 'seamless\n"
        "    underwear' (any language) → is_fashion=true. NOT beauty/health.\n"
        "  - 'Progressive glasses' / 'reading glasses' / 'sunglasses' →\n"
        "    is_fashion=true, subniche='fashion'. NOT 'other'.\n"
        "  - 'Wedding-guest dress' / 'für Hochzeitsgäste' / 'robe de mariée'\n"
        "    → is_fashion=true. NOT 'other'.\n"
        "  - Costumes (Halloween, superhero, dog costumes), pet sunglasses,\n"
        "    pet goggles, dog collars → is_fashion=false, subniche='toys-books'.\n"
        "  - Hearing aids, magnifying glasses, watch boxes (storage), foot\n"
        "    pads (gel cushions), back braces → is_fashion=false. They\n"
        "    LOOK adjacent to fashion but aren't worn as garments.\n\n"
        "When the title or type makes it CLEAR this is a checkout add-on\n"
        "(e.g. 'Versicherter Versand', 'Shipping Protection', 'Geschenkkarte',\n"
        "'Extended Warranty') you MUST set is_fashion=false AND subniche='exclude'.\n"
        "Be strict.\n\n"
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


# ===================================================================
# Wearable-focused reclassification — used by the admin endpoint to
# scan the existing General tab for any item that should have been on
# Fashion. The regex safety net catches the easy multilingual cases;
# this function uses Gemini directly with an explicit "is this worn /
# carried / used to dress?" framing so we catch translingual edge
# cases, branded items with opaque titles, and image-only signals
# (Shopify CDN path often reveals the category when the title doesn't).
# ===================================================================


_RECLASSIFY_STRICT_QUESTION = (
    "Is this product WORN ON THE BODY, or used TO DRESS THE BODY, or "
    "used as a WEARABLE ACCESSORY?\n\n"
    "Examples that are YES (is_fashion=true):\n"
    " - Clothing of any kind (shirts, dresses, pants, jackets, coats, "
    "bathrobes, robes, pajamas, sleepwear, loungewear, ponchos)\n"
    " - Shoes / footwear (sneakers, boots, sandals, slippers, slip-ons, "
    "heels, orthopedic / diabetic shoes, shoe covers)\n"
    " - Underwear / intimates (panties, briefs, boxers, thongs, bras, "
    "lingerie, shapewear, hosiery, tights, stockings, socks, slips)\n"
    " - Eyewear (sunglasses, glasses, reading glasses, progressive "
    "lenses, frames, eyewear, NOT magnifying glasses or hearing aids)\n"
    " - Hats, scarves, belts, gloves, ties, umbrellas, wallets\n"
    " - Bags (handbags, backpacks, totes, clutches, crossbody, pouches)\n"
    " - Jewelry (necklaces, earrings, rings, bracelets, pendants, "
    "watches, anklets, brooches, chokers)\n"
    " - Swimwear (bikinis, swimsuits, board shorts)\n"
    " - Wedding apparel (bride dresses, wedding-guest dresses)\n"
    " - Pet collars, pet harnesses, pet clothes, pet sweaters, pet "
    "raincoats, pet boots — these COUNT as wearable accessories.\n\n"
    "Examples that are NO (is_fashion=false):\n"
    " - Toys, games, books, stationery\n"
    " - Electronics (phones, cases, chargers, headphones, cameras, SSDs)\n"
    " - Home decor (lamps, candles, furniture, kitchenware, bedding, rugs)\n"
    " - Beauty (skincare, makeup, perfume, hair tools)\n"
    " - Health (supplements, vitamins, medical aids, braces, urine bags)\n"
    " - Food / drinks / supplements\n"
    " - Tools / hardware / pest control / outdoor gear\n"
    " - Pet food, pet beds, pet cushions, pet treats\n"
    " - Costumes (Halloween, cosplay, superhero, latex masks) — these "
    "are NOT fashion, they're toys.\n"
    " - Watch boxes (storage), magnifying glasses, foot pain pads, "
    "walking sticks, gel cushions, hearing aids."
)

_RECLASSIFY_BROAD_QUESTION = (
    "Could a reasonable person consider this product to be in the "
    "FASHION, APPAREL, FOOTWEAR, ACCESSORY, EYEWEAR, JEWELRY, or "
    "WEARABLE category?\n\n"
    "If a person could wear it, carry it on their person, dress "
    "themselves with it, or buy it specifically to dress their pet, "
    "the answer is YES.\n\n"
    "If it's a household object, decoration, electronic device, "
    "beauty/health/food product, kitchenware, tool, pet bed/food/treat, "
    "OR specifically a costume / Halloween mask / cosplay item, the "
    "answer is NO.\n\n"
    "Be DECISIVE — when in doubt, say YES rather than NO. The cost of "
    "false positives (a borderline home item ending up on Fashion) is "
    "far less than the cost of false negatives (real apparel staying on "
    "General). When the title is opaque, lean on the handle (URL slug) "
    "and image_url (CDN filename) — those reveal the category."
)


async def reclassify_general_with_gemini(
    products: list, framing: str = "strict",
) -> tuple[list, list]:
    """Run the focused 'is wearable?' check against `products` (each is
    a dict with at least title, handle, product_type, image_url).
    Returns (flagged, errors): `flagged` is the subset Gemini said yes
    to, with a `reason` field attached describing why; `errors` is a
    list of human-readable problem strings (empty when everything was
    clean).

    Each product is scored independently — we don't flip is_fashion
    here, the caller decides what to do with the YES list.
    """
    if not products:
        return [], []

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return [], ["GEMINI_API_KEY not set on server"]

    question = (
        _RECLASSIFY_STRICT_QUESTION if framing == "strict"
        else _RECLASSIFY_BROAD_QUESTION
    )

    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "index": {"type": "integer"},
                "is_fashion": {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": ["index", "is_fashion", "reason"],
        },
    }

    flagged: list = []
    errors: list = []
    total = len(products)
    for start in range(0, total, BATCH_SIZE):
        batch = products[start:start + BATCH_SIZE]
        if start > 0:
            await asyncio.sleep(INTER_BATCH_DELAY)

        items = [
            {
                "index": idx,
                "title": p.get("title", "") or "",
                "handle": p.get("handle", "") or "",
                "product_type": p.get("product_type", "") or "",
                # Send only the CDN filename, not the full URL — saves
                # tokens and the basename is the only signal-bearing part.
                "image_filename": (p.get("image_url") or "").split("/")[-1].split("?")[0][:120],
            }
            for idx, p in enumerate(batch)
        ]

        prompt = (
            f"{question}\n\n"
            "For EACH product below, answer is_fashion=true or "
            "is_fashion=false plus a 1-sentence reason. Use the title, "
            "handle (URL slug), product_type (Shopify category), AND "
            "image_filename together — sometimes the title is opaque "
            "(e.g. 'Mia™ Premium Set') but the handle / image filename "
            "reveals the category clearly.\n\n"
            f"Products (JSON):\n{json.dumps(items, ensure_ascii=False)}\n\n"
            "Respond with a JSON array, one object per product, matching "
            "input indices."
        )

        ok, parsed_or_err = await _gemini_yesno_call(prompt, schema, api_key)
        if not ok:
            errors.append(parsed_or_err)
            continue

        for r in parsed_or_err:
            i = r.get("index")
            if not (isinstance(i, int) and 0 <= i < len(batch)):
                continue
            if not r.get("is_fashion"):
                continue
            entry = dict(batch[i])
            entry["reclassify_reason"] = (r.get("reason") or "").strip()[:200]
            flagged.append(entry)

    logger.info(
        "reclassify_general_with_gemini[%s]: flagged %d/%d (errors=%d)",
        framing, len(flagged), total, len(errors),
    )
    return flagged, errors


async def _gemini_yesno_call(prompt: str, schema: dict, api_key: str):
    """Stripped-down Gemini call mirroring _classify_batch_with_gemini's
    retry / fallback chain. Returns (ok, parsed_or_error_string).
    Separate from _classify_batch_with_gemini because the caller wants
    the raw parsed JSON array, not a side-effect mutation of `batch`.
    """
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
            "responseSchema": schema,
        },
    }
    models = [GEMINI_PRIMARY] + [m for m in GEMINI_FALLBACKS if m != GEMINI_PRIMARY]
    first_error = None
    for model in models:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                last_status = None
                last_body = ""
                for attempt in range(MAX_RETRIES):
                    resp = await client.post(
                        _gemini_url(model),
                        params={"key": api_key},
                        json=payload,
                        headers={"Content-Type": "application/json"},
                    )
                    if resp.status_code == 200:
                        break
                    last_status = resp.status_code
                    last_body = (resp.text or "")[:300].replace("\n", " ")
                    if resp.status_code not in RETRY_STATUSES or attempt == MAX_RETRIES - 1:
                        msg = f"[{model}] HTTP {resp.status_code}: {last_body}"
                        if first_error is None:
                            first_error = msg
                        break
                    wait = (2 ** (attempt + 1)) + random.uniform(0, 2)
                    await asyncio.sleep(wait)
                else:
                    msg = f"[{model}] HTTP {last_status} after {MAX_RETRIES} retries: {last_body}"
                    if first_error is None:
                        first_error = msg
                    continue
                if resp.status_code != 200:
                    continue
                data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            parsed = json.loads(text)
            return True, parsed
        except Exception as e:
            msg = f"[{model}] {type(e).__name__}: {e}"
            if first_error is None:
                first_error = msg
            continue
    return False, first_error or "unknown Gemini failure"
