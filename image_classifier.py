"""Image-based product classification via Gemini vision.

Why this exists
---------------
Titles lie. Merchants title a knee-high boot "Women's Boots" for SEO,
tag it "footwear, elegant, comfort", and the text classifier can't
possibly tell it apart from an ankle boot. The image doesn't lie —
one glance and Gemini says "knee-high-boots, women, black, leather".

This module fetches each product's primary image, resizes to 768x768
(the free tier of Gemini vision), and asks gemini-2.5-flash-lite to
classify it against a controlled vocabulary of ~200 fine-grained
product types. The output is prepended to Product.ai_tags with an
`img:` namespace (`img:type:knee-high-boots, img:gender:women,
img:color:black, img:material:leather, img:attr:knee-height`), so
downstream search can hit these via the same ILIKE prefilter it uses
for text tags but with much sharper precision.

Design decisions
----------------
- ONE image per Gemini call. Batching multiple images degrades
  fine-attribute accuracy (documented in Google's vision guidance).
- Runs AFTER the text classifier, ONLY on is_fashion==True products.
  Text handles the fashion-vs-general routing cheaply; vision handles
  the within-fashion precision the user is demanding.
- The Developer API (GEMINI_API_KEY, not Vertex) accepts inline_data
  base64 only — external URLs are Files-API-only. So we fetch,
  resize, and base64-encode inline.
- Fail-open: if vision fails (network / rate limit / SAFETY block),
  the product keeps its text-only ai_tags and life continues. The
  scrape does not fail.

Cost math
---------
- ~1,100 tokens per image (250 for the 768x768 tile + ~800 prompt +
  ~50 output) × Flash-lite $0.10/1M = ~$0.00011 per image.
- 5,000-product catalog backfill = ~$0.55, ~13 minutes at 8x
  concurrency.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import random
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Lazy Pillow import so this module remains importable even if Pillow
# is missing (test envs, CI type-checks). The actual functions that
# need Pillow raise if it's not available at call time.
try:
    from PIL import Image, ImageOps  # type: ignore
    _PIL_OK = True
except ImportError:  # pragma: no cover
    Image = None  # type: ignore
    ImageOps = None  # type: ignore
    _PIL_OK = False


# ---------------------------------------------------------------------
# Constants.
# ---------------------------------------------------------------------
VISION_MODEL_PRIMARY = os.getenv("VISION_MODEL", "gemini-2.5-flash-lite")
VISION_FALLBACKS = ["gemini-2.5-flash", "gemini-flash-latest"]

MAX_IMG_BYTES = 8 * 1024 * 1024      # 8 MB pre-resize cap
RESIZE_MAX = 768                      # matches Gemini's 1-tile ceiling
JPEG_QUALITY = 85
JPEG_FALLBACK_QUALITY = 70            # second-pass if base64 > 4 MB

RETRY_STATUSES = (429, 500, 502, 503, 504)
MAX_RETRIES = 3
VISION_CONCURRENCY_DEFAULT = 8
VISION_TIMEOUT_SECONDS = 30.0


def _gemini_url(model: str) -> str:
    return (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )


# ---------------------------------------------------------------------
# Prompt + response schema.
# ---------------------------------------------------------------------
PROMPT_TEXT = """You are a fashion / e-commerce product vision classifier. You will see ONE product photo. Return a JSON object describing what the product actually IS in the image — ignore any title or marketing text; the image is the ground truth. Precision matters more than recall: if you are not sure, prefer the more generic tag.

Output JSON:
{
  "description":  "<1-3 sentence natural-language description of what the image shows — see DESCRIPTION RULES below>",
  "product_type": "<one of the FINE_TYPES below, or 'other'>",
  "gender":       "<women | men | unisex | kids | unknown>",
  "colors":       "<comma-separated from COLORS>",
  "materials":    "<comma-separated from MATERIALS, may be empty>",
  "occasion":     "<comma-separated from OCCASIONS, may be empty>",
  "attributes":   "<comma-separated from STYLE_ATTRS, may be empty>",
  "confidence":   "<high | medium | low>",
  "not_a_product": <true if the image is a logo, banner, size chart, empty scene, or non-product>
}

DESCRIPTION RULES (the MOST important field — the search judge reads
this directly and reasons about semantic match against the user's
query):
- Write 1-3 short sentences (max ~300 characters) describing what
  is ACTUALLY IN THE IMAGE.
- Cover EVERY axis a shopper might search by:
    * exact product category ("knee-high boots" not just "boots";
      "cocktail dress" not just "dress"; "puffer jacket" not
      "jacket"; "lederhosen" not "pants";)
    * length / height / silhouette ("reaching just below the knee",
      "floor-length", "ankle-length", "cropped at the waist",
      "oversized fit", "bodycon fit")
    * gender presentation ("women's", "men's", "unisex")
    * dominant colors and pattern ("black", "floral print",
      "cream and beige")
    * material or texture cues ("leather", "quilted", "satin",
      "sequin", "chiffon", "knit")
    * key style features ("pointed toe", "sweetheart neckline",
      "high split", "cargo pockets", "faux-fur trim", "hooded")
    * intended occasion ("evening formal", "casual streetwear",
      "athletic sportswear", "beach vacation")
- Use everyday English shopping vocabulary the way a real shopper
  would describe the product to a friend. Do NOT use tag syntax
  (no "knee-high-boots" hyphenated tokens — write "knee-high boots").
- Do NOT include marketing language, brand names, or price info.
- WORKED EXAMPLES (aim for this level of specificity):
    "Tall black leather knee-high boots on a mannequin, reaching
     just below the knee with a pointed toe and low block heel.
     Sleek fitted silhouette for women's evening or workwear."
    "A short women's puffer jacket in bright red with a hood and
     zip front. Quilted channels typical of a down puffer,
     mid-hip length, suitable for cold-weather casual wear."
    "Bavarian men's lederhosen — traditional brown leather
     knee-length shorts with green suspenders and embroidered
     detailing. Oktoberfest / cultural costume style."
    "A casual women's linen shirt-and-shorts summer set in beige.
     Short-sleeve button-up top with matching mid-thigh shorts.
     Beach/resort styling — this is NOT athletic sportswear."

FINE_TYPES (pick exactly ONE):
  # Footwear
  knee-high-boots, thigh-high-boots, ankle-boots, chelsea-boots, combat-boots,
  cowboy-boots, rain-boots, snow-boots, hiking-boots, work-boots,
  sneakers, running-shoes, basketball-shoes, skate-shoes, loafers, mules, clogs,
  ballet-flats, flats, pumps, stiletto-heels, block-heels, wedges,
  sandals, flip-flops, slides, espadrilles, slippers,
  # Dresses / gowns
  mini-dress, midi-dress, maxi-dress, cocktail-dress, evening-gown, ball-gown,
  wedding-dress, bridesmaid-dress, prom-dress, sundress, shirt-dress, sweater-dress,
  bodycon-dress, wrap-dress, slip-dress, t-shirt-dress,
  # Tops
  t-shirt, tank-top, crop-top, blouse, button-up-shirt, polo-shirt, henley,
  sweatshirt, hoodie, sweater, cardigan, turtleneck, tube-top, bodysuit, corset,
  # Outerwear
  puffer-jacket, down-jacket, bomber-jacket, denim-jacket, leather-jacket,
  biker-jacket, blazer, sport-coat, trench-coat, peacoat, overcoat, parka,
  windbreaker, rain-jacket, fleece-jacket, vest, gilet, cape, poncho, kimono,
  # Bottoms
  jeans, skinny-jeans, wide-leg-jeans, cargo-pants, chinos, dress-pants,
  leggings, joggers, sweatpants, shorts, denim-shorts, biker-shorts,
  mini-skirt, midi-skirt, maxi-skirt, pleated-skirt, pencil-skirt,
  # Sets / suits
  tracksuit, sweatsuit, linen-set, matching-set, two-piece-set,
  suit, jumpsuit, romper, overalls, lederhosen, dirndl,
  # Swim / intimates / active
  bikini, one-piece-swimsuit, tankini, swim-trunks, board-shorts,
  bra, sports-bra, panties, thong, lingerie-set, shapewear, pajamas, robe, nightgown,
  sports-top, activewear-set, yoga-pants,
  # Bags
  handbag, tote-bag, shoulder-bag, crossbody-bag, clutch, evening-bag, hobo-bag,
  bucket-bag, backpack, gym-bag, weekender, briefcase, laptop-bag, belt-bag, wallet,
  # Jewelry
  necklace, pendant, choker, earrings, hoop-earrings, stud-earrings, ring,
  bracelet, bangle, cuff, anklet, watch, brooch,
  # Accessories
  sunglasses, eyeglasses, hat, cap, beanie, beret, fedora, bucket-hat,
  scarf, shawl, gloves, mittens, belt, tie, bow-tie, hair-accessory,
  # Non-fashion (short list; anything else -> "other")
  electronics, home-decor, kitchenware, beauty, toy, book, food, tool, lamp, other

COLORS: black, white, gray, beige, brown, tan, cream, ivory, red, burgundy, wine,
  pink, hot-pink, blush, orange, coral, yellow, mustard, gold, green, olive, mint,
  teal, blue, navy, sky-blue, cobalt, purple, lavender, lilac, magenta, silver,
  multicolor, floral-print, animal-print, striped, plaid, polka-dot, tie-dye, camo

MATERIALS: leather, faux-leather, suede, denim, cotton, linen, wool, cashmere,
  silk, satin, chiffon, velvet, corduroy, tweed, knit, crochet, mesh, lace,
  sequin, sheer, ribbed, quilted, fur, faux-fur, shearling, nylon, polyester,
  spandex, metallic

OCCASIONS: formal, evening, cocktail, wedding, prom, business, workwear, casual,
  loungewear, athletic, gym, running, yoga, hiking, beach, resort, festival,
  club, party, streetwear, everyday, oktoberfest

STYLE_ATTRS: backless, open-back, halter, high-neck, mock-neck, off-shoulder,
  one-shoulder, strapless, spaghetti-strap, sleeveless, short-sleeve, long-sleeve,
  puff-sleeve, bell-sleeve, cropped, oversized, fitted, bodycon, loose, relaxed,
  slit, side-slit, front-slit, wrap, ruched, ruffled, tiered, pleated, asymmetric,
  cutout, sheer, mesh-panel, embroidered, embellished, rhinestone, glitter,
  cargo, distressed, ripped, high-waist, low-rise, wide-leg, skinny, flared,
  bootcut, tapered, drop-shoulder, cinched-waist, corset-style,
  ankle-length, knee-length, mini-length, midi-length, maxi-length,
  above-knee, below-knee, floor-length,
  # CRITICAL for boot queries — height is the whole ballgame
  ankle-height, calf-height, knee-height, over-the-knee, thigh-height

Rules:
- product_type: EXACTLY ONE from FINE_TYPES. If the image shows multiple items
  (a full outfit), pick the item that is centered / largest / on the mannequin.
- gender: use apparent presentation + garment cut. Kids only if clearly a child.
  Unisex only for inherently ungendered categories (plain crewneck, unisex sneaker).
  Prefer women/men over unisex when a garment is clearly cut for one.
- colors: 1–3 dominant. Include "floral-print" / "striped" etc. when the pattern
  is the defining visual, ALONGSIDE the base color.
- materials: only when visually obvious (leather sheen, denim weave, satin drape,
  sequin sparkle, ribbed knit, quilted puffer channels). Leave empty rather than guess.
- occasion: infer from garment + styling. Sequin gown = evening,cocktail.
  Hoodie = casual,streetwear,everyday. Empty is fine.
- attributes: everything that helps a shopper filter. For BOOTS, ALWAYS include the
  height attribute (ankle-height / calf-height / knee-height / over-the-knee /
  thigh-height). For DRESSES, ALWAYS include a length attribute.
- not_a_product: true for size charts, brand banners, shipping-info graphics,
  empty studio backdrops, memes, screenshots. When true, leave all other fields
  as empty strings and product_type as "other".

Return ONLY the JSON object, no prose."""


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "description":   {"type": "string"},
        "product_type":  {"type": "string"},
        "gender":        {"type": "string"},
        "colors":        {"type": "string"},
        "materials":     {"type": "string"},
        "occasion":      {"type": "string"},
        "attributes":    {"type": "string"},
        "confidence":    {"type": "string"},
        "not_a_product": {"type": "boolean"},
    },
    "required": ["description", "product_type", "gender", "colors", "confidence", "not_a_product"],
}


# ---------------------------------------------------------------------
# Image fetch + resize pipeline.
# ---------------------------------------------------------------------
async def _fetch_and_prepare_image(url: str, client: httpx.AsyncClient) -> Optional[tuple[bytes, str]]:
    """Fetch a product image, validate content, and encode to a
    Gemini-friendly JPEG (max 768x768). Returns (jpeg_bytes,
    "image/jpeg") on success, None on any failure (bad URL, non-image
    content-type, too-large payload, decode error, Pillow missing).
    """
    if not url or not isinstance(url, str) or not _PIL_OK:
        return None
    try:
        r = await client.get(url, headers={"Accept": "image/*"})
    except Exception as e:
        logger.debug("vision fetch: %r for %s", e, url[:80])
        return None
    if r.status_code != 200:
        return None
    ctype = (r.headers.get("content-type") or "").lower()
    if not ctype.startswith("image/"):
        return None
    raw = r.content
    if not raw or len(raw) > MAX_IMG_BYTES:
        return None
    try:
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        img.thumbnail((RESIZE_MAX, RESIZE_MAX), Image.LANCZOS)
    except Exception as e:
        logger.debug("vision decode: %r for %s", e, url[:80])
        return None

    def _encode(quality: int) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()

    data = _encode(JPEG_QUALITY)
    # base64 grows bytes ~4/3; we want the eventual base64 payload < 4 MB.
    if len(data) * 4 // 3 > 4 * 1024 * 1024:
        data = _encode(JPEG_FALLBACK_QUALITY)
        if len(data) * 4 // 3 > 4 * 1024 * 1024:
            return None
    return data, "image/jpeg"


# ---------------------------------------------------------------------
# Single-image Gemini call with retry + fallback.
# ---------------------------------------------------------------------
async def _call_vision_once(
    img_bytes: bytes,
    mime: str,
    model: str,
    api_key: str,
    client: httpx.AsyncClient,
) -> Optional[dict]:
    """One HTTP request to Gemini with the image + prompt. Returns
    the parsed JSON dict on success, None on any failure."""
    b64 = base64.b64encode(img_bytes).decode("ascii")
    payload = {
        "contents": [{
            "parts": [
                {"text": PROMPT_TEXT},
                {"inline_data": {"mime_type": mime, "data": b64}},
            ]
        }],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
            "responseSchema": RESPONSE_SCHEMA,
            "thinkingConfig": {"thinkingBudget": 0},
        },
        "safetySettings": [
            {"category": c, "threshold": "BLOCK_ONLY_HIGH"}
            for c in (
                "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "HARM_CATEGORY_HATE_SPEECH",
                "HARM_CATEGORY_HARASSMENT",
                "HARM_CATEGORY_DANGEROUS_CONTENT",
            )
        ],
    }
    for attempt in range(MAX_RETRIES):
        try:
            r = await client.post(
                _gemini_url(model),
                params={"key": api_key},
                json=payload,
            )
        except Exception as e:
            logger.debug("vision call: %r on %s attempt %d", e, model, attempt)
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2 ** (attempt + 1) + random.uniform(0, 1))
                continue
            return None
        if r.status_code == 200:
            try:
                data = r.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                return json.loads(text)
            except Exception as e:
                logger.debug("vision parse: %r on %s", e, model)
                return None
        if r.status_code in RETRY_STATUSES and attempt < MAX_RETRIES - 1:
            await asyncio.sleep(2 ** (attempt + 1) + random.uniform(0, 1))
            continue
        logger.debug(
            "vision non-2xx: %d on %s: %s",
            r.status_code, model, r.text[:200],
        )
        return None
    return None


async def _call_vision_with_fallback(
    img_bytes: bytes, mime: str, api_key: str,
    client: httpx.AsyncClient,
) -> Optional[dict]:
    """Try the primary vision model, then each fallback in turn."""
    for model in [VISION_MODEL_PRIMARY, *VISION_FALLBACKS]:
        result = await _call_vision_once(img_bytes, mime, model, api_key, client)
        if result is not None:
            return result
    return None


# ---------------------------------------------------------------------
# Tag rendering.
# ---------------------------------------------------------------------
def _render_tags(vision_json: dict, existing_text_tags: str) -> str:
    """Flatten the vision JSON into img: namespaced tokens plus the
    existing text tags after a `||` separator. See module docstring
    for the exact format."""
    pt = (vision_json.get("product_type") or "").strip().lower()
    gender = (vision_json.get("gender") or "").strip().lower()
    conf = (vision_json.get("confidence") or "").strip().lower()
    colors = _split_csv(vision_json.get("colors") or "")
    mats = _split_csv(vision_json.get("materials") or "")
    occs = _split_csv(vision_json.get("occasion") or "")
    attrs = _split_csv(vision_json.get("attributes") or "")

    parts: list[str] = []
    if pt:
        parts.append(f"img:type:{pt}")
        parts.append(f"img:{pt}")
    if gender:
        parts.append(f"img:gender:{gender}")
    if conf:
        parts.append(f"img:conf:{conf}")
    for c in colors:
        parts.append(f"img:color:{c}")
    for m in mats:
        parts.append(f"img:material:{m}")
    for o in occs:
        parts.append(f"img:occasion:{o}")
    for a in attrs:
        parts.append(f"img:attr:{a}")
        # Also emit as a bare token so query "knee-high boots" hits
        # img:attr:knee-height directly. Redundant but cheap.
        parts.append(f"img:{a}")

    img_str = ", ".join(parts)
    text = (existing_text_tags or "").strip()
    if img_str and text:
        return f"{img_str} || {text}"
    if img_str:
        return img_str
    return text


def _split_csv(s: str) -> list[str]:
    if not s:
        return []
    out = []
    for piece in s.split(","):
        v = piece.strip().lower()
        if v and v not in out:
            out.append(v)
    return out


# ---------------------------------------------------------------------
# Batch entry point (called by scraper + backfill).
# ---------------------------------------------------------------------
async def classify_images_batch(
    products: list[dict],
    *,
    concurrency: int = VISION_CONCURRENCY_DEFAULT,
) -> list[str]:
    """Classify a batch of products by their `image_url`. Each dict
    is mutated in place:
      - `ai_tags` gets prepended with img: tokens + " || " separator
      - `vision_type` set to the classified product_type
      - `vision_confidence` set to "high"|"medium"|"low"
      - `vision_classified` bool marker for the scraper's insert path
      - `_excluded` set to True if the vision model flagged the
        image as not_a_product (banner / size chart / etc.)

    Returns a de-duplicated list of error strings (empty on total
    success). NEVER raises — errors are collected and returned.
    """
    if not products:
        return []
    if not _PIL_OK:
        return ["Pillow not installed — vision classifier disabled"]

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return ["GEMINI_API_KEY not set — vision classifier disabled"]

    errors: set[str] = set()
    sem = asyncio.Semaphore(max(1, min(16, concurrency)))

    async with httpx.AsyncClient(timeout=VISION_TIMEOUT_SECONDS) as client:
        async def _one(p: dict) -> None:
            url = p.get("image_url") or ""
            if not url:
                errors.add("image_url missing on product")
                return
            async with sem:
                fetched = await _fetch_and_prepare_image(url, client)
                if fetched is None:
                    errors.add("image fetch/prepare failed")
                    return
                img_bytes, mime = fetched
                vision_json = await _call_vision_with_fallback(
                    img_bytes, mime, api_key, client,
                )
            if vision_json is None:
                errors.add("vision call failed on all fallback models")
                return
            # Success: mutate the product dict.
            existing = p.get("ai_tags") or ""
            p["ai_tags"] = _render_tags(vision_json, existing)
            p["vision_description"] = (
                (vision_json.get("description") or "").strip()
            )
            p["vision_type"] = (vision_json.get("product_type") or "").lower()
            p["vision_confidence"] = (vision_json.get("confidence") or "").lower()
            p["vision_classified"] = True
            if vision_json.get("not_a_product") is True:
                p["_excluded"] = True

        tasks = [_one(p) for p in products]
        await asyncio.gather(*tasks, return_exceptions=False)

    return sorted(errors)


# ---------------------------------------------------------------------
# One-shot classifier for the /api/debug/vision endpoint.
# ---------------------------------------------------------------------
async def classify_single_image(image_url: str) -> dict:
    """Diagnostic: fetch a single image_url, classify it, return the
    parsed vision JSON. Used by /api/debug/vision to smoke-test."""
    if not _PIL_OK:
        return {"error": "Pillow not installed"}
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"error": "GEMINI_API_KEY not set"}
    async with httpx.AsyncClient(timeout=VISION_TIMEOUT_SECONDS) as client:
        fetched = await _fetch_and_prepare_image(image_url, client)
        if fetched is None:
            return {"error": "image fetch/prepare failed", "image_url": image_url}
        img_bytes, mime = fetched
        vj = await _call_vision_with_fallback(img_bytes, mime, api_key, client)
    if vj is None:
        return {"error": "vision call failed on all models", "image_url": image_url}
    text_placeholder = ""
    return {
        "image_url": image_url,
        "vision": vj,
        "rendered_tags": _render_tags(vj, text_placeholder),
    }
