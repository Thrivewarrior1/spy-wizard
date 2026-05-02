import os
import asyncio
import logging
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, or_
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from pydantic import BaseModel
from typing import Optional

from database import get_db, engine, SessionLocal, widen_text_columns
from models import Base, Store, Product, PositionHistory
from scraper import (
    scrape_all_stores,
    scrape_store_bestsellers,
    update_products_in_db,
    reset_all_labels,
    debug_fetch,
    cleanup_non_product_rows,
    migrate_wearables_to_fashion,
)
from seed import seed_stores

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APP_PASSWORD = os.getenv("APP_PASSWORD", "mats2310")
PORT = int(os.getenv("PORT", "8000"))


# =====================================================================
# Trust epoch — heroes / villains baseline guard
# =====================================================================
# Snapshots in PositionHistory dated BEFORE TRUST_EPOCH_UTC are not
# trustworthy comparators because the underlying product set differed
# (different cap, different hard-drop regex, different fashion-only
# filter, different schema). Bump this every time we ship a change
# that meaningfully reshapes the catalog so day-over-day deltas don't
# get polluted by structural reshuffles.
#
# Override at deploy time with the TRUST_EPOCH_UTC env var (ISO 8601).
# Default is the most recent significant change — bump it when shipping
# a cap change, hard-drop tightening, schema migration, etc.
_DEFAULT_TRUST_EPOCH = datetime(2026, 5, 1, 0, 0, 0)


def _parse_trust_epoch(raw: Optional[str]) -> datetime:
    if not raw:
        return _DEFAULT_TRUST_EPOCH
    raw = raw.strip()
    if raw.endswith("Z"):
        raw = raw[:-1]
    try:
        parsed = datetime.fromisoformat(raw)
        # Strip tzinfo so comparisons against naive PositionHistory.date
        # (also UTC, also naive) behave consistently.
        return parsed.replace(tzinfo=None)
    except ValueError:
        logger.warning(
            "TRUST_EPOCH_UTC=%r is not parseable ISO 8601; falling back to %s",
            raw, _DEFAULT_TRUST_EPOCH.isoformat(),
        )
        return _DEFAULT_TRUST_EPOCH


TRUST_EPOCH_UTC = _parse_trust_epoch(os.getenv("TRUST_EPOCH_UTC"))


# Hard cap on plausible day-over-day rank movement. Anything larger
# is almost certainly a structural reshuffle (catalog grew/shrank,
# cap changed, scrape source changed) rather than organic movement,
# so we suppress the hero/villain label and call it 'normal' instead.
# The per-store catalog-size sanity check (30% of catalog, applied in
# _compute_label_map) further tightens this for smaller stores.
HERO_VILLAIN_DELTA_CAP = 30
HERO_VILLAIN_CATALOG_FRACTION = 0.30

scheduler = AsyncIOScheduler()

# Track an in-flight all-stores scrape so the API can report progress and
# avoid stacking parallel scrapes. Stored in module state because each
# scrape needs its own DB session and runs longer than any HTTP request
# Railway will tolerate.
_scrape_state = {"running": False, "started_at": None, "result": None}


async def daily_scrape():
    """Run the daily scrape job."""
    if _scrape_state["running"]:
        logger.info("Skipping daily scrape — another scrape is already in flight")
        return
    logger.info("Running daily scrape job...")
    _scrape_state["running"] = True
    _scrape_state["started_at"] = datetime.utcnow().isoformat()
    db = SessionLocal()
    try:
        _scrape_state["result"] = await scrape_all_stores(db)
    finally:
        db.close()
        _scrape_state["running"] = False


async def _background_scrape_all():
    db = SessionLocal()
    try:
        _scrape_state["result"] = await scrape_all_stores(db)
    except Exception as e:
        _scrape_state["result"] = {
            "stores": [], "total_products": 0,
            "stores_with_products": 0, "stores_failed": 0,
            "error": str(e),
        }
        logger.exception("Background scrape failed")
    finally:
        db.close()
        _scrape_state["running"] = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    widen_text_columns()
    seed_stores()
    # One-shot DB-wide junk sweep on every redeploy. Idempotent —
    # purges any product whose title/product_type matches the
    # hard-exclude regex, fixing legacy "services" rows that slipped
    # in before the regex was tightened.
    db = SessionLocal()
    try:
        cleanup_non_product_rows(db)
    except Exception as e:
        logger.warning(f"startup junk cleanup failed: {e}")
    try:
        # Promote legacy jewelry/accessories/bags rows from General to
        # Fashion so the user-visible reclassification takes effect on
        # the next page load instead of waiting for tomorrow's scrape.
        migrate_wearables_to_fashion(db)
    except Exception as e:
        logger.warning(f"startup wearable migration failed: {e}")
    finally:
        db.close()
    # Schedule daily scrape at 6 AM UTC
    scheduler.add_job(daily_scrape, "cron", hour=6, minute=0, id="daily_scrape")
    # Also scrape every 12 hours as backup
    scheduler.add_job(daily_scrape, "interval", hours=12, id="backup_scrape")
    scheduler.start()
    logger.info("Scheduler started - daily scrape at 06:00 UTC + every 12h backup")
    yield
    scheduler.shutdown()

app = FastAPI(title="Spy Wizard", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- Auth ---
class LoginRequest(BaseModel):
    password: str

@app.post("/api/login")
async def login(req: LoginRequest):
    if req.password == APP_PASSWORD:
        return {"success": True, "token": "authenticated"}
    raise HTTPException(status_code=401, detail="Invalid password")

# --- Stores CRUD ---
class StoreCreate(BaseModel):
    name: str
    url: str
    monthly_visitors: str = "0"
    niche: str = "Fashion"
    country: str = ""

class StoreUpdate(BaseModel):
    name: Optional[str] = None
    url: Optional[str] = None
    monthly_visitors: Optional[str] = None
    niche: Optional[str] = None
    country: Optional[str] = None

@app.get("/api/stores")
async def get_stores(db: Session = Depends(get_db)):
    stores = db.query(Store).order_by(Store.name).all()
    # Counts are split by feed and exposed separately so the UI can show
    # whichever is relevant. The Stores tab no longer surfaces them at
    # all per the latest design decision, but the API still returns
    # them for /api/stats and any future consumer.
    return [{
        "id": s.id, "name": s.name, "url": s.url,
        "monthly_visitors": s.monthly_visitors, "niche": s.niche,
        "country": s.country,
        "product_count": sum(1 for p in s.products if p.is_fashion),
        "general_count": sum(1 for p in s.products if not p.is_fashion and p.subniche),
        "last_scraped": max((p.last_scraped for p in s.products if p.is_fashion or p.subniche), default=None),
    } for s in stores]

@app.post("/api/stores")
async def create_store(store: StoreCreate, db: Session = Depends(get_db)):
    existing = db.query(Store).filter(Store.url == store.url).first()
    if existing:
        raise HTTPException(status_code=400, detail="Store already exists")
    new_store = Store(**store.model_dump())
    db.add(new_store)
    db.commit()
    db.refresh(new_store)
    return {"id": new_store.id, "name": new_store.name}

@app.put("/api/stores/{store_id}")
async def update_store(store_id: int, store: StoreUpdate, db: Session = Depends(get_db)):
    existing = db.query(Store).filter(Store.id == store_id).first()
    if not existing:
        raise HTTPException(status_code=404, detail="Store not found")
    for key, value in store.model_dump(exclude_none=True).items():
        setattr(existing, key, value)
    db.commit()
    return {"success": True}

@app.delete("/api/stores/{store_id}")
async def delete_store(store_id: int, db: Session = Depends(get_db)):
    store = db.query(Store).filter(Store.id == store_id).first()
    if not store:
        raise HTTPException(status_code=404, detail="Store not found")
    db.delete(store)
    db.commit()
    return {"success": True}

# --- Bestsellers ---
def parse_visitors(v: str) -> int:
    s = v.strip().upper()
    try:
        if s.endswith("M"): return int(float(s[:-1]) * 1_000_000)
        elif s.endswith("K"): return int(float(s[:-1]) * 1_000)
        else: return int(s)
    except: return 0

# Translation map for multi-language search
TRANSLATIONS = {
    "dress": ["kleid", "robe", "vestido", "jurk", "abito"],
    "summer": ["sommer", "été", "verano", "zomer", "estate"],
    "shirt": ["hemd", "chemise", "camisa", "overhemd", "camicia"],
    "pants": ["hose", "pantalon", "pantalones", "broek", "pantaloni"],
    "jacket": ["jacke", "veste", "chaqueta", "jas", "giacca"],
    "coat": ["mantel", "manteau", "abrigo", "jas", "cappotto"],
    "skirt": ["rock", "jupe", "falda", "rok", "gonna"],
    "sweater": ["pullover", "pull", "suéter", "trui", "maglione"],
    "shoes": ["schuhe", "chaussures", "zapatos", "schoenen", "scarpe"],
    "boots": ["stiefel", "bottes", "botas", "laarzen", "stivali"],
    "bag": ["tasche", "sac", "bolso", "tas", "borsa"],
    "jeans": ["jeans", "jean", "vaqueros", "spijkerbroek"],
    "blouse": ["bluse", "chemisier", "blusa", "blouse"],
    "top": ["oberteil", "haut", "parte superior", "top"],
    "winter": ["winter", "hiver", "invierno", "inverno"],
    "spring": ["frühling", "printemps", "primavera", "lente"],
    "autumn": ["herbst", "automne", "otoño", "herfst", "autunno"],
    "cotton": ["baumwolle", "coton", "algodón", "katoen", "cotone"],
    "silk": ["seide", "soie", "seda", "zijde", "seta"],
    "linen": ["leinen", "lin", "lino", "linnen"],
    "wool": ["wolle", "laine", "lana", "wol"],
    "men": ["herren", "homme", "hombre", "heren", "uomo"],
    "women": ["damen", "femme", "mujer", "dames", "donna"],
    "trousers": ["hose", "hosen", "pantalon", "pantalones", "broek", "pantaloni", "pants"],
    "t-shirt": ["t-shirt", "tshirt", "tee"],
    "hoodie": ["hoodie", "kapuzenpullover", "sweat"],
    "cardigan": ["cardigan", "strickjacke"],
    "vest": ["weste", "gilet", "chaleco"],
    "shorts": ["shorts", "kurze hose", "bermuda"],
    "tracksuit": ["trainingsanzug", "jogging", "survêtement"],
    "blazer": ["blazer", "sakko"],
    "polo": ["polo", "poloshirt"],
    "sneakers": ["sneaker", "turnschuhe", "baskets"],
    "sandals": ["sandalen", "sandales", "sandalias"],
    "heels": ["absatzschuhe", "talons", "tacones"],
    "accessories": ["accessoires", "zubehör", "accesorios"],
    "hat": ["hut", "mütze", "chapeau", "sombrero"],
    "scarf": ["schal", "écharpe", "bufanda"],
    "belt": ["gürtel", "ceinture", "cinturón"],
    "sunglasses": ["sonnenbrille", "lunettes", "gafas"],
    "lingerie": ["unterwäsche", "dessous", "lencería"],
    "swimwear": ["bademode", "maillot", "bañador", "bikini"],
    "maxi": ["maxi", "lang"],
    "mini": ["mini", "kurz"],
    "casual": ["casual", "lässig", "décontracté"],
    "elegant": ["elegant", "élégant", "elegante", "festlich"],
    "vintage": ["vintage", "retro"],
    "floral": ["blumen", "floral", "blumenmuster"],
    "striped": ["gestreift", "rayé", "rayado"],
}

def _singularize(term: str) -> set:
    """Return {term, naive-singular(term)} so a search for 'bags' also
    matches 'bag', 'shoes' also matches 'shoe', etc. Naive on purpose —
    we'd rather over-match than miss obvious plurals."""
    out = {term}
    if len(term) > 3 and term.endswith("s") and not term.endswith("ss"):
        out.add(term[:-1])
    if len(term) > 4 and term.endswith("es"):
        out.add(term[:-2])
    return out


# Reverse-lookup map: typing a colloquial keyword should expand the
# search to also match the canonical Gemini subniche label stored on
# Product.subniche. The user explicitly does NOT want subniche to be a
# user-facing filter — it's purely backend search metadata. So when
# someone types "earring" we add "jewelry" to the variant set, which
# then matches Product.subniche='jewelry' even when the title is
# "Stud Studded Loops" with no obvious keyword. Keys are the canonical
# subniche labels Gemini emits in classifier.py.
SUBNICHE_SYNONYMS = {
    "bags": [
        "bag", "handbag", "handbags", "backpack", "backpacks", "tote",
        "totes", "clutch", "clutches", "wallet", "wallets", "purse",
        "purses", "crossbody", "pouch", "pouches", "satchel", "satchels",
    ],
    "jewelry": [
        "jewellery", "jewel", "jewels", "earring", "earrings", "necklace",
        "necklaces", "ring", "rings", "bracelet", "bracelets", "pendant",
        "pendants", "watch", "watches", "anklet", "anklets", "brooch",
        "brooches", "choker", "chokers",
    ],
    "accessories": [
        "hat", "hats", "scarf", "scarfs", "scarves", "belt", "belts",
        "sunglasses", "glasses", "glove", "gloves", "tie", "ties",
        "wallet", "wallets", "cap", "caps", "beanie", "umbrella",
    ],
    "electronics": [
        "phone", "phones", "smartphone", "tablet", "tablets", "laptop",
        "laptops", "headphone", "headphones", "earbud", "earbuds",
        "speaker", "speakers", "charger", "chargers", "cable", "cables",
        "gadget", "gadgets", "tech", "device", "devices", "camera",
        "cameras", "drone", "drones", "console", "consoles", "monitor",
        "keyboard", "mouse", "powerbank", "smartwatch",
    ],
    "home": [
        "lamp", "lamps", "candle", "candles", "furniture", "decor",
        "decoration", "kitchenware", "kitchen", "bedding", "pillow",
        "pillows", "blanket", "blankets", "rug", "rugs", "vase", "vases",
        "mug", "mugs", "plate", "plates", "cup", "cups", "bowl",
        "bowls", "curtain", "curtains", "frame", "frames", "clock",
        "clocks", "diffuser",
    ],
    "beauty": [
        "skincare", "makeup", "perfume", "perfumes", "fragrance",
        "fragrances", "lipstick", "mascara", "foundation", "cream",
        "creams", "lotion", "lotions", "serum", "serums", "shampoo",
        "conditioner", "cosmetic", "cosmetics", "nail", "nails",
    ],
    "health": [
        "supplement", "supplements", "vitamin", "vitamins", "wellness",
        "fitness", "protein", "collagen", "probiotic", "probiotics",
        "massager", "thermometer",
    ],
    "food": [
        "snack", "snacks", "drink", "drinks", "candy", "candies",
        "chocolate", "chocolates", "tea", "teas", "coffee", "coffees",
        "biscuit", "biscuits", "cookie", "cookies",
    ],
    "toys-books": [
        "toy", "toys", "book", "books", "game", "games", "puzzle",
        "puzzles", "stationery", "pet", "pets", "plush", "plushie",
        "doll", "dolls", "lego", "boardgame",
    ],
    "other": [],
}


def expand_single_term(term: str) -> list:
    """Expand a single search term to include its translations and a
    naive singular form. 'bags' -> {bags, bag, tasche, sac, ...},
    'women' -> {women, woman, damen, femme, ...}.

    Also reverse-lookups subniche synonyms so 'earring' adds 'jewelry'
    as a variant, letting the search hit Product.subniche when the
    title doesn't contain the keyword directly.
    """
    term = term.lower().strip()
    variants = set()
    for base in _singularize(term):
        variants.add(base)
        if base in TRANSLATIONS:
            variants.update(TRANSLATIONS[base])
        for eng, trans in TRANSLATIONS.items():
            if base in trans:
                variants.add(eng)
                variants.update(trans)
        # Subniche reverse-lookup: typing 'earring' should add 'jewelry'
        # so we hit Product.subniche='jewelry' even when nothing else
        # mentions earrings. Symmetric: typing 'jewelry' itself adds
        # all known synonyms, broadening the match into title/ai_tags.
        for subniche_label, synonyms in SUBNICHE_SYNONYMS.items():
            if base == subniche_label or base in synonyms:
                variants.add(subniche_label)
                variants.update(synonyms)
    return list(variants)


import re as _re


def _is_postgres_db() -> bool:
    """True when the live DB is Postgres so we can use regex matching
    with word boundaries; otherwise fall back to plain ILIKE."""
    return os.getenv("DATABASE_URL", "").startswith(("postgres://", "postgresql://"))


def _match_clauses(column, variant: str):
    """Return one or more SQLAlchemy clauses that match `variant`
    against `column`. On Postgres we use case-insensitive regex with
    a trailing word boundary (\\y) so "bag" matches "bag" / "handbag"
    but NOT "baggy" / "bagstrap". On other dialects we degrade to
    ILIKE — overmatch is tolerable in dev.
    """
    if _is_postgres_db():
        # \y is the Postgres word boundary. \mvariant\M would force a
        # full-word match (rejecting "handbag"); we want trailing
        # boundary only so suffixed compounds still hit.
        pat = _re.escape(variant) + r"\y"
        return [column.op("~*")(pat)]
    return [column.ilike(f"%{variant}%")]


# Columns the search hits. subniche is included here as BACKEND-ONLY
# search metadata: a search for 'earring' must match a product whose
# title is opaque ('Stud Studded Loops') but whose Gemini-assigned
# subniche is 'jewelry'. The user explicitly does NOT want subniche
# surfaced as a filter UI — see SUBNICHE_SYNONYMS above and the
# General-tab UI in index.html.
SEARCH_COLUMNS = (
    Product.title, Product.ai_tags, Product.product_type, Product.subniche,
)
AI_TAG_SEARCH_COLUMNS = (
    Product.ai_tags, Product.product_type, Product.subniche,
)


def _word_match_condition(word: str):
    """One OR-clause that matches `word` (any of its variants) against
    title, ai_tags, product_type, OR subniche. The product_type field
    carries Shopify's own categorisation (e.g. 'Women Handbags', 'Men
    Winter Coats'); subniche carries Gemini's high-level category
    label so that 'earring' can hit subniche='jewelry'.

    Variants under 3 characters are dropped — too short to match
    meaningfully and they cause runaway false positives.
    """
    variants = [v for v in expand_single_term(word) if len(v) >= 3]
    pieces = []
    for v in variants:
        for col in SEARCH_COLUMNS:
            pieces.extend(_match_clauses(col, v))
    return or_(*pieces) if pieces else None


def build_search_filters(search_query: str):
    """Smart search: returns (strict_AND_filters, loose_OR_filters).

    Strict: every word must match somewhere (title / ai_tags /
    product_type / subniche). Loose: any variant of any word matches
    anywhere.
    """
    words = [w for w in search_query.lower().split() if w]
    strict_conditions = []
    for w in words:
        cond = _word_match_condition(w)
        if cond is not None:
            strict_conditions.append(cond)

    loose_conditions = []
    for word in words:
        variants = [v for v in expand_single_term(word) if len(v) >= 3]
        for v in variants:
            for col in SEARCH_COLUMNS:
                loose_conditions.extend(_match_clauses(col, v))
    return strict_conditions, loose_conditions


def build_ai_tag_filters(search_query: str):
    """AND-of-words match across ai_tags + product_type + subniche.
    Each word is expanded via expand_single_term so naive plurals,
    multi-language translations, and subniche reverse-lookup
    ('earring' → 'jewelry') ALL participate on the primary search
    path — otherwise the fallback would miss subniche-only hits
    whenever the ai_tags query returns any unrelated results.
    """
    words = [w for w in search_query.lower().split() if w]
    conds = []
    for w in words:
        word_or = []
        for variant in expand_single_term(w):
            if len(variant) < 3:
                continue
            for col in AI_TAG_SEARCH_COLUMNS:
                word_or.extend(_match_clauses(col, variant))
        if word_or:
            conds.append(or_(*word_or))
    return conds

@app.get("/api/bestsellers/combined")
async def get_combined_bestsellers(
    sort: str = Query("high-low"),
    label: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    limit: int = Query(30, ge=1, le=500),
    db: Session = Depends(get_db),
):
    query = db.query(Product).join(Store).filter(Product.is_fashion == True)
    query = _apply_label_filter(query, db, label)

    if search and search.strip():
        s = search.strip()
        ai_query = query
        for cond in build_ai_tag_filters(s):
            ai_query = ai_query.filter(cond)
        ai_results = ai_query.all()
        if ai_results:
            products = ai_results
        else:
            strict, loose = build_search_filters(s)
            strict_query = query
            for cond in strict:
                strict_query = strict_query.filter(cond)
            strict_results = strict_query.all()
            if strict_results:
                products = strict_results
            else:
                products = query.filter(or_(*loose)).all() if loose else []
    else:
        products = query.all()

    if sort == "volume":
        products.sort(key=lambda p: (-parse_visitors(p.store.monthly_visitors), p.current_position))
    elif sort == "low-high":
        products.sort(key=lambda p: (-p.current_position,))
    else:
        products.sort(key=lambda p: p.current_position)

    label_map = _compute_label_map(db, products)
    products = _filter_products_by_computed_label(products, label_map, label)
    products = products[:limit]
    return [_product_dict(p, label_map.get(p.id)) for p in products]


@app.get("/api/bestsellers/store/{store_id}")
async def get_store_bestsellers(
    store_id: int,
    sort: str = Query("high-low"),
    label: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    limit: int = Query(30, ge=1, le=500),
    db: Session = Depends(get_db),
):
    store = db.query(Store).filter(Store.id == store_id).first()
    if not store:
        raise HTTPException(status_code=404, detail="Store not found")
    query = db.query(Product).filter(Product.store_id == store_id, Product.is_fashion == True)
    query = _apply_label_filter(query, db, label)
    if search and search.strip():
        s = search.strip()
        ai_query = query
        for cond in build_ai_tag_filters(s):
            ai_query = ai_query.filter(cond)
        ai_results = ai_query.order_by(Product.current_position).all()
        if ai_results:
            products = ai_results
        else:
            strict, loose = build_search_filters(s)
            strict_query = query
            for cond in strict:
                strict_query = strict_query.filter(cond)
            strict_results = strict_query.order_by(Product.current_position).all()
            if strict_results:
                products = strict_results
            elif loose:
                products = query.filter(or_(*loose)).order_by(Product.current_position).all()
            else:
                products = []
    else:
        products = query.order_by(Product.current_position).all()
    label_map = _compute_label_map(db, products)
    products = _filter_products_by_computed_label(products, label_map, label)
    products = products[:limit]
    return [_product_dict(p, label_map.get(p.id)) for p in products]

# =====================================================================
# Heroes / Villains — computed at READ time from PositionHistory.
# =====================================================================
# Source of truth for movement labels is the PositionHistory table, not
# the deprecated Product.label / Product.previous_position columns. The
# scraper writes a new history row every run; for any product the
# "today" snapshot is its latest history row and the "prior" snapshot
# is the most recent row dated:
#   - on a DIFFERENT UTC calendar day from today (excludes 12-h backup
#     scrapes and same-day manual scrapes), AND
#   - on or after TRUST_EPOCH_UTC (snapshots from before the most
#     recent breaking change are not trustworthy comparators).
# Day-over-day delta = current_position vs prior position. Brand-new
# products (no qualifying prior row) get the explicit label "new".
# A delta exceeding HERO_VILLAIN_DELTA_CAP — or 30% of the store's
# catalog size, whichever is smaller — is treated as a reshuffle, not
# organic movement, and labelled "normal" instead of hero/villain.

def _today_start_utc():
    """UTC midnight of today. Boundary between 'today's snapshot' and
    'prior snapshot' for the day-over-day delta. Same-UTC-day priors
    do NOT count, so the boundary excludes everything from 00:00 of
    today onward."""
    return datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)


def _trustworthy_prior_filters(today_start: datetime):
    """Two filter clauses every prior-position lookup must apply:
    (1) prior must come from a DIFFERENT UTC calendar day (i.e. before
        today's UTC midnight), AND
    (2) prior must be from on or after TRUST_EPOCH_UTC, since older
        snapshots reflect a different catalog and aren't comparable.
    """
    return (
        PositionHistory.date < today_start,
        PositionHistory.date >= TRUST_EPOCH_UTC,
    )


def _prior_position_subquery(db: Session, today_start: Optional[datetime] = None):
    """Subquery returning (product_id, prior_position): the most recent
    PositionHistory.position dated < today_start AND >= TRUST_EPOCH_UTC,
    per product. Uses ROW_NUMBER() so it works on both Postgres and
    SQLite (3.25+). The trust-epoch filter ensures pre-epoch snapshots
    are invisible to every caller.
    """
    today_start = today_start or _today_start_utc()
    inner = (
        db.query(
            PositionHistory.product_id.label("product_id"),
            PositionHistory.position.label("position"),
            func.row_number().over(
                partition_by=PositionHistory.product_id,
                order_by=PositionHistory.date.desc(),
            ).label("rn"),
        )
        .filter(*_trustworthy_prior_filters(today_start))
        .subquery()
    )
    return (
        db.query(
            inner.c.product_id.label("product_id"),
            inner.c.position.label("prior_position"),
        )
        .filter(inner.c.rn == 1)
        .subquery()
    )


def _delta_threshold(catalog_size: int) -> int:
    """Maximum plausible day-over-day rank delta for a store with
    `catalog_size` tracked products. The smaller of:
      - HERO_VILLAIN_DELTA_CAP (absolute, currently 30), and
      - 30% of catalog_size (rounded down, floored at 1).
    A delta exceeding this is almost certainly a structural reshuffle
    (catalog change, cap change, scrape-source change) and gets
    suppressed in _compute_label_map.
    """
    if catalog_size <= 0:
        return HERO_VILLAIN_DELTA_CAP
    pct = max(1, int(catalog_size * HERO_VILLAIN_CATALOG_FRACTION))
    return min(HERO_VILLAIN_DELTA_CAP, pct)


def _store_catalog_sizes(db: Session, store_ids: set, *, is_fashion: bool) -> dict:
    """Return {store_id: count of tracked products in that store on the
    relevant feed}. Used by _compute_label_map to size the per-store
    delta-magnitude threshold. Empty input returns {}.
    """
    if not store_ids:
        return {}
    q = db.query(Product.store_id, func.count(Product.id)).filter(
        Product.store_id.in_(store_ids)
    )
    if is_fashion:
        q = q.filter(Product.is_fashion == True)
    else:
        q = q.filter(Product.is_fashion == False, Product.subniche != "")
    rows = q.group_by(Product.store_id).all()
    return {sid: cnt for sid, cnt in rows}


def _compute_label_map(db: Session, products: list) -> dict:
    """For each product, compute (label, position_change, prior_position).

    label is one of:
      - "hero"    : moved up vs trustworthy prior snapshot (lower rank
                    number = better) AND delta within the per-store
                    threshold
      - "villain" : moved down vs trustworthy prior, delta in threshold
      - "normal"  : same rank as prior, OR delta exceeded the threshold
                    (treated as a reshuffle, not organic movement)
      - "new"     : no trustworthy prior snapshot exists yet for this
                    product (debut, or only same-day / pre-epoch priors)

    Returns {product_id: (label, position_change, prior_position)}.
    Matched by Product.id internally; tracked at the shopify_id level
    via the Product row's stable identity.
    """
    if not products:
        return {}
    pids = [p.id for p in products]
    today_start = _today_start_utc()
    rows = (
        db.query(PositionHistory.product_id, PositionHistory.position)
        .filter(PositionHistory.product_id.in_(pids))
        .filter(*_trustworthy_prior_filters(today_start))
        .order_by(
            PositionHistory.product_id,
            PositionHistory.date.desc(),
        )
        .all()
    )
    prior_map: dict = {}
    for pid, pos in rows:
        # First row per product_id is the most recent; subsequent rows
        # for the same product are older — ignore.
        if pid not in prior_map:
            prior_map[pid] = pos

    fashion_store_ids = {p.store_id for p in products if p.is_fashion}
    general_store_ids = {p.store_id for p in products if not p.is_fashion}
    fashion_sizes = _store_catalog_sizes(db, fashion_store_ids, is_fashion=True)
    general_sizes = _store_catalog_sizes(db, general_store_ids, is_fashion=False)

    out: dict = {}
    for p in products:
        prior = prior_map.get(p.id, 0)
        cur = p.current_position or 0
        if not prior:
            out[p.id] = ("new", 0, 0)
            continue
        if cur == prior:
            out[p.id] = ("normal", 0, prior)
            continue
        sizes = fashion_sizes if p.is_fashion else general_sizes
        threshold = _delta_threshold(sizes.get(p.store_id, 0))
        delta = abs(cur - prior)
        if delta > threshold:
            # Implausible jump — almost certainly a structural reshuffle
            # rather than organic movement. Demote to 'normal' so the
            # impossible 55→5 cases stop showing up as heroes.
            out[p.id] = ("normal", 0, prior)
            continue
        if cur < prior:
            out[p.id] = ("hero", prior - cur, prior)
        else:
            out[p.id] = ("villain", prior - cur, prior)  # negative magnitude
    return out


def _apply_label_filter(query, db: Session, label: str):
    """Restrict `query` to products whose dynamic label matches `label`.
    Pushes the trust-epoch + same-day-prior + absolute-delta-cap filters
    into SQL via a JOIN on the prior-position subquery so we don't have
    to fetch and scan in Python.

    NOTE: the per-store 30%-of-catalog tightening lives in
    _compute_label_map only; SQL applies the absolute 30-rank cap. The
    endpoints additionally post-filter by computed label so the per-
    store fraction is honoured for display correctness.
    """
    if not label or label == "all":
        return query
    sub = _prior_position_subquery(db)
    if label == "new":
        return query.outerjoin(sub, sub.c.product_id == Product.id).filter(
            sub.c.prior_position.is_(None)
        )
    query = query.join(sub, sub.c.product_id == Product.id)
    if label == "hero":
        return query.filter(
            Product.current_position < sub.c.prior_position,
            sub.c.prior_position - Product.current_position <= HERO_VILLAIN_DELTA_CAP,
        )
    if label == "villain":
        return query.filter(
            Product.current_position > sub.c.prior_position,
            Product.current_position - sub.c.prior_position <= HERO_VILLAIN_DELTA_CAP,
        )
    if label == "normal":
        # Display 'normal' covers both genuine no-movement AND demoted
        # large-delta items. Match unchanged-position rows here; the
        # endpoints' computed-label post-filter folds the demoted rows
        # in for an exact match.
        return query.filter(Product.current_position == sub.c.prior_position)
    return query


def _filter_products_by_computed_label(
    products: list, label_map: dict, label: Optional[str]
) -> list:
    """Drop products whose computed label disagrees with the requested
    label filter. This is the second pass — _apply_label_filter has
    already done the SQL-level coarse filter; this catches the per-
    store 30%-of-catalog tightening that SQL doesn't express.
    """
    if not label or label == "all":
        return products
    return [p for p in products if label_map.get(p.id, ("new", 0, 0))[0] == label]


def _product_dict(p, label_info=None):
    label, pos_diff, prior = label_info or ("new", 0, 0)
    return {
        "id": p.id, "title": p.title, "image_url": p.image_url,
        "price": p.price, "vendor": p.vendor, "product_url": p.product_url,
        "current_position": p.current_position,
        "previous_position": prior,
        "position_change": pos_diff,
        "label": label,
        "ai_tags": p.ai_tags or "",
        "is_fashion": bool(p.is_fashion),
        "subniche": p.subniche or "",
        "store_name": p.store.name, "store_url": p.store.url,
        "store_visitors": p.store.monthly_visitors,
        "last_scraped": p.last_scraped.isoformat() if p.last_scraped else None,
    }


def _general_base_query(db: Session, store_id: Optional[int] = None):
    q = db.query(Product).join(Store).filter(
        Product.is_fashion == False,
        Product.subniche != "",
    )
    if store_id is not None:
        q = q.filter(Product.store_id == store_id)
    return q


@app.get("/api/general/combined")
async def get_combined_general(
    sort: str = Query("high-low"),
    label: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    limit: int = Query(30, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """Cross-store General feed. Same shape as /api/bestsellers/combined
    but filtered to is_fashion=False with a non-empty subniche so we
    only surface products Gemini actually labelled for this tab.
    `subniche` is BACKEND-only metadata used by the search box; it is
    never exposed as a user-facing filter parameter (see SUBNICHE_SYNONYMS).
    """
    query = _general_base_query(db)
    query = _apply_label_filter(query, db, label)

    if search and search.strip():
        s = search.strip()
        ai_query = query
        for cond in build_ai_tag_filters(s):
            ai_query = ai_query.filter(cond)
        ai_results = ai_query.all()
        if ai_results:
            products = ai_results
        else:
            strict, loose = build_search_filters(s)
            strict_query = query
            for cond in strict:
                strict_query = strict_query.filter(cond)
            strict_results = strict_query.all()
            if strict_results:
                products = strict_results
            else:
                products = query.filter(or_(*loose)).all() if loose else []
    else:
        products = query.all()

    if sort == "volume":
        products.sort(key=lambda p: (-parse_visitors(p.store.monthly_visitors), p.current_position))
    elif sort == "low-high":
        products.sort(key=lambda p: (-p.current_position,))
    else:
        products.sort(key=lambda p: p.current_position)

    label_map = _compute_label_map(db, products)
    products = _filter_products_by_computed_label(products, label_map, label)
    products = products[:limit]
    return [_product_dict(p, label_map.get(p.id)) for p in products]


@app.get("/api/general/store/{store_id}")
async def get_store_general(
    store_id: int,
    sort: str = Query("high-low"),
    label: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    limit: int = Query(30, ge=1, le=500),
    db: Session = Depends(get_db),
):
    store = db.query(Store).filter(Store.id == store_id).first()
    if not store:
        raise HTTPException(status_code=404, detail="Store not found")
    query = _general_base_query(db, store_id=store_id)
    query = _apply_label_filter(query, db, label)
    if search and search.strip():
        s = search.strip()
        ai_query = query
        for cond in build_ai_tag_filters(s):
            ai_query = ai_query.filter(cond)
        ai_results = ai_query.order_by(Product.current_position).all()
        if ai_results:
            products = ai_results
        else:
            strict, loose = build_search_filters(s)
            strict_query = query
            for cond in strict:
                strict_query = strict_query.filter(cond)
            strict_results = strict_query.order_by(Product.current_position).all()
            if strict_results:
                products = strict_results
            elif loose:
                products = query.filter(or_(*loose)).order_by(Product.current_position).all()
            else:
                products = []
    else:
        products = query.order_by(Product.current_position).all()
    label_map = _compute_label_map(db, products)
    products = _filter_products_by_computed_label(products, label_map, label)
    products = products[:limit]
    return [_product_dict(p, label_map.get(p.id)) for p in products]


# --- Scrape triggers ---
@app.post("/api/scrape")
async def trigger_scrape():
    """Kick off a full scrape in the background and return immediately.

    A full scrape across 12 stores takes longer than the Railway HTTP
    timeout, so we cannot await it inline. The frontend should poll
    /api/scrape/status to learn when the run finishes.
    """
    if _scrape_state["running"]:
        return {
            "started": False,
            "running": True,
            "started_at": _scrape_state["started_at"],
            "message": "Scrape already in progress",
        }
    _scrape_state["running"] = True
    _scrape_state["started_at"] = datetime.utcnow().isoformat()
    _scrape_state["result"] = None
    asyncio.create_task(_background_scrape_all())
    return {
        "started": True,
        "running": True,
        "started_at": _scrape_state["started_at"],
        "message": "Scrape started in background. Poll /api/scrape/status for progress.",
    }


@app.get("/api/scrape/status")
async def scrape_status():
    return {
        "running": _scrape_state["running"],
        "started_at": _scrape_state["started_at"],
        "result": _scrape_state["result"],
    }

@app.post("/api/reset-labels")
async def trigger_reset_labels(db: Session = Depends(get_db)):
    count = reset_all_labels(db)
    return {"success": True, "products_reset": count}

@app.post("/api/scrape/{store_id}")
async def trigger_store_scrape(store_id: int, db: Session = Depends(get_db)):
    store = db.query(Store).filter(Store.id == store_id).first()
    if not store:
        raise HTTPException(status_code=404, detail="Store not found")
    fashion, general, errors = await scrape_store_bestsellers(store.url)
    if fashion or general:
        update_products_in_db(db, store, fashion, general)
    return {
        "success": (len(fashion) + len(general)) > 0,
        "products_found": len(fashion),
        "general_found": len(general),
        "errors": errors,
    }


@app.get("/api/debug/fetch/{store_id}")
async def debug_fetch_store(store_id: int, db: Session = Depends(get_db)):
    """Diagnostic: fetch one collection page and report what we see."""
    store = db.query(Store).filter(Store.id == store_id).first()
    if not store:
        raise HTTPException(status_code=404, detail="Store not found")
    return await debug_fetch(store.url)


@app.get("/api/debug/env")
async def debug_env():
    """Report which env-derived behavior is active without leaking secrets."""
    return {
        "gemini_key_set": bool(os.getenv("GEMINI_API_KEY")),
        "database_url_kind": (
            "postgres" if os.getenv("DATABASE_URL", "").startswith(("postgres://", "postgresql://"))
            else "sqlite" if not os.getenv("DATABASE_URL") or "sqlite" in os.getenv("DATABASE_URL", "")
            else "other"
        ),
    }


@app.get("/api/debug/gemini")
async def debug_gemini():
    """One-shot Gemini call against a synthetic 2-item batch.

    Returns the actual outcome so we can diagnose auth / model / schema
    issues without waiting for a multi-minute full scrape. Never echoes
    the API key.
    """
    from classifier import classify_products_batch
    sample = [
        {"shopify_id": "x1", "handle": "x1", "title": "Floral summer maxi dress",
         "vendor": "Test", "product_type": "", "is_fashion": None, "ai_tags": ""},
        {"shopify_id": "x2", "handle": "x2", "title": "Shipping protection",
         "vendor": "Test", "product_type": "", "is_fashion": None, "ai_tags": ""},
    ]
    errors = await classify_products_batch(sample)
    return {
        "gemini_key_set": bool(os.getenv("GEMINI_API_KEY")),
        "errors": errors,
        "results": [
            {"handle": p["handle"], "is_fashion": p.get("is_fashion"),
             "ai_tags": p.get("ai_tags", "")}
            for p in sample
        ],
    }

@app.get("/api/stats")
async def get_stats(db: Session = Depends(get_db)):
    """Heroes/villains counted dynamically from the prior-position
    subquery — no reliance on the deprecated Product.label column.
    Both totals scope to the Fashion feed (is_fashion=True). Products
    with no TRUSTWORTHY prior snapshot (i.e. dated before today UTC and
    on/after TRUST_EPOCH_UTC) contribute to neither, and products
    whose delta exceeds HERO_VILLAIN_DELTA_CAP are filtered out as
    suspect (almost certainly a structural reshuffle, not movement).
    """
    total_stores = db.query(Store).count()
    total_products = db.query(Product).filter(Product.is_fashion == True).count()
    sub = _prior_position_subquery(db)
    heroes = (
        db.query(Product)
        .join(sub, sub.c.product_id == Product.id)
        .filter(
            Product.is_fashion == True,
            Product.current_position < sub.c.prior_position,
            sub.c.prior_position - Product.current_position <= HERO_VILLAIN_DELTA_CAP,
        )
        .count()
    )
    villains = (
        db.query(Product)
        .join(sub, sub.c.product_id == Product.id)
        .filter(
            Product.is_fashion == True,
            Product.current_position > sub.c.prior_position,
            Product.current_position - sub.c.prior_position <= HERO_VILLAIN_DELTA_CAP,
        )
        .count()
    )
    return {"stores": total_stores, "products": total_products, "heroes": heroes, "villains": villains}

# --- Serve Frontend ---
frontend_path = os.path.dirname(__file__)

@app.get("/")
async def serve_root():
    index_path = os.path.join(frontend_path, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>Spy Wizard API</h1>")

@app.get("/{path:path}")
async def serve_static(path: str):
    file_path = os.path.join(frontend_path, path)
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return FileResponse(file_path)
    index_path = os.path.join(frontend_path, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404)
