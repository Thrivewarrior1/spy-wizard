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
)
from seed import seed_stores

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APP_PASSWORD = os.getenv("APP_PASSWORD", "mats2310")
PORT = int(os.getenv("PORT", "8000"))

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


def expand_single_term(term: str) -> list:
    """Expand a single search term to include its translations and a
    naive singular form. 'bags' -> {bags, bag, tasche, sac, ...},
    'women' -> {women, woman, damen, femme, ...}.
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


def _word_match_condition(word: str):
    """One OR-clause that matches `word` (any of its variants) against
    title, ai_tags, OR product_type. The product_type field carries
    Shopify's own categorisation (e.g. 'Women Handbags', 'Men Winter
    Coats') which is why a search for 'Women Bags' should hit it even
    when the user-facing title doesn't say 'women'.

    Variants under 3 characters are dropped — too short to match
    meaningfully and they cause runaway false positives.
    """
    variants = [v for v in expand_single_term(word) if len(v) >= 3]
    pieces = []
    for v in variants:
        for col in (Product.title, Product.ai_tags, Product.product_type):
            pieces.extend(_match_clauses(col, v))
    return or_(*pieces) if pieces else None


def build_search_filters(search_query: str):
    """Smart search: returns (strict_AND_filters, loose_OR_filters).

    Strict: every word must match somewhere (title / ai_tags /
    product_type). Loose: any variant of any word matches anywhere.
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
            for col in (Product.title, Product.ai_tags, Product.product_type):
                loose_conditions.extend(_match_clauses(col, v))
    return strict_conditions, loose_conditions


def build_ai_tag_filters(search_query: str):
    """AND-of-words match across ai_tags + product_type, with naive
    singularisation so 'bags' matches both 'bag' and 'bags' tags.
    """
    words = [w for w in search_query.lower().split() if w]
    conds = []
    for w in words:
        word_or = []
        for variant in _singularize(w):
            if len(variant) < 3:
                continue
            for col in (Product.ai_tags, Product.product_type):
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

    products = products[:limit]
    label_map = _compute_label_map(db, products)
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
        ai_results = ai_query.order_by(Product.current_position).limit(limit).all()
        if ai_results:
            products = ai_results
        else:
            strict, loose = build_search_filters(s)
            strict_query = query
            for cond in strict:
                strict_query = strict_query.filter(cond)
            strict_results = strict_query.order_by(Product.current_position).limit(limit).all()
            if strict_results:
                products = strict_results
            elif loose:
                products = query.filter(or_(*loose)).order_by(Product.current_position).limit(limit).all()
            else:
                products = []
    else:
        products = query.order_by(Product.current_position).limit(limit).all()
    label_map = _compute_label_map(db, products)
    return [_product_dict(p, label_map.get(p.id)) for p in products]

# =====================================================================
# Heroes / Villains — computed at READ time from PositionHistory.
# =====================================================================
# Source of truth for movement labels is the PositionHistory table, not
# the deprecated Product.label / Product.previous_position columns. The
# scraper writes a new history row every run; for any product the
# "today" snapshot is its latest history row and the "prior" snapshot
# is the most recent row dated before today's UTC midnight. Day-over-
# day delta = current_position vs prior position. Brand-new products
# (no row dated < today_start) get the explicit label "new".

def _today_start_utc():
    """UTC midnight of today. Boundary between 'today's snapshot' and
    'prior snapshot' for the day-over-day delta."""
    return datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)


def _prior_position_subquery(db: Session, today_start: Optional[datetime] = None):
    """Subquery returning (product_id, prior_position): the most recent
    PositionHistory.position dated < today_start, per product. Uses
    ROW_NUMBER() so it works on both Postgres and SQLite (3.25+).
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
        .filter(PositionHistory.date < today_start)
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


def _compute_label_map(db: Session, products: list) -> dict:
    """For each product, compute (label, position_change, prior_position).

    label is one of:
      - "hero"    : moved up vs prior snapshot (lower rank number = better)
      - "villain" : moved down vs prior snapshot
      - "normal"  : same rank as prior snapshot
      - "new"     : no prior snapshot dated < today (debut, no compare)

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
        .filter(PositionHistory.date < today_start)
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

    out: dict = {}
    for p in products:
        prior = prior_map.get(p.id, 0)
        cur = p.current_position or 0
        if not prior:
            out[p.id] = ("new", 0, 0)
        elif cur < prior:
            out[p.id] = ("hero", prior - cur, prior)
        elif cur > prior:
            out[p.id] = ("villain", prior - cur, prior)  # negative magnitude
        else:
            out[p.id] = ("normal", 0, prior)
    return out


def _apply_label_filter(query, db: Session, label: str):
    """Restrict `query` to products whose dynamic label matches `label`.
    Pushes the filter into SQL via a JOIN on the prior-position
    subquery so we don't have to fetch and scan in Python.
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
        return query.filter(Product.current_position < sub.c.prior_position)
    if label == "villain":
        return query.filter(Product.current_position > sub.c.prior_position)
    if label == "normal":
        return query.filter(Product.current_position == sub.c.prior_position)
    return query


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


# Allowed subniche labels for the General feed filter pills. Mirrors the
# enum in classifier.py so the UI surfaces the same vocabulary.
GENERAL_SUBNICHES = (
    "jewelry", "accessories", "electronics", "home", "beauty",
    "health", "food", "toys-books", "services", "other",
)


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
    subniche: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    limit: int = Query(30, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """Cross-store General feed. Same shape as /api/bestsellers/combined
    but filtered to is_fashion=False with a non-empty subniche so we
    only surface products Gemini actually labelled for this tab.
    """
    query = _general_base_query(db)
    query = _apply_label_filter(query, db, label)
    if subniche and subniche != "all":
        query = query.filter(Product.subniche == subniche)

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

    products = products[:limit]
    label_map = _compute_label_map(db, products)
    return [_product_dict(p, label_map.get(p.id)) for p in products]


@app.get("/api/general/store/{store_id}")
async def get_store_general(
    store_id: int,
    sort: str = Query("high-low"),
    label: Optional[str] = Query(None),
    subniche: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    limit: int = Query(30, ge=1, le=500),
    db: Session = Depends(get_db),
):
    store = db.query(Store).filter(Store.id == store_id).first()
    if not store:
        raise HTTPException(status_code=404, detail="Store not found")
    query = _general_base_query(db, store_id=store_id)
    query = _apply_label_filter(query, db, label)
    if subniche and subniche != "all":
        query = query.filter(Product.subniche == subniche)
    if search and search.strip():
        s = search.strip()
        ai_query = query
        for cond in build_ai_tag_filters(s):
            ai_query = ai_query.filter(cond)
        ai_results = ai_query.order_by(Product.current_position).limit(limit).all()
        if ai_results:
            products = ai_results
        else:
            strict, loose = build_search_filters(s)
            strict_query = query
            for cond in strict:
                strict_query = strict_query.filter(cond)
            strict_results = strict_query.order_by(Product.current_position).limit(limit).all()
            if strict_results:
                products = strict_results
            elif loose:
                products = query.filter(or_(*loose)).order_by(Product.current_position).limit(limit).all()
            else:
                products = []
    else:
        products = query.order_by(Product.current_position).limit(limit).all()
    label_map = _compute_label_map(db, products)
    return [_product_dict(p, label_map.get(p.id)) for p in products]


@app.get("/api/general/subniches")
async def get_general_subniches(db: Session = Depends(get_db)):
    """Return the set of subniches that actually have at least one
    product in the DB right now, with counts. Powers the filter pills.
    """
    rows = (
        db.query(Product.subniche, func.count(Product.id))
        .filter(Product.is_fashion == False, Product.subniche != "")
        .group_by(Product.subniche)
        .all()
    )
    return [{"subniche": s, "count": c} for s, c in rows if s in GENERAL_SUBNICHES]

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
    with no prior snapshot before today contribute to neither.
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
        )
        .count()
    )
    villains = (
        db.query(Product)
        .join(sub, sub.c.product_id == Product.id)
        .filter(
            Product.is_fashion == True,
            Product.current_position > sub.c.prior_position,
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
