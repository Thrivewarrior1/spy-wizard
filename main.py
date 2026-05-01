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
    # Count only fashion-flagged products. Non-fashion rows (shipping
    # protection, gift cards, retired stale entries) stay in the table
    # for history but must not inflate the per-store count the user sees.
    return [{
        "id": s.id, "name": s.name, "url": s.url,
        "monthly_visitors": s.monthly_visitors, "niche": s.niche,
        "country": s.country,
        "product_count": sum(1 for p in s.products if p.is_fashion),
        "last_scraped": max((p.last_scraped for p in s.products if p.is_fashion), default=None),
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


def _word_match_condition(word: str):
    """One OR-clause that matches `word` (any of its variants) against
    title, ai_tags, OR product_type. The product_type field carries
    Shopify's own categorisation (e.g. 'Women Handbags', 'Men Winter
    Coats') which is why a search for 'Women Bags' should hit it even
    when the user-facing title doesn't say 'women'.
    """
    variants = expand_single_term(word)
    pieces = []
    for v in variants:
        pat = f"%{v}%"
        pieces.append(Product.title.ilike(pat))
        pieces.append(Product.ai_tags.ilike(pat))
        pieces.append(Product.product_type.ilike(pat))
    return or_(*pieces)


def build_search_filters(search_query: str):
    """Smart search: returns (strict_AND_filters, loose_OR_filters).

    Strict: every word must match somewhere (title / ai_tags /
    product_type). Loose: any variant of any word matches anywhere.
    """
    words = [w for w in search_query.lower().split() if w]
    strict_conditions = [_word_match_condition(w) for w in words]
    loose_conditions = []
    for word in words:
        variants = expand_single_term(word)
        for v in variants:
            pat = f"%{v}%"
            loose_conditions.append(Product.title.ilike(pat))
            loose_conditions.append(Product.ai_tags.ilike(pat))
            loose_conditions.append(Product.product_type.ilike(pat))
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
            pat = f"%{variant}%"
            word_or.append(Product.ai_tags.ilike(pat))
            word_or.append(Product.product_type.ilike(pat))
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
    if label and label != "all":
        query = query.filter(Product.label == label)

    if search and search.strip():
        s = search.strip()
        # Try AI tags first (AND of search words)
        ai_query = query
        for cond in build_ai_tag_filters(s):
            ai_query = ai_query.filter(cond)
        ai_results = ai_query.all()
        if ai_results:
            products = ai_results
        else:
            # Fall back to title-based strict-then-loose
            strict, loose = build_search_filters(s)
            strict_query = query
            for cond in strict:
                strict_query = strict_query.filter(cond)
            strict_results = strict_query.all()
            if strict_results:
                products = strict_results
            else:
                products = query.filter(or_(*loose)).all()
    else:
        products = query.all()

    if sort == "volume":
        products.sort(key=lambda p: (-parse_visitors(p.store.monthly_visitors), p.current_position))
    elif sort == "low-high":
        products.sort(key=lambda p: (-p.current_position,))
    else:
        products.sort(key=lambda p: p.current_position)

    return [_product_dict(p) for p in products[:limit]]

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
    if label and label != "all":
        query = query.filter(Product.label == label)
    if search and search.strip():
        s = search.strip()
        # Try AI tags first (AND of search words)
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
            else:
                products = query.filter(or_(*loose)).order_by(Product.current_position).limit(limit).all()
    else:
        products = query.order_by(Product.current_position).limit(limit).all()
    return [_product_dict(p) for p in products]

def _product_dict(p):
    pos_diff = 0
    if p.previous_position > 0:
        pos_diff = p.previous_position - p.current_position
    return {
        "id": p.id, "title": p.title, "image_url": p.image_url,
        "price": p.price, "vendor": p.vendor, "product_url": p.product_url,
        "current_position": p.current_position,
        "previous_position": p.previous_position,
        "position_change": pos_diff,
        "label": p.label,
        "ai_tags": p.ai_tags or "",
        "is_fashion": bool(p.is_fashion),
        "store_name": p.store.name, "store_url": p.store.url,
        "store_visitors": p.store.monthly_visitors,
        "last_scraped": p.last_scraped.isoformat() if p.last_scraped else None,
    }

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
    products, errors = await scrape_store_bestsellers(store.url)
    if products:
        update_products_in_db(db, store, products)
    return {"success": len(products) > 0, "products_found": len(products), "errors": errors}


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
    total_stores = db.query(Store).count()
    total_products = db.query(Product).filter(Product.is_fashion == True).count()
    heroes = db.query(Product).filter(
        Product.label == "hero", Product.is_fashion == True
    ).count()
    villains = db.query(Product).filter(
        Product.label == "villain", Product.is_fashion == True
    ).count()
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
