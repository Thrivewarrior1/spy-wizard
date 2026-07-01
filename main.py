import os
import asyncio
import logging
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, or_
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from pydantic import BaseModel
from typing import Optional

from database import (
    get_db, engine, SessionLocal,
    widen_text_columns,
    enforce_fk_cascade,
    cleanup_orphans,
)
from models import Base, Store, Product, PositionHistory
from scraper import (
    scrape_all_stores,
    scrape_store_bestsellers,
    update_products_in_db,
    reset_all_labels,
    debug_fetch,
    cleanup_non_product_rows,
    migrate_wearables_to_fashion,
    migrate_apparel_to_fashion,
    migrate_force_general_to_general,
    migrate_drop_off_cap_positions,
    migrate_backfill_product_category,
)
from categories import (
    assign_product_category,
    lookup_categories_for_query_token,
    is_category_token,
    PRODUCT_CATEGORIES,
    PARENTS_TO_CHILDREN,
    TOKEN_TO_CATEGORIES,
    ALL_CATEGORY_NAMES,
)
from seed import seed_stores

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APP_PASSWORD = os.getenv("APP_PASSWORD", "mats2310")
PORT = int(os.getenv("PORT", "8000"))


# =====================================================================
# Trust epoch + delta-cap constants live in labels.py so both the read-
# time API and the scrape-time event writer use the SAME values. The
# imports below re-export them at module level so the rest of main.py
# can keep using the bare names.
# =====================================================================
from labels import (
    TRUST_EPOCH_UTC,
    HERO_VILLAIN_DELTA_CAP,
    HERO_VILLAIN_CATALOG_FRACTION,
    LABEL_EVENT_RETENTION_DAYS,
    DATA_START_DATE,
    today_start_utc as _today_start_utc,
    trustworthy_prior_filters as _trustworthy_prior_filters,
    delta_threshold as _delta_threshold,
    compute_and_write_events,
    cleanup_label_events,
    cleanup_pre_start_label_events,
    backfill_label_events,
    fetch_label_events_window,
)
from models import LabelEvent


def _trust_epoch_invariant_check() -> Optional[str]:
    """Return an error message if TRUST_EPOCH_UTC is at or past today's
    00:00 UTC — that's the misconfiguration that silently breaks the
    hero/villain query (mutually-exclusive prior filters). Called at
    startup and exposed on /api/debug/heroes so the failure mode is
    loud rather than a silent 0/0.
    """
    today_start = _today_start_utc()
    if TRUST_EPOCH_UTC >= today_start:
        return (
            f"TRUST_EPOCH_UTC ({TRUST_EPOCH_UTC.isoformat()}) is at or past "
            f"today's UTC midnight ({today_start.isoformat()}). The "
            f"hero/villain query requires "
            f"prior_date < today_start AND prior_date >= TRUST_EPOCH_UTC; "
            f"those clauses are mutually exclusive when the epoch == today. "
            f"Roll the epoch back to a date that is strictly before today."
        )
    return None

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
    # Upgrade legacy foreign-key constraints to ON DELETE CASCADE so
    # deleting a Store actually wipes its Products / PositionHistory /
    # LabelEvent rows at the DB layer — without this, the user's
    # delete leaks orphan rows that continue to show in the dashboard.
    # Idempotent: a no-op after the first successful upgrade.
    enforce_fk_cascade()
    # Belt-and-braces: even after CASCADE is in place, mop up any
    # orphans that pre-date the migration so the dashboard stops
    # surfacing products from already-deleted stores.
    cleanup_orphans()
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
    try:
        # Sister migration: promote any General-tab row whose title /
        # product_type / handle / image_url matches the multilingual
        # apparel/footwear/eyewear/intimates allowlist. Catches Gemini
        # mis-routing of Bademantel→home, Unterwäsche→beauty,
        # Orthoschuh→health, wedding-guest dresses→other.
        migrate_apparel_to_fashion(db)
    except Exception as e:
        logger.warning(f"startup apparel migration failed: {e}")
    try:
        # Function-over-form sweep: demote any Fashion-tab row that's
        # actually a wearable gadget (smartwatch, posture corrector,
        # dog raincoat, magnifying glass, trekking pole, etc.). Runs
        # AFTER the apparel migration so anything legitimately on
        # Fashion stays on Fashion, and only the function-driven
        # items get pulled back to General.
        migrate_force_general_to_general(db)
    except Exception as e:
        logger.warning(f"startup force-general migration failed: {e}")
    try:
        # Off-cap sweep: any row whose current_position exceeds
        # MAX_SOURCE_POSITION is from a pre-cap scrape and the new
        # logic can never refresh it. Retire unconditionally so the
        # Fashion / General feeds don't surface zombie rows from the
        # deep catalog tail (Breuermode at #1614 etc.).
        migrate_drop_off_cap_positions(db)
    except Exception as e:
        logger.warning(f"startup off-cap migration failed: {e}")
    try:
        # Backfill the new product_category column on every existing
        # row using the curated regex catalog. Cheap (no Gemini calls)
        # and idempotent — runs on every redeploy but only assigns to
        # rows that don't already have a category.
        migrate_backfill_product_category(db)
    except Exception as e:
        logger.warning(f"startup product_category backfill failed: {e}")
    try:
        # Drop any pre-DATA_START_DATE events first. These are
        # pre-Spy-Wizard-2 events that look real but reflect a
        # different catalog shape and the user explicitly excluded
        # them. Idempotent.
        pre_start_pruned = cleanup_pre_start_label_events(db)
        if pre_start_pruned:
            logger.info(
                f"startup pre-start cleanup: dropped {pre_start_pruned} "
                f"events with date < {DATA_START_DATE.isoformat()}"
            )
        # Backfill the LabelEvent ledger from existing PositionHistory.
        # User asked for 30-day retained heroes/villains; this gives the
        # UI immediate historical data instead of waiting for new
        # scrapes to populate. Idempotent — skips events that already
        # exist for a given (product_id, date). Respects DATA_START_DATE.
        inserted = backfill_label_events(db)
        if inserted:
            logger.info(
                f"startup label-event backfill: inserted {inserted} events "
                f"from existing PositionHistory"
            )
        # Run retention immediately so any pre-existing >30d events get
        # dropped on the same redeploy.
        pruned = cleanup_label_events(db)
        if pruned:
            logger.info(f"startup label-event retention: pruned {pruned}")
    except Exception as e:
        logger.warning(f"startup label-event backfill / retention failed: {e}")
    finally:
        db.close()
    # Schedule daily scrape at 6 AM UTC
    scheduler.add_job(daily_scrape, "cron", hour=6, minute=0, id="daily_scrape")
    # Also scrape every 12 hours as backup
    scheduler.add_job(daily_scrape, "interval", hours=12, id="backup_scrape")
    scheduler.start()
    logger.info("Scheduler started - daily scrape at 06:00 UTC + every 12h backup")
    # Loud failure mode if the trust-epoch was bumped to >= today_start.
    # Without this, hero/villain queries silently return 0/0 because the
    # `prior_date < today_start AND prior_date >= TRUST_EPOCH_UTC` filters
    # become mutually exclusive. We've hit this regression twice now.
    epoch_problem = _trust_epoch_invariant_check()
    if epoch_problem:
        logger.error("TRUST EPOCH INVARIANT VIOLATED — %s", epoch_problem)
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


def _normalise_store_url(raw: str) -> str:
    """Make slight differences hash to the same store.

      - strip leading/trailing whitespace
      - prepend https:// when no scheme is given
      - **strip the URL path entirely** so we always end up with the
        bare origin (https://shop.example), never a per-collection
        URL like https://shop.example/collections/bags. The scraper
        appends `/collections/<slug>?sort_by=best-selling&page=N` on
        top of whatever's stored, so a path here turns into
        `/collections/bags/collections/all?...` which 404s. We treat
        the path as junk the user copy-pasted, not as authoritative.
      - strip a trailing slash so 'novigood.com/' and 'novigood.com'
        collide on the unique-URL check
    """
    if not raw:
        return ""
    u = raw.strip()
    if not u:
        return ""
    if not u.startswith(("http://", "https://")):
        u = "https://" + u
    # Strip everything after the host: path, query, fragment. The
    # scraper builds its own /collections/all?sort_by=best-selling URL
    # from this origin so anything beyond the host is noise (and a
    # bug magnet — see Oliva Mode 2026-05-30: the user pasted
    # https://olivamode.com/collections/bags and the scraper turned
    # it into a 404).
    scheme, rest = u.split("://", 1)
    host = rest.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
    u = f"{scheme}://{host}"
    if u.endswith("/"):
        u = u[:-1]
    return u


def _validate_store_payload(name: str, url: str) -> Optional[str]:
    """Return None if the (name, url) pair is acceptable, else a
    human-readable error string. The string is surfaced as the 400
    detail so the UI can show it instead of a generic 'Failed to save'."""
    if not name or not name.strip():
        return "Store name is required"
    if len(name.strip()) > 255:
        return "Store name must be 255 characters or fewer"
    if not url or not url.strip():
        return "Store URL is required"
    cleaned = _normalise_store_url(url)
    if not cleaned.startswith(("http://", "https://")):
        return "Store URL must start with http:// or https://"
    # Reject obviously broken URLs that survive normalisation —
    # 'https://' alone, or hosts without a dot.
    host_part = cleaned.split("://", 1)[1].split("/", 1)[0]
    if not host_part or "." not in host_part:
        return "Store URL host looks invalid (missing TLD?)"
    return None

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

@app.post("/api/stores", status_code=201)
async def create_store(store: StoreCreate, db: Session = Depends(get_db)):
    """Create a new competitor store.

    Returns:
      201 + {id, name, url, niche, country, monthly_visitors} on success
      400 with `detail` when name / url are missing or malformed
      409 with `detail` when a store with the same normalised URL
          already exists (the previous code returned 400 with the
          generic message "Store already exists" which is technically
          a Conflict, not a Bad Request — frontend was unable to
          distinguish duplicate from validation error).
    """
    err = _validate_store_payload(store.name, store.url)
    if err:
        raise HTTPException(status_code=400, detail=err)

    cleaned_url = _normalise_store_url(store.url)
    # Match against the normalised URL on both sides so trailing-slash
    # / scheme differences still trip the dedupe.
    existing = next(
        (s for s in db.query(Store).all()
         if _normalise_store_url(s.url or "") == cleaned_url),
        None,
    )
    if existing:
        raise HTTPException(
            status_code=409,
            detail=(
                f"A store with this URL already exists "
                f"(id={existing.id}, name={existing.name!r})"
            ),
        )

    payload = store.model_dump()
    payload["name"] = payload["name"].strip()
    payload["url"] = cleaned_url
    new_store = Store(**payload)
    db.add(new_store)
    db.commit()
    db.refresh(new_store)
    return {
        "id": new_store.id, "name": new_store.name, "url": new_store.url,
        "niche": new_store.niche, "country": new_store.country,
        "monthly_visitors": new_store.monthly_visitors,
    }

@app.put("/api/stores/{store_id}")
async def update_store(store_id: int, store: StoreUpdate, db: Session = Depends(get_db)):
    """Update a store's metadata. Every submitted (non-None) field is
    persisted. URL is re-normalised on save so trailing-slash /
    scheme drift can't accumulate.

    Returns the FULL updated row so the frontend can refresh its
    cache without an extra round-trip — and `loadCombined` /
    `loadCombinedGeneral` pick up the new store_name / store_url
    on the next render of any product card.
    """
    existing = db.query(Store).filter(Store.id == store_id).first()
    if not existing:
        raise HTTPException(status_code=404, detail="Store not found")
    payload = store.model_dump(exclude_none=True)
    # Validate any submitted fields. A PUT can patch a single field,
    # so only validate fields that were actually sent.
    if "name" in payload and (not payload["name"] or not payload["name"].strip()):
        raise HTTPException(status_code=400, detail="Store name cannot be empty")
    if "url" in payload:
        if not payload["url"] or not payload["url"].strip():
            raise HTTPException(status_code=400, detail="Store URL cannot be empty")
        # Validate URL shape against the same rules as create.
        err = _validate_store_payload(payload.get("name") or existing.name, payload["url"])
        if err and "url" in err.lower():
            raise HTTPException(status_code=400, detail=err)
        cleaned_url = _normalise_store_url(payload["url"])
        # Conflict check — a different store already owns this URL.
        clash = next(
            (s for s in db.query(Store).all()
             if s.id != store_id
             and _normalise_store_url(s.url or "") == cleaned_url),
            None,
        )
        if clash:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Another store with this URL already exists "
                    f"(id={clash.id}, name={clash.name!r})"
                ),
            )
        payload["url"] = cleaned_url
    if "name" in payload:
        payload["name"] = payload["name"].strip()
    for key, value in payload.items():
        setattr(existing, key, value)
    db.commit()
    db.refresh(existing)
    logger.info(
        "update_store: id=%d name=%r url=%r niche=%r country=%r visitors=%r",
        existing.id, existing.name, existing.url, existing.niche,
        existing.country, existing.monthly_visitors,
    )
    return {
        "success": True,
        "id": existing.id,
        "name": existing.name,
        "url": existing.url,
        "monthly_visitors": existing.monthly_visitors,
        "niche": existing.niche,
        "country": existing.country,
    }

@app.delete("/api/stores/{store_id}")
async def delete_store(store_id: int, db: Session = Depends(get_db)):
    """Delete a store AND every row that referenced it.

    Explicit cascade rather than relying solely on the DB-level
    ON DELETE CASCADE — that constraint is upgraded by the startup
    `enforce_fk_cascade()` migration, but if for any reason it
    didn't fire (table didn't exist yet, permission error, etc.)
    we still want the delete to wipe the whole tree.

    Response includes per-table delete counts so the UI can show
    the user exactly what was removed.
    """
    store = db.query(Store).filter(Store.id == store_id).first()
    if not store:
        raise HTTPException(status_code=404, detail="Store not found")
    name_before = store.name
    # Snapshot the product ids so the explicit child deletes hit a
    # stable set (the DB-level cascade may or may not have already
    # purged them by the time we run these queries).
    product_ids = [pid for (pid,) in db.query(Product.id).filter(
        Product.store_id == store_id,
    ).all()]
    counts = {"store": 0, "products": 0, "position_history": 0, "label_events": 0}
    if product_ids:
        counts["label_events"] = db.query(LabelEvent).filter(
            LabelEvent.product_id.in_(product_ids)
        ).delete(synchronize_session=False)
        counts["position_history"] = db.query(PositionHistory).filter(
            PositionHistory.product_id.in_(product_ids)
        ).delete(synchronize_session=False)
        counts["products"] = db.query(Product).filter(
            Product.id.in_(product_ids)
        ).delete(synchronize_session=False)
    # Also purge any LabelEvent rows tied to the store directly that
    # weren't covered above (shouldn't exist, but defensive).
    counts["label_events"] += db.query(LabelEvent).filter(
        LabelEvent.store_id == store_id
    ).delete(synchronize_session=False)
    db.delete(store)
    db.commit()
    counts["store"] = 1
    logger.info(
        "delete_store: removed %r (id=%d) + %d products / %d history / "
        "%d label_events",
        name_before, store_id,
        counts["products"], counts["position_history"], counts["label_events"],
    )
    return {"success": True, "deleted": counts}

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


# =====================================================================
# CATEGORY-aware search — typing a category word ('lighting', 'shoes',
# 'eyewear', 'earring') does NOT substring-match the title. A naïve
# substring search for 'lighting' catches 'Lightweight Hiking Shoes'
# via the 'light' substring; a search for 'shoes' catches 'Shoehorn
# Steel'. Wrong both times.
#
# For category words we expand to a curated list of NOUN tokens and
# search them with a word boundary on title / handle / product_type.
# The bare category word is ALSO matched against ai_tags / subniche
# (Gemini may have tagged 'lighting' or 'jewelry' directly), but
# never against title — that's where the over-match would happen.
# =====================================================================
CATEGORY_NOUN_MAP = {
    # --- Lighting (the user's BUG 2 case) ---
    "lighting": [
        "chandelier", "chandeliers", "sconce", "sconces", "candelabra",
        "candleholder", "candle holder", "pendant light", "ceiling light",
        "ceiling lamp", "wall light", "wall lamp", "wall sconce",
        "table lamp", "floor lamp", "desk lamp", "night light",
        "nightlight", "lamp shade", "lampshade", "string lights",
        "fairy lights", "light bulb", "light fixture", "led bulb",
        "led strip", "led panel",
        "lamp", "lamps", "lampe", "lampen", "leuchte", "leuchten",
        "kronleuchter", "tischlampe", "stehlampe", "wandlampe",
        "deckenlampe", "hängelampe", "pendelleuchte", "wandleuchte",
        "lichterkette", "lustre", "suspension", "plafonnier",
        "applique", "abat-jour", "lampadario", "lampada", "paralume",
        "kroonluchter", "tafellamp", "vloerlamp",
        "araña", "lámpara", "candelabro",
    ],
    "lights": "lighting", "lamp": "lighting", "lamps": "lighting",
    "chandelier": "lighting", "lampe": "lighting", "lampen": "lighting",
    # --- Footwear ---
    # Bare 'shoe' / 'shoes' deliberately omitted — they substring-match
    # 'Shoehorn' / 'Shoe Polish' / 'Shoe Rack' (not footwear) too
    # often. Real footwear is almost always titled with a specific
    # noun (sneaker / boot / sandal / loafer / heel / slipper /
    # oxford / Schuh / chaussure / zapato / scarpe / schoenen).
    "footwear": [
        "sneaker", "sneakers", "boot", "boots",
        "sandal", "sandals", "slipper", "slippers", "heel", "heels",
        "loafer", "loafers", "stiletto", "stilettos", "oxfords",
        "schuh", "schuhe", "stiefel", "sandalen", "hausschuh",
        "halbschuh", "chaussure", "chaussures", "botte", "bottes",
        "sandale", "sandales", "basket", "baskets", "escarpins",
        "zapato", "zapatos", "bota", "botas", "sandalia", "sandalias",
        "zapatilla", "zapatillas", "scarpe", "stivali", "sandali",
        "schoenen", "laarzen", "slip-on", "slip-ons", "orthoschuh",
    ],
    "shoes": "footwear", "shoe": "footwear", "sneakers": "footwear",
    "boots": "footwear", "sandals": "footwear",
    # --- Apparel / clothing ---
    "apparel": [
        "dress", "dresses", "shirt", "shirts", "blouse", "blouses",
        "sweater", "sweaters", "hoodie", "hoodies", "jacket", "jackets",
        "coat", "coats", "skirt", "skirts", "pants", "jeans", "shorts",
        "trousers", "jumpsuit", "jumpsuits", "robe", "robes",
        "bathrobe", "bathrobes", "pajamas", "pyjamas",
        "kleid", "kleider", "hemd", "hemden", "bluse", "blusen",
        "pullover", "jacke", "jacken", "mantel", "mäntel",
        "rock", "röcke", "hose", "hosen", "bademantel",
        "chemise", "chemisier", "veste", "manteau", "jupe", "pantalon",
        "vestido", "camisa", "blusa", "chaqueta", "abrigo", "falda",
        "vestiti", "vestito", "camicia", "giacca", "cappotto", "gonna",
        "jurk", "rok", "broek", "trui", "jas",
    ],
    "clothing": "apparel", "clothes": "apparel",
    # --- Eyewear (style — not eyewear-accessory utility) ---
    "eyewear": [
        "sunglasses", "glasses", "eyewear", "frames", "brille",
        "brillen", "sonnenbrille", "lesebrille", "lunettes", "gafas",
        "occhiali", "bril",
    ],
    "sunglasses": "eyewear", "glasses": "eyewear",
    # --- Jewelry sub-categories ---
    "earring": [
        "earring", "earrings", "stud", "studs", "hoop", "hoops",
        "drop earring", "ohrring", "ohrringe", "boucle d'oreille",
        "pendiente", "orecchino", "oorbel",
    ],
    "earrings": "earring",
    "necklace": [
        "necklace", "necklaces", "pendant", "chain", "halskette",
        "kette", "collier", "collar", "collana", "ketting",
    ],
    "necklaces": "necklace",
    "bracelet": [
        "bracelet", "bracelets", "armband", "armbänder",
        "pulsera", "braccialetto", "armbandje",
    ],
    "bracelets": "bracelet",
    # --- Bags ---
    "bag": [
        "bag", "bags", "handbag", "handbags", "backpack", "backpacks",
        "tote", "totes", "clutch", "clutches", "wallet", "wallets",
        "purse", "purses", "crossbody", "pouch", "pouches", "satchel",
        "satchels", "tasche", "taschen", "rucksack", "sac", "bolso",
        "borsa", "tas",
    ],
    "bags": "bag", "handbag": "bag", "backpack": "bag",
}


def _resolve_category_alias(term: str) -> str | None:
    """If `term` (or its singular) is a key in CATEGORY_NOUN_MAP and
    the value is a string, follow the alias chain to find the canonical
    category. Returns the canonical key (whose value is the noun list)
    or None if the term isn't a category word."""
    for cand in _singularize(term.lower()):
        if cand not in CATEGORY_NOUN_MAP:
            continue
        seen = set()
        key = cand
        while isinstance(CATEGORY_NOUN_MAP[key], str):
            if key in seen:  # alias loop guard
                return None
            seen.add(key)
            key = CATEGORY_NOUN_MAP[key]
        return key
    return None


def category_nouns_for(term: str) -> list:
    """Return the curated noun list for a category term, or [] if
    the term isn't a category word."""
    canonical = _resolve_category_alias(term)
    if canonical is None:
        return []
    return list(CATEGORY_NOUN_MAP[canonical])


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


def _strict_word_clauses(column, term: str):
    """Match `term` against `column` requiring a word boundary on BOTH
    sides — so 'lamp' matches 'Table Lamp' / 'Floor Lamp' but NOT
    'lamppost' or 'lampoon'. Used for CATEGORY-noun matches against
    title / handle / product_type so 'lighting' doesn't catch
    'Lightweight Hiking Shoes' via the bare 'light' substring.

    Postgres: \\y boundary on both sides. SQLite: REGEXP function
    registered in database.py with \\b boundary.
    """
    if _is_postgres_db():
        pat = r"\y" + _re.escape(term) + r"\y"
        return [column.op("~*")(pat)]
    # SQLite (and any other dialect with REGEXP via the connect hook).
    pat = r"\b" + _re.escape(term) + r"\b"
    return [column.op("REGEXP")(pat)]


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

    For CATEGORY words (lighting, footwear, apparel, eyewear, earring,
    etc.) the bare word is NOT substring-matched against the title —
    that would catch 'lightweight' for 'lighting' or 'shoehorn' for
    'shoes'. Instead the search expands to a curated noun list with
    word-boundary matches on title/handle/product_type. The bare
    category word is still matched against ai_tags/subniche where
    Gemini may have tagged it directly.

    Variants under 3 characters are dropped — too short to match
    meaningfully and they cause runaway false positives.
    """
    pieces = []

    # PRIMARY signal: the new `product_category` column. Multilingual
    # token resolution (chandelier / Kronleuchter / lustre / lampadario
    # all → 'chandelier') AND parent expansion (typing 'lighting' →
    # union of every lighting child) live in categories.py.
    matched_categories = lookup_categories_for_query_token(word)
    if matched_categories:
        pieces.append(Product.product_category.in_(list(matched_categories)))

    nouns = category_nouns_for(word)
    if nouns:
        # Legacy strict-noun list: word-boundary match in title /
        # handle / product_type for products that haven't been
        # re-classified yet. Plus loose match in ai_tags / subniche.
        for n in nouns:
            for col in (Product.title, Product.handle, Product.product_type):
                pieces.extend(_strict_word_clauses(col, n))
            for col in (Product.ai_tags, Product.subniche):
                pieces.extend(_match_clauses(col, n))
        for col in (Product.ai_tags, Product.subniche):
            pieces.extend(_match_clauses(col, word.lower()))

    # If we got ANY category-driven match (new column or legacy nouns),
    # ALSO apply the existing SUBNICHE_SYNONYMS expansion against
    # title (with WORD BOUNDARIES so 'lighting' doesn't catch
    # 'Lightweight') AND ai_tags / subniche (loose substring is fine
    # there — those columns are curated). Lets umbrella terms like
    # 'jewelry' still hit pre-backfill products via subniche AND via
    # title substring of synonyms like 'earring'.
    if matched_categories or nouns:
        variants = [v for v in expand_single_term(word) if len(v) >= 3]
        for v in variants:
            # Title / handle / product_type — strict word boundary.
            for col in (Product.title, Product.handle, Product.product_type):
                pieces.extend(_strict_word_clauses(col, v))
            # ai_tags / subniche — loose substring (curated columns).
            for col in (Product.ai_tags, Product.subniche):
                pieces.extend(_match_clauses(col, v))
        return or_(*pieces) if pieces else None

    # Non-category token: existing substring search across title /
    # ai_tags / product_type / subniche, with translations + reverse
    # synonym expansion folded in.
    variants = [v for v in expand_single_term(word) if len(v) >= 3]
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
        # Skip the loose-OR fallback for category words too — typing
        # 'lighting' must NEVER fall through to a substring match for
        # 'light' that catches 'Lightweight Hiking Shoes'.
        if category_nouns_for(word):
            cond = _word_match_condition(word)
            if cond is not None:
                loose_conditions.append(cond)
            continue
        variants = [v for v in expand_single_term(word) if len(v) >= 3]
        for v in variants:
            for col in SEARCH_COLUMNS:
                loose_conditions.append(_match_clauses(col, v)[0])
    return strict_conditions, loose_conditions


def build_ai_tag_filters(search_query: str):
    """AND-of-words match across ai_tags + product_type + subniche.
    Each word is expanded via expand_single_term so naive plurals,
    multi-language translations, and subniche reverse-lookup
    ('earring' → 'jewelry') ALL participate on the primary search
    path — otherwise the fallback would miss subniche-only hits
    whenever the ai_tags query returns any unrelated results.

    For CATEGORY words, the same strict mode applies — the bare word
    is matched against ai_tags / subniche only (where Gemini may have
    tagged it), but the title-substring path is bypassed entirely so
    'lighting' doesn't catch 'Lightweight'.
    """
    words = [w for w in search_query.lower().split() if w]
    conds = []
    for w in words:
        nouns = category_nouns_for(w)
        if nouns:
            # Category mode: noun-list with strict word boundary on
            # title/handle/product_type, plus loose match on ai_tags +
            # subniche for nouns AND the bare category word.
            word_or = []
            for n in nouns:
                for col in (Product.title, Product.handle, Product.product_type):
                    word_or.extend(_strict_word_clauses(col, n))
                for col in (Product.ai_tags, Product.subniche):
                    word_or.extend(_match_clauses(col, n))
            for col in (Product.ai_tags, Product.subniche):
                word_or.extend(_match_clauses(col, w))
            if word_or:
                conds.append(or_(*word_or))
            continue
        word_or = []
        for variant in expand_single_term(w):
            if len(variant) < 3:
                continue
            for col in AI_TAG_SEARCH_COLUMNS:
                word_or.extend(_match_clauses(col, variant))
        if word_or:
            conds.append(or_(*word_or))
    return conds

async def hybrid_search(
    db: Session,
    base_query,
    search_text: str,
    *,
    limit: int = 300,
    rerank: bool = True,
) -> list:
    """AI-powered search: query expansion + OR-merged keyword
    prefilter + Python scoring + optional Gemini re-rank.

    Replaces the old 3-stage AND-then-OR short-circuit (formerly in
    every feed endpoint) which dropped users into a "0 results" hole
    whenever ANY query word was unknown to the keyword index. The new
    pipeline:

      1. expand_query(search_text) — Gemini turns "prom dress" into
         occasion/style/material tags + multilingual synonyms +
         catalog phrases. Cached aggressively.
      2. SQL prefilter — a single OR-of-OR query against
         (title | ai_tags | product_category | subniche |
         product_type | handle) using every expansion term. NO AND
         across query words: each one independently widens the
         candidate set. Capped at 5x the user's display limit so
         the Python scoring stage stays fast.
      3. Python scoring — score_product_against_expansion(...) gives
         each candidate a weighted relevance score (exact-phrase title
         match > tag overlap > free-token overlap).
      4. Drop score==0 (no signal at all), sort by score desc.
      5. Optional Gemini re-rank of the top 50 (env-flag
         SEARCH_RERANK=1 by default) — gives Gemini the user query
         plus the top candidates and asks it to drop irrelevant ones.

    Args:
      base_query: SQLAlchemy query already scoped by is_fashion + label.
      search_text: raw user input. Trailing/leading whitespace ignored.
      limit: max products to return.
      rerank: whether to invoke Gemini re-rank on the top N. Off for
        unit tests; on by default in production.

    Returns: list of Product ORM instances in relevance order.
    """
    from query_expander import (
        expand_query,
        rerank_with_gemini,
        score_product_against_expansion,
    )

    s = (search_text or "").strip()
    if not s:
        return base_query.all()[:limit]

    # 1. Query expansion (cached, ~500ms first call, <1ms cached)
    exp = await expand_query(s)

    # 2. SQL prefilter. KEY CHANGE: only OR-match against title and
    #    ai_tags — NOT product_category / subniche / product_type /
    #    handle. Those weaker columns frequently contain shared common
    #    tokens (e.g. subniche='fashion', product_category='dress')
    #    and ORing %term% against them turned the prefilter into a
    #    near-full-table scan that returned irrelevant cross-category
    #    candidates. Title and ai_tags are curated per-product
    #    signals; weaker columns get used in Python scoring only.
    strong_terms = exp.strong_signal_terms()
    if not strong_terms:
        return []

    or_clauses = []
    for term in strong_terms:
        if len(term) < 3:
            continue
        like_pat = f"%{term}%"
        or_clauses.append(Product.title.ilike(like_pat))
        or_clauses.append(Product.ai_tags.ilike(like_pat))

    if not or_clauses:
        return []

    candidate_query = base_query.filter(or_(*or_clauses)).limit(limit * 5)
    candidates = candidate_query.all()

    if not candidates:
        return []

    # 3. HARD CATEGORY GATE. If the expander committed to one or more
    #    intent_types (e.g. "dress", "shoes", "bag"), every returned
    #    product MUST contain at least one keyword for one of those
    #    types in its title or ai_tags. This is what stops "Mini Dress"
    #    from polluting a shoe search and "Derby Dress Shoes" from
    #    polluting a dress search. Open-ended queries (intent_types
    #    is empty) skip this gate.
    #
    #    CRITICAL: use word-boundary matching, NOT substring. Plain
    #    substring caused "ring" (in jewelry kws) to match "spring"
    #    (in a dress title), polluting earrings search with dresses.
    #    Same for "lamp" matching "lamp" but NOT "blamp" / "lampoon"
    #    / "Klampenstein". Compile pattern once per call so the
    #    regex cost is constant per query.
    intent_kws = exp.intent_keywords()
    if intent_kws:
        # Build a single alternation pattern for cheap N-product
        # match against every intent keyword at once.
        escaped = [_re.escape(kw) for kw in intent_kws if kw]
        # \b is the standard Python word-boundary which handles
        # multilingual unicode word characters via re.UNICODE.
        # For hyphenated kws (mary-jane, faux-leather) \b on the
        # outside works because the hyphen IS a word boundary.
        gate_re = _re.compile(
            r"\b(?:" + "|".join(escaped) + r")\b",
            _re.IGNORECASE | _re.UNICODE,
        )
        gated = []
        for p in candidates:
            haystack = (
                (p.title or "").lower()
                + " "
                + (p.ai_tags or "").lower()
            )
            if gate_re.search(haystack):
                gated.append(p)
        candidates = gated
        if not candidates:
            return []

    # 3b. DETERMINISTIC GENDER FILTER. Gemini Flash-lite is
    #     unreliable on this specific axis — a "puffer jacket for
    #     women" query kept leaking "Doudoune Homme" because Flash
    #     ignored the explicit gender constraint. This filter is a
    #     mechanical safety net: detect the gender of the query and
    #     drop products whose title clearly indicates the opposite
    #     gender. Multilingual: men = men/man/men's/homme/herren/
    #     hombre/uomo/heren/man, women = women/woman/ladies/femme/
    #     dame/damen/dama/mujer/donna/dames.
    _MEN_TOKENS = (
        r"\bmen\b", r"\bman\b", r"\bmen's\b", r"\bmens\b",
        r"\bhomme\b", r"\bhommes\b",
        r"\bherren\b", r"\bherr\b",
        r"\bhombre\b", r"\bhombres\b",
        r"\buomo\b", r"\buomini\b",
        r"\bheren\b",
    )
    _WOMEN_TOKENS = (
        r"\bwomen\b", r"\bwoman\b", r"\bwomen's\b", r"\bwomens\b",
        r"\bladies\b", r"\blady\b",
        r"\bfemme\b", r"\bfemmes\b",
        r"\bdame\b", r"\bdamen\b",
        r"\bdama\b", r"\bmujer\b", r"\bmujeres\b",
        r"\bdonna\b", r"\bdonne\b",
        r"\bdames\b",
    )
    _MEN_RE = _re.compile("|".join(_MEN_TOKENS), _re.IGNORECASE | _re.UNICODE)
    _WOMEN_RE = _re.compile("|".join(_WOMEN_TOKENS), _re.IGNORECASE | _re.UNICODE)
    s_lower = s.lower()
    query_wants_men = bool(_MEN_RE.search(s_lower))
    query_wants_women = bool(_WOMEN_RE.search(s_lower))
    if query_wants_men or query_wants_women:
        filtered = []
        for p in candidates:
            haystack = (p.title or "") + " " + (p.ai_tags or "")
            has_men = bool(_MEN_RE.search(haystack))
            has_women = bool(_WOMEN_RE.search(haystack))
            # Drop products whose gender tag CONTRADICTS the query.
            if query_wants_women and has_men and not has_women:
                continue
            if query_wants_men and has_women and not has_men:
                continue
            filtered.append(p)
        candidates = filtered
        if not candidates:
            return []

    # 4. Python-side scoring. Pure functions, no I/O.
    scored = []
    for p in candidates:
        score = score_product_against_expansion(
            title=p.title or "",
            ai_tags=p.ai_tags or "",
            product_category=p.product_category or "",
            subniche=p.subniche or "",
            product_type=p.product_type or "",
            handle=p.handle or "",
            exp=exp,
        )
        # Min-score threshold: a product needs at least one real
        # signal (not just an accidental category/handle token hit).
        # Raised from 0 to 4 to cut the long tail of "barely matches"
        # results the user complained about. With Gemini expansion
        # active in production, tagged-relevant products score 20+
        # easily; this threshold only filters the noise tier.
        if score >= 4:
            scored.append((score, p))

    if not scored:
        return []

    # 5. Sort by score desc, then by current_position asc (tie-break
    #    on bestseller rank — lower is better).
    scored.sort(key=lambda sp: (-sp[0], sp[1].current_position or 999999))

    # 6. STRICT MATCH JUDGE (Gemini). For every candidate, Gemini
    #    decides match=true|false against the user's full query
    #    (every constraint — type, sub-type, gender, color, material,
    #    occasion). match=false items are HARD-DROPPED and never
    #    shown. The user explicitly demanded "100% accuracy" and
    #    "prefer zero results over inaccurate results" — this honors
    #    that promise.
    #
    #    Up to 100 candidates go to the judge (was 50 for the old
    #    soft rerank). Anything beyond 100 isn't shown — we'd rather
    #    cap visible results than show un-judged ones. Common queries
    #    cache after the first call so repeat searches are O(1).
    judge_enabled = (
        rerank
        and os.getenv("SEARCH_RERANK", "1") != "0"
        and len(scored) >= 1
    )
    if judge_enabled:
        top_n = min(100, len(scored))
        judge_candidates = [
            {
                "id": p.id,
                "title": p.title or "",
                "ai_tags": p.ai_tags or "",
                "subniche": p.subniche or "",
            }
            for _, p in scored[:top_n]
        ]
        try:
            verdicts = await rerank_with_gemini(s, judge_candidates)
        except Exception as e:
            logger.warning(
                "hybrid_search: strict judge failed: %s — falling back to hybrid order",
                e,
            )
            verdicts = [
                {"idx": i, "match": True, "reason": "judge errored"}
                for i in range(top_n)
            ]
        match_by_idx = {v["idx"]: v["match"] for v in verdicts}
        kept = []
        for i, (orig_score, p) in enumerate(scored[:top_n]):
            if match_by_idx.get(i, False):
                kept.append((orig_score, p))
        kept.sort(key=lambda sp: (-sp[0], sp[1].current_position or 999999))
        # NO tail. Strict mode means: only items the judge approved.
        # Anything in scored[top_n:] wasn't judged → not shown.
        scored = kept

    return [p for _, p in scored[:limit]]


def _label_events_to_product_dicts(
    pairs: list, is_fashion: bool,
) -> list[dict]:
    """Render (product, event) pairs from fetch_label_events_window into
    the same dict shape `_product_dict` produces, with the event's
    date / prior_position / position_change so the UI can show
    "moved on May 4: 12 → 7"."""
    out = []
    for prod, ev in pairs:
        d = _product_dict(prod, (ev.label, ev.position_change, ev.prior_position))
        d["event_date"] = ev.date.date().isoformat()
        d["is_active"] = bool(
            (is_fashion and prod.is_fashion)
            or ((not is_fashion) and (not prod.is_fashion) and prod.subniche)
        )
        out.append(d)
    return out


@app.get("/api/bestsellers/combined")
async def get_combined_bestsellers(
    sort: str = Query("high-low"),
    label: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    limit: int = Query(30, ge=1, le=500),
    days: int = Query(1, ge=1, le=LABEL_EVENT_RETENTION_DAYS),
    db: Session = Depends(get_db),
):
    """Combined Fashion feed.

    When `label` is 'hero' or 'villain', the result is sourced from
    the persistent LabelEvent ledger for the last `days` UTC days
    (default 1 = today only, preserves the legacy behaviour). Each
    product appears at most once, paired with its MOST RECENT event
    of the requested label inside the window. Products that have
    since dropped from the Fashion feed are still included with
    `is_active: false` so the UI can render them differently.

    For other labels (normal / new / null) or when days==1 with no
    LabelEvent data, falls back to the read-time _compute_label_map
    path against PositionHistory.
    """
    use_events = label in ("hero", "villain")
    if use_events:
        pairs = fetch_label_events_window(
            db, label=label, days=days, is_fashion=True,
        )
        # Apply search via the AI-powered hybrid_search restricted to
        # this hero/villain candidate set. The expander+rerank give us
        # the same multilingual / occasion-tag superpowers here that
        # the no-events path gets.
        if search and search.strip():
            id_set = {p.id for p, _ in pairs}
            if not id_set:
                return []
            scoped = db.query(Product).filter(Product.id.in_(list(id_set)))
            ranked = await hybrid_search(db, scoped, search.strip(), limit=limit * 3)
            ranked_ids = {p.id for p in ranked}
            pairs = [pair for pair in pairs if pair[0].id in ranked_ids]
        # Sort.
        if sort == "volume":
            pairs.sort(key=lambda pe: (
                -parse_visitors(pe[0].store.monthly_visitors),
                pe[0].current_position or 0,
            ))
        elif sort == "low-high":
            pairs.sort(key=lambda pe: -(pe[0].current_position or 0))
        else:
            pairs.sort(key=lambda pe: pe[0].current_position or 0)
        pairs = pairs[:limit]
        return _label_events_to_product_dicts(pairs, is_fashion=True)

    query = db.query(Product).join(Store).filter(Product.is_fashion == True)
    query = _apply_label_filter(query, db, label)

    if search and search.strip():
        # AI-powered search: Gemini query expansion + OR-merged
        # prefilter + Python scoring + Gemini re-rank top 50. See
        # hybrid_search() docstring for the full pipeline.
        products = await hybrid_search(db, query, search.strip(), limit=limit * 3)
    else:
        products = query.all()

    if sort == "volume":
        products.sort(key=lambda p: (-parse_visitors(p.store.monthly_visitors), p.current_position))
    elif sort == "low-high":
        products.sort(key=lambda p: (-p.current_position,))
    elif sort == "high-low":
        products.sort(key=lambda p: p.current_position)
    # else: leave hybrid_search's relevance order intact when search
    # is active (no explicit secondary sort).

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
        products = await hybrid_search(db, query, search.strip(), limit=limit * 3)
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

# _today_start_utc / _trustworthy_prior_filters / _delta_threshold are
# re-exported from labels.py at the top of this module so the rest of
# the file can use them under their existing names. Keep _prior_position_subquery
# local — it's specific to the read-time SQL join pattern.


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


# _delta_threshold is imported from labels.py at the top of this module.


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
    days: int = Query(1, ge=1, le=LABEL_EVENT_RETENTION_DAYS),
    db: Session = Depends(get_db),
):
    """Cross-store General feed. Same shape + day-range semantics as
    /api/bestsellers/combined. With `label` ∈ {hero, villain} the
    result is sourced from LabelEvent for the last `days` days.
    `subniche` is BACKEND-only metadata used by the search box; it is
    never exposed as a user-facing filter parameter.
    """
    use_events = label in ("hero", "villain")
    if use_events:
        pairs = fetch_label_events_window(
            db, label=label, days=days, is_fashion=False,
        )
        if search and search.strip():
            id_set = {p.id for p, _ in pairs}
            if not id_set:
                return []
            scoped = db.query(Product).filter(Product.id.in_(list(id_set)))
            ranked = await hybrid_search(db, scoped, search.strip(), limit=limit * 3)
            ranked_ids = {p.id for p in ranked}
            pairs = [pair for pair in pairs if pair[0].id in ranked_ids]
        if sort == "volume":
            pairs.sort(key=lambda pe: (
                -parse_visitors(pe[0].store.monthly_visitors),
                pe[0].current_position or 0,
            ))
        elif sort == "low-high":
            pairs.sort(key=lambda pe: -(pe[0].current_position or 0))
        else:
            pairs.sort(key=lambda pe: pe[0].current_position or 0)
        pairs = pairs[:limit]
        return _label_events_to_product_dicts(pairs, is_fashion=False)

    query = _general_base_query(db)
    query = _apply_label_filter(query, db, label)

    if search and search.strip():
        products = await hybrid_search(db, query, search.strip(), limit=limit * 3)
    else:
        products = query.all()

    if sort == "volume":
        products.sort(key=lambda p: (-parse_visitors(p.store.monthly_visitors), p.current_position))
    elif sort == "low-high":
        products.sort(key=lambda p: (-p.current_position,))
    elif sort == "high-low":
        products.sort(key=lambda p: p.current_position)
    # else: leave hybrid_search relevance order intact

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
        products = await hybrid_search(db, query, search.strip(), limit=limit * 3)
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


@app.post("/api/admin/reset-products")
async def admin_reset_products(db: Session = Depends(get_db)):
    """One-shot wipe of the products / history / event tables.

    Needed after the Railway → Render migration: the first scrape ran
    in degraded mode (no GEMINI_API_KEY) which routed EVERY product
    into the Fashion feed with subniche='fashion'. After the key was
    set, only currently-bestselling products got re-classified by
    re-scrape; previously-seen products that dropped from the
    bestseller list kept their stale degraded-mode classification,
    polluting the Fashion tab with non-fashion items (smartwatches,
    drum mats, blood pressure monitors, etc.).

    This endpoint clears the slate so the next /api/scrape produces
    a 100% Gemini-classified catalog. Stores are NOT touched. The
    DATA_START_DATE floor already wiped the LabelEvent ledger on
    boot; this just makes that the visible-products state too.
    """
    counts = {}
    counts["label_events"] = db.query(LabelEvent).delete(synchronize_session=False)
    counts["position_history"] = db.query(PositionHistory).delete(synchronize_session=False)
    counts["products"] = db.query(Product).delete(synchronize_session=False)
    db.commit()
    logger.warning(
        "admin_reset_products: wiped %d products, %d history rows, %d label_events",
        counts["products"], counts["position_history"], counts["label_events"],
    )
    return {"success": True, "deleted": counts}


@app.get("/api/debug/vision")
async def debug_vision(image_url: str):
    """Diagnostic: classify a single product image via Gemini vision.
    Returns the raw vision JSON + the flattened `img:` tag string
    that would be prepended to Product.ai_tags. Use to smoke-test
    the vision pipeline in production without triggering a full
    scrape or backfill."""
    from image_classifier import classify_single_image
    return await classify_single_image(image_url)


@app.post("/api/admin/backfill-vision")
async def admin_backfill_vision(
    limit: int = Query(50, ge=1, le=500),
    only_fashion: bool = Query(True),
    force: bool = Query(False),
    store_id: Optional[int] = Query(None),
    concurrency: int = Query(8, ge=1, le=16),
    db: Session = Depends(get_db),
):
    """Backfill vision classification onto existing products.

    Iterates products that have not yet been vision-classified,
    fetches their primary image, runs the vision model, and prepends
    `img:*` tokens to `ai_tags`. Idempotent — skips rows with
    `vision_classified_at IS NOT NULL` unless `force=true`.

    Cost math: ~$0.00011 per image on Flash-lite. Backfilling 5,000
    fashion products = ~$0.55, ~15 minutes at 8x concurrency.

    Query params:
      limit          max products this call (1..500, default 50)
      only_fashion   skip general/electronics (default true)
      force          reclassify even if already vision-classified
      store_id       optional per-store filter
      concurrency    parallel Gemini calls (1..16, default 8)
    """
    from image_classifier import classify_images_batch

    q = db.query(Product).filter(Product.image_url != "")
    if only_fashion:
        q = q.filter(Product.is_fashion == True)
    if not force:
        q = q.filter(Product.vision_classified_at.is_(None))
    if store_id is not None:
        q = q.filter(Product.store_id == store_id)
    q = q.order_by(Product.last_scraped.desc()).limit(limit)
    products = q.all()

    if not products:
        return {"processed": 0, "errors": [], "message": "no candidates"}

    # Shape a per-product dict the vision classifier expects.
    dicts = [
        {
            "product_id": p.id,
            "image_url": p.image_url or "",
            "ai_tags": p.ai_tags or "",
            "handle": p.handle or "",
        }
        for p in products
    ]

    import time
    t0 = time.monotonic()
    errors = await classify_images_batch(dicts, concurrency=concurrency)
    elapsed = time.monotonic() - t0

    # Persist changes back to the DB row.
    now = datetime.utcnow()
    processed = 0
    classified = 0
    skipped_not_a_product = 0
    for prod, d in zip(products, dicts):
        if d.get("vision_classified"):
            classified += 1
            prod.ai_tags = d.get("ai_tags") or prod.ai_tags
            prod.vision_classified_at = now
            if d.get("_excluded"):
                skipped_not_a_product += 1
        processed += 1
    db.commit()

    return {
        "processed": processed,
        "vision_classified": classified,
        "flagged_not_a_product": skipped_not_a_product,
        "elapsed_seconds": round(elapsed, 1),
        "errors": errors[:5],
    }


@app.get("/api/debug/search")
async def debug_search(q: str, db: Session = Depends(get_db)):
    """Diagnostic: show the live Gemini expansion + per-candidate
    score breakdown for a given query, so we can see exactly why a
    "shoes" search surfaced a dress (or didn't).

    Returns:
      - expansion: every field the expander produced (intent_types,
        canonical_terms, occasion/style/material/color tags,
        multilingual_nouns, semantic_phrases, expander_used, cached)
      - intent_keywords: the union the gate was checking against
      - candidates: top 15 results post-gate post-scoring with their
        title + ai_tags + final score, so we can see which signals
        fired and which didn't
    """
    from query_expander import (
        expand_query, expansion_to_dict,
        score_product_against_expansion,
    )
    s = (q or "").strip()
    if not s:
        return {"error": "missing q"}

    exp = await expand_query(s)
    intent_kws = exp.intent_keywords()

    # Run the same prefilter the real search uses
    strong_terms = exp.strong_signal_terms()
    or_clauses = []
    for term in strong_terms:
        if len(term) < 3:
            continue
        like_pat = f"%{term}%"
        or_clauses.append(Product.title.ilike(like_pat))
        or_clauses.append(Product.ai_tags.ilike(like_pat))

    base_query = db.query(Product).filter(Product.is_fashion == True)
    candidates = base_query.filter(or_(*or_clauses)).limit(150).all() if or_clauses else []

    # Apply the same gate the real search uses
    gated = []
    if intent_kws:
        escaped = [_re.escape(kw) for kw in intent_kws if kw]
        gate_re = _re.compile(
            r"\b(?:" + "|".join(escaped) + r")\b",
            _re.IGNORECASE | _re.UNICODE,
        )
        for p in candidates:
            haystack = (p.title or "").lower() + " " + (p.ai_tags or "").lower()
            if gate_re.search(haystack):
                gated.append(p)
    else:
        gated = candidates

    # Score
    scored = []
    for p in gated:
        score = score_product_against_expansion(
            title=p.title or "", ai_tags=p.ai_tags or "",
            product_category=p.product_category or "",
            subniche=p.subniche or "", product_type=p.product_type or "",
            handle=p.handle or "", exp=exp,
        )
        scored.append((score, p))
    scored.sort(key=lambda sp: (-sp[0], sp[1].current_position or 999999))

    return {
        "query": s,
        "expansion": expansion_to_dict(exp),
        "intent_keywords": sorted(intent_kws),
        "candidates_after_prefilter": len(candidates),
        "candidates_after_gate": len(gated),
        "top_15_scored": [
            {
                "score": s,
                "id": p.id,
                "title": p.title,
                "ai_tags": p.ai_tags,
                "product_category": p.product_category,
                "subniche": p.subniche,
                "store_id": p.store_id,
            }
            for s, p in scored[:15]
        ],
    }


@app.post("/api/reset-labels")
async def trigger_reset_labels(db: Session = Depends(get_db)):
    count = reset_all_labels(db)
    return {"success": True, "products_reset": count}

async def _background_scrape_one(store_id: int):
    """Run a single-store scrape in the background. Mirrors
    _background_scrape_all so the API can return immediately and the
    user can poll for the result without hitting Render's 60-second
    request timeout (which used to kill long per-store scrapes mid-
    flight and surface as a misleading "page 1 HTTP error:" alert).
    """
    db = SessionLocal()
    try:
        store = db.query(Store).filter(Store.id == store_id).first()
        if not store:
            _scrape_state["result"] = {
                "stores": [], "total_products": 0, "total_general": 0,
                "stores_with_products": 0, "stores_failed": 1,
                "error": f"Store id={store_id} not found",
            }
            return
        s_name, s_url, s_niche = store.name, store.url, (store.niche or "")
        try:
            fashion, general, errors = await scrape_store_bestsellers(s_url)
            if fashion or general:
                update_products_in_db(db, store, fashion, general)
            _scrape_state["result"] = {
                "stores": [{
                    "id": store_id, "name": s_name, "niche": s_niche,
                    "products": len(fashion), "general": len(general),
                    "errors": errors, "warning": None,
                }],
                "total_products": len(fashion),
                "total_general": len(general),
                "stores_with_products": 1 if (fashion or general) else 0,
                "stores_failed": 0 if (fashion or general) else 1,
            }
        except Exception as e:
            _scrape_state["result"] = {
                "stores": [{
                    "id": store_id, "name": s_name, "niche": s_niche,
                    "products": 0, "general": 0,
                    "errors": [f"unhandled: {e}"], "warning": None,
                }],
                "total_products": 0, "total_general": 0,
                "stores_with_products": 0, "stores_failed": 1,
                "error": str(e),
            }
            logger.exception(f"Background single-store scrape failed for {s_name}")
    finally:
        db.close()


@app.post("/api/scrape/{store_id}")
async def trigger_store_scrape(store_id: int, db: Session = Depends(get_db)):
    """Kick off a single-store scrape in the background and return
    immediately. The frontend polls /api/scrape/status for the result.

    Previously this endpoint was synchronous — Render kills any HTTP
    request that runs longer than 60s, so any merchant whose scrape
    took longer (Shopify pagination + Gemini classification commonly
    takes 60-300s) returned a 502 to the browser. The frontend then
    rendered the 502 HTML as the API response and the user saw a
    misleading "Scraped 0 products. page 1 HTTP error:" alert.
    """
    store = db.query(Store).filter(Store.id == store_id).first()
    if not store:
        raise HTTPException(status_code=404, detail="Store not found")
    if _scrape_state["running"]:
        return {
            "started": False, "running": True,
            "started_at": _scrape_state["started_at"],
            "message": "Another scrape is already in progress",
        }
    _scrape_state["running"] = True
    _scrape_state["started_at"] = datetime.utcnow().isoformat()
    _scrape_state["result"] = None

    async def _runner():
        try:
            await _background_scrape_one(store_id)
        finally:
            _scrape_state["running"] = False

    asyncio.create_task(_runner())
    return {
        "started": True, "running": True,
        "store_id": store_id,
        "started_at": _scrape_state["started_at"],
        "message": "Per-store scrape started in background. Poll /api/scrape/status for progress.",
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


@app.get("/api/debug/search")
async def debug_search(q: str = Query(""), db: Session = Depends(get_db)):
    """Diagnostic for the hybrid search pipeline. Returns the parsed
    query, per-token category resolution (with multilingual + parent
    expansion), the SQL clause counts each strategy contributes, and
    a sample 5 matches per strategy. Use to debug 'why didn't X show
    up' in one curl.
    """
    tokens = [t for t in (q or "").lower().split() if t]
    per_token = []
    union_categories: set = set()
    for tok in tokens:
        cats = lookup_categories_for_query_token(tok)
        legacy = category_nouns_for(tok) or []
        per_token.append({
            "token": tok,
            "matched_categories": sorted(cats),
            "legacy_nouns_sample": legacy[:8],
            "is_category_token": bool(cats),
            "is_parent_token": tok in PARENTS_TO_CHILDREN,
        })
        union_categories.update(cats)

    counts = {}
    samples = {}

    # Strategy A — match on Product.product_category (the new column).
    if union_categories:
        cat_query = (
            db.query(Product)
            .filter(Product.product_category.in_(list(union_categories)))
            .limit(500)
        )
        cat_rows = cat_query.all()
        counts["product_category"] = len(cat_rows)
        samples["product_category"] = [
            {"id": p.id, "title": (p.title or "")[:80],
             "category": p.product_category, "subniche": p.subniche}
            for p in cat_rows[:5]
        ]
    else:
        counts["product_category"] = 0
        samples["product_category"] = []

    # Strategy B — title substring (the existing path).
    if q:
        from sqlalchemy import or_
        sub_q = db.query(Product)
        for cond in build_ai_tag_filters(q):
            sub_q = sub_q.filter(cond)
        ai_rows = sub_q.limit(500).all()
        counts["ai_tag_strict_AND"] = len(ai_rows)
        samples["ai_tag_strict_AND"] = [
            {"id": p.id, "title": (p.title or "")[:80],
             "category": p.product_category, "subniche": p.subniche}
            for p in ai_rows[:5]
        ]

        strict, loose = build_search_filters(q)
        sq2 = db.query(Product)
        for cond in strict:
            sq2 = sq2.filter(cond)
        strict_rows = sq2.limit(500).all()
        counts["title_strict_AND"] = len(strict_rows)
        samples["title_strict_AND"] = [
            {"id": p.id, "title": (p.title or "")[:80],
             "category": p.product_category, "subniche": p.subniche}
            for p in strict_rows[:5]
        ]

    return {
        "query": q,
        "tokens": tokens,
        "per_token": per_token,
        "matched_categories_union": sorted(union_categories),
        "strategy_counts": counts,
        "strategy_samples": samples,
    }


@app.get("/api/debug/heroes")
async def debug_heroes(db: Session = Depends(get_db)):
    """Diagnostic for the hero/villain pipeline. Reports:

      - the active TRUST_EPOCH_UTC and today's UTC midnight
      - the invariant-violation flag (TRUST_EPOCH >= today_start
        silently disables labels)
      - per-store snapshot date histogram (last 5 UTC days)
      - delta distribution between the most-recent prior snapshot
        (date < today_start AND date >= TRUST_EPOCH_UTC) and
        Product.current_position
      - sample 10 fashion products with their prior + current
        position so the operator can eyeball whether the join is
        finding the right pairs

    The user has hit the silent-0/0 regression twice now. This
    endpoint exists so the next time it happens, the diagnosis is
    one curl away instead of an investigation.
    """
    today_start = _today_start_utc()
    invariant = _trust_epoch_invariant_check()

    # Snapshot dates per store (last 5 UTC days)
    five_days_ago = today_start - timedelta(days=5)
    rows = (
        db.query(
            Store.name,
            func.date(PositionHistory.date).label("snapshot_date"),
            func.count(PositionHistory.id).label("n"),
        )
        .join(Product, Product.id == PositionHistory.product_id)
        .join(Store, Store.id == Product.store_id)
        .filter(PositionHistory.date >= five_days_ago)
        .group_by(Store.name, func.date(PositionHistory.date))
        .order_by(Store.name, func.date(PositionHistory.date))
        .all()
    )
    histogram = {}
    for store_name, snap_date, n in rows:
        histogram.setdefault(store_name, []).append(
            {"date": str(snap_date), "snapshots": int(n)},
        )

    # Compute the prior subquery and join Products to it.
    sub = _prior_position_subquery(db, today_start)
    pairs = (
        db.query(
            Product.id, Product.title, Store.name,
            Product.is_fashion, Product.current_position,
            sub.c.prior_position,
        )
        .join(Store, Store.id == Product.store_id)
        .outerjoin(sub, sub.c.product_id == Product.id)
        .all()
    )
    deltas = []
    fashion_movers = []
    for pid, title, sname, is_fash, cur, prior in pairs:
        if prior is None:
            continue
        delta = (cur or 0) - prior
        deltas.append(delta)
        if is_fash and delta != 0 and len(fashion_movers) < 10:
            fashion_movers.append({
                "id": pid, "store": sname, "title": title[:80],
                "prior": prior, "current": cur, "delta": delta,
            })

    if deltas:
        deltas_sorted = sorted(deltas, key=abs)
        n = len(deltas)
        delta_summary = {
            "n_with_prior": n,
            "abs_min": min(abs(d) for d in deltas),
            "abs_median": abs(deltas_sorted[n // 2]),
            "abs_p90": abs(deltas_sorted[int(n * 0.9)]),
            "abs_max": max(abs(d) for d in deltas),
            "within_30_cap": sum(1 for d in deltas if abs(d) <= 30),
            "exceeds_30_cap": sum(1 for d in deltas if abs(d) > 30),
            "moved_up": sum(1 for d in deltas if d < 0),  # rank decreased = better
            "moved_down": sum(1 for d in deltas if d > 0),
            "unchanged": sum(1 for d in deltas if d == 0),
        }
    else:
        delta_summary = {"n_with_prior": 0}

    # Hero/villain SQL counts (mirrors get_stats but exposed here).
    heroes_q = (
        db.query(func.count(Product.id))
        .join(sub, sub.c.product_id == Product.id)
        .filter(
            Product.is_fashion == True,
            Product.current_position < sub.c.prior_position,
            sub.c.prior_position - Product.current_position <= HERO_VILLAIN_DELTA_CAP,
        )
        .scalar() or 0
    )
    villains_q = (
        db.query(func.count(Product.id))
        .join(sub, sub.c.product_id == Product.id)
        .filter(
            Product.is_fashion == True,
            Product.current_position > sub.c.prior_position,
            Product.current_position - sub.c.prior_position <= HERO_VILLAIN_DELTA_CAP,
        )
        .scalar() or 0
    )

    # Persistent LabelEvent ledger stats — verify retention is
    # maintained and the 7/14/30-day windows have data.
    retention_cutoff = today_start - timedelta(days=LABEL_EVENT_RETENTION_DAYS)
    total_events = db.query(func.count(LabelEvent.id)).scalar() or 0
    pre_start_events = (
        db.query(func.count(LabelEvent.id))
        .filter(LabelEvent.date < DATA_START_DATE)
        .scalar() or 0
    )
    events_7d = (
        db.query(func.count(LabelEvent.id))
        .filter(LabelEvent.date >= today_start - timedelta(days=6))
        .scalar() or 0
    )
    events_14d = (
        db.query(func.count(LabelEvent.id))
        .filter(LabelEvent.date >= today_start - timedelta(days=13))
        .scalar() or 0
    )
    events_30d = (
        db.query(func.count(LabelEvent.id))
        .filter(LabelEvent.date >= today_start - timedelta(days=29))
        .scalar() or 0
    )
    oldest_event = (
        db.query(func.min(LabelEvent.date)).scalar()
    )
    newest_event = (
        db.query(func.max(LabelEvent.date)).scalar()
    )
    heroes_by_day = (
        db.query(
            func.date(LabelEvent.date).label("day"),
            func.count(LabelEvent.id).label("n"),
        )
        .filter(LabelEvent.label == "hero")
        .filter(LabelEvent.date >= today_start - timedelta(days=29))
        .group_by(func.date(LabelEvent.date))
        .order_by(func.date(LabelEvent.date).desc())
        .all()
    )
    villains_by_day = (
        db.query(
            func.date(LabelEvent.date).label("day"),
            func.count(LabelEvent.id).label("n"),
        )
        .filter(LabelEvent.label == "villain")
        .filter(LabelEvent.date >= today_start - timedelta(days=29))
        .group_by(func.date(LabelEvent.date))
        .order_by(func.date(LabelEvent.date).desc())
        .all()
    )

    return {
        "trust_epoch_utc": TRUST_EPOCH_UTC.isoformat(),
        "today_start_utc": today_start.isoformat(),
        "invariant_violation": invariant,
        "snapshots_per_store_last_5_days": histogram,
        "delta_summary": delta_summary,
        "fashion_heroes_count_today": int(heroes_q),
        "fashion_villains_count_today": int(villains_q),
        "sample_fashion_movers": fashion_movers,
        "label_event_ledger": {
            "total_events": int(total_events),
            "events_in_last_7d": int(events_7d),
            "events_in_last_14d": int(events_14d),
            "events_in_last_30d": int(events_30d),
            "oldest_event_date": (
                oldest_event.date().isoformat() if oldest_event else None
            ),
            "newest_event_date": (
                newest_event.date().isoformat() if newest_event else None
            ),
            "retention_window_days": LABEL_EVENT_RETENTION_DAYS,
            "retention_cutoff": retention_cutoff.date().isoformat(),
            "data_start_date": DATA_START_DATE.date().isoformat(),
            "events_excluded_pre_start_count": int(pre_start_events),
            "heroes_by_day_last_30d": [
                {"date": str(d), "n": int(n)} for d, n in heroes_by_day
            ],
            "villains_by_day_last_30d": [
                {"date": str(d), "n": int(n)} for d, n in villains_by_day
            ],
        },
    }


class ForcePromotePayload(BaseModel):
    handles: list[str] = []
    ids: list[int] = []


@app.post("/api/admin/force-promote")
async def force_promote(req: ForcePromotePayload, db: Session = Depends(get_db)):
    """Escape hatch: force is_fashion=True on a hand-picked list of
    products by handle (Shopify slug) or DB id. Used to resolve
    Gemini false negatives the operator catches in manual review —
    Oktoberfest Ensembles, kids' sleep sacks, brand-cryptic titles
    that no automated check can resolve.

    Subniche is rewritten to 'fashion' unless the row already had a
    wearable subniche (jewelry/bags/accessories), which is preserved.
    Idempotent: items already on Fashion remain unchanged.
    """
    from scraper import WEARABLE_SUBNICHES

    if not req.handles and not req.ids:
        raise HTTPException(status_code=400, detail="Provide handles or ids.")

    q = db.query(Product)
    filters = []
    if req.handles:
        filters.append(Product.handle.in_(req.handles))
    if req.ids:
        filters.append(Product.id.in_(req.ids))
    rows = q.filter(or_(*filters)).all() if filters else []

    promoted = []
    for p in rows:
        was_fashion = bool(p.is_fashion)
        p.is_fashion = True
        sub = (p.subniche or "").strip().lower()
        if sub not in WEARABLE_SUBNICHES:
            p.subniche = "fashion"
        promoted.append({
            "id": p.id, "handle": p.handle, "title": p.title,
            "store": p.store.name if p.store else "",
            "previous_subniche": sub, "was_already_fashion": was_fashion,
        })
    if promoted:
        db.commit()
    return {"matched_count": len(rows), "promoted": promoted}


@app.post("/api/admin/reclassify-general")
async def reclassify_general(
    framing: str = Query(
        "strict",
        pattern="^(strict|broad)$",
        description="strict = body-worn focus; broad = fashion-adjacent",
    ),
    dry_run: bool = Query(
        False, description="When true, return the flagged list without flipping is_fashion."
    ),
    db: Session = Depends(get_db),
):
    """One-shot Gemini-driven sweep over the General tab. For every
    is_fashion=False row we currently track, ask Gemini directly:
    'is this worn / carried / used to dress?' and promote everything
    that comes back YES.

    The regex safety net (FORCE_FASHION_TITLE_RE) catches obvious
    multilingual cases at scrape time. This endpoint is the second
    line — it catches translingual edge cases, branded items with
    opaque titles, and image-only signals that the regex would miss.
    Idempotent: a second call after promotion only flags items that
    are STILL on General, so you can run it twice (strict then broad)
    to mop up edge cases.

    Skips items matching NON_PRODUCT_*_RE so junk (shipping protection,
    surprise boxes, gift cards) isn't mistakenly resurrected to Fashion.
    """
    from classifier import reclassify_general_with_gemini
    from scraper import _is_non_product, WEARABLE_SUBNICHES

    rows = (
        db.query(Product)
        .filter(Product.is_fashion == False, Product.subniche != "")
        .all()
    )
    candidates = []
    for p in rows:
        if _is_non_product(
            title=p.title or "", product_type=p.product_type or "",
            handle=p.handle or "", product_url=p.product_url or "",
            image_url=p.image_url or "",
        ):
            continue
        candidates.append({
            "_id": p.id,
            "title": p.title or "",
            "handle": p.handle or "",
            "product_type": p.product_type or "",
            "image_url": p.image_url or "",
            "store_name": p.store.name if p.store else "",
            "subniche": p.subniche or "",
        })

    flagged, errors = await reclassify_general_with_gemini(candidates, framing=framing)

    promoted = []
    if not dry_run:
        flagged_ids = {f["_id"] for f in flagged}
        id_to_reason = {f["_id"]: f.get("reclassify_reason", "") for f in flagged}
        for p in rows:
            if p.id not in flagged_ids:
                continue
            p.is_fashion = True
            sub = (p.subniche or "").strip().lower()
            if sub not in WEARABLE_SUBNICHES:
                p.subniche = "fashion"
            promoted.append({
                "id": p.id,
                "title": p.title,
                "store": p.store.name if p.store else "",
                "handle": p.handle,
                "previous_subniche": sub,
                "reason": id_to_reason.get(p.id, ""),
            })
        if promoted:
            db.commit()

    return {
        "framing": framing,
        "dry_run": dry_run,
        "candidates_scanned": len(candidates),
        "flagged_count": len(flagged),
        "promoted_count": len(promoted),
        "promoted": promoted,
        "errors": errors,
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
async def get_stats(
    days: int = Query(1, ge=1, le=LABEL_EVENT_RETENTION_DAYS),
    feed: Optional[str] = Query(None, pattern="^(fashion|general)$"),
    db: Session = Depends(get_db),
):
    """Top-of-page counter. Heroes / villains counts follow the
    `days` window AND the active tab — Fashion tab counts only
    fashion-feed events; General tab counts only general-feed events.

    `feed` query param:
      - "fashion"  → is_fashion=True
      - "general"  → is_fashion=False
      - omitted    → BOTH feeds combined (backward-compat for older
                     clients; new UI always sends explicit feed)

    Source of truth is the LabelEvent ledger (same as the feed
    endpoints) — de-duplicated by product_id, taking the most recent
    qualifying event per product. Pre-DATA_START_DATE events are
    excluded by fetch_label_events_window so the numbers here always
    match what the user sees in the hero/villain filtered grids.
    """
    total_stores = db.query(Store).count()
    total_products = db.query(Product).filter(Product.is_fashion == True).count()
    # Resolve `feed` to the is_fashion arg fetch_label_events_window expects:
    #   "fashion"  -> True  (only is_fashion=True products)
    #   "general"  -> False (only is_fashion=False products WITH subniche)
    #   None       -> None  (no filter, both feeds counted)
    if feed == "fashion":
        is_fashion_arg = True
    elif feed == "general":
        is_fashion_arg = False
    else:
        is_fashion_arg = None
    hero_pairs = fetch_label_events_window(
        db, label="hero", days=days, is_fashion=is_fashion_arg,
    )
    villain_pairs = fetch_label_events_window(
        db, label="villain", days=days, is_fashion=is_fashion_arg,
    )
    heroes = len(hero_pairs)
    villains = len(villain_pairs)
    # Health check — flag stores whose last scrape went visibly wrong.
    # Two failure modes worth alerting on:
    #
    #  (1) Dead scrape: 0 fashion + 0 general. Usually means a new-
    #      store config issue, Cloudflare block, or the upstream
    #      collection page is gone.
    #
    #  (2) Skewed feeds against the store's declared niche: e.g. a
    #      store with niche='General' returning 100+ fashion and 0
    #      general (mis-routing); or niche='Fashion & General' with
    #      only one feed populated. Pure-Fashion stores legitimately
    #      have general=0, and pure-General stores legitimately have
    #      fashion=0 — the niche field tells us what to expect.
    def _expects_both(niche: str) -> bool:
        n = (niche or "").lower()
        # Mixed-niche stores ("Fashion & General", "Fashion & HD",
        # "Multi") are expected to have both feeds. Pure single-niche
        # stores ("Fashion" / "General") are not.
        return ("&" in n) or ("multi" in n) or ("mixed" in n)

    unhealthy_stores = []
    for s in db.query(Store).all():
        fashion_n = sum(1 for p in s.products if p.is_fashion)
        general_n = sum(1 for p in s.products if not p.is_fashion and p.subniche)
        if fashion_n == 0 and general_n == 0:
            unhealthy_stores.append({
                "id": s.id, "name": s.name, "url": s.url,
                "niche": s.niche,
                "fashion": 0, "general": 0,
                "reason": "dead-scrape (both feeds empty)",
            })
            continue
        # Skew check only fires when the store's niche says BOTH feeds
        # should populate — single-niche stores legitimately have one
        # empty side and we don't want to spam alerts.
        if _expects_both(s.niche or ""):
            if (fashion_n > 100 and general_n == 0) or (general_n > 100 and fashion_n == 0):
                unhealthy_stores.append({
                    "id": s.id, "name": s.name, "url": s.url,
                    "niche": s.niche,
                    "fashion": fashion_n, "general": general_n,
                    "reason": (
                        "skewed-feeds (mixed-niche store but only one "
                        "feed populated — likely mis-routing)"
                    ),
                })
    if unhealthy_stores:
        # Surface as logged warning too so Railway logs flag it without
        # waiting for a /api/stats request to discover it.
        logger.warning(
            "Health check: %d unhealthy stores: %s",
            len(unhealthy_stores),
            [u["name"] for u in unhealthy_stores],
        )
    return {
        "stores": total_stores,
        "products": total_products,
        "heroes": heroes,
        "villains": villains,
        "days": days,
        "feed": feed,
        "data_start_date": DATA_START_DATE.date().isoformat(),
        "unhealthy_stores": unhealthy_stores,
    }

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
