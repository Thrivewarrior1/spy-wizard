import os
import asyncio
import logging
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import func
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from pydantic import BaseModel
from typing import Optional

from database import get_db, engine, SessionLocal
from models import Base, Store, Product, PositionHistory
from scraper import scrape_all_stores, scrape_store_bestsellers, update_products_in_db
from seed import seed_stores

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APP_PASSWORD = os.getenv("APP_PASSWORD", "mats2310")

scheduler = AsyncIOScheduler()

async def daily_scrape():
    """Run the daily scrape job."""
    logger.info("Running daily scrape job...")
    db = SessionLocal()
    try:
        await scrape_all_stores(db)
    finally:
        db.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    Base.metadata.create_all(bind=engine)
    seed_stores()

    # Schedule daily scrape at 6 AM UTC
    scheduler.add_job(daily_scrape, "cron", hour=6, minute=0)
    scheduler.start()
    logger.info("Scheduler started - daily scrape at 06:00 UTC")

    yield

    # Shutdown
    scheduler.shutdown()

app = FastAPI(title="Spy Wizard", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Auth ---
class LoginRequest(BaseModel):
    password: str

@app.post("/api/login")
async def login(req: LoginRequest):
    if req.password == APP_PASSWORD:
        return {"success": True, "token": "authenticated"}
    raise HTTPException(status_code=401, detail="Invalid password")

# --- Stores ---
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
    return [{
        "id": s.id,
        "name": s.name,
        "url": s.url,
        "monthly_visitors": s.monthly_visitors,
        "niche": s.niche,
        "country": s.country,
        "product_count": len(s.products),
        "created_at": s.created_at.isoformat() if s.created_at else None,
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

# --- Products / Bestsellers ---
def parse_visitors(visitors_str: str) -> int:
    """Convert '1.9M', '314K', etc. to an integer for sorting."""
    s = visitors_str.strip().upper()
    try:
        if s.endswith("M"):
            return int(float(s[:-1]) * 1_000_000)
        elif s.endswith("K"):
            return int(float(s[:-1]) * 1_000)
        else:
            return int(s)
    except (ValueError, TypeError):
        return 0

@app.get("/api/bestsellers/combined")
async def get_combined_bestsellers(
    sort: str = Query("high-low", regex="^(high-low|low-high|volume)$"),
    label: Optional[str] = Query(None, regex="^(hero|villain|normal)$"),
    limit: int = Query(30, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Get combined top bestsellers from all stores."""
    query = db.query(Product).join(Store)

    if label:
        query = query.filter(Product.label == label)

    products = query.all()

    # Sort based on parameter
    if sort == "volume":
        # Sort by store monthly visitors (highest first), then by position
        products.sort(key=lambda p: (-parse_visitors(p.store.monthly_visitors), p.current_position))
    elif sort == "low-high":
        products.sort(key=lambda p: p.current_position)
    else:  # high-low (default - best sellers first)
        products.sort(key=lambda p: p.current_position)

    products = products[:limit]

    return [{
        "id": p.id,
        "title": p.title,
        "image_url": p.image_url,
        "price": p.price,
        "vendor": p.vendor,
        "product_url": p.product_url,
        "current_position": p.current_position,
        "previous_position": p.previous_position,
        "label": p.label,
        "store_name": p.store.name,
        "store_url": p.store.url,
        "store_visitors": p.store.monthly_visitors,
        "last_scraped": p.last_scraped.isoformat() if p.last_scraped else None,
    } for p in products]

@app.get("/api/bestsellers/store/{store_id}")
async def get_store_bestsellers(
    store_id: int,
    sort: str = Query("high-low", regex="^(high-low|low-high|volume)$"),
    label: Optional[str] = Query(None, regex="^(hero|villain|normal)$"),
    limit: int = Query(30, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Get top bestsellers for a specific store."""
    store = db.query(Store).filter(Store.id == store_id).first()
    if not store:
        raise HTTPException(status_code=404, detail="Store not found")

    query = db.query(Product).filter(Product.store_id == store_id)

    if label:
        query = query.filter(Product.label == label)

    products = query.order_by(Product.current_position).limit(limit).all()

    return [{
        "id": p.id,
        "title": p.title,
        "image_url": p.image_url,
        "price": p.price,
        "vendor": p.vendor,
        "product_url": p.product_url,
        "current_position": p.current_position,
        "previous_position": p.previous_position,
        "label": p.label,
        "store_name": store.name,
        "store_url": store.url,
        "last_scraped": p.last_scraped.isoformat() if p.last_scraped else None,
    } for p in products]

# --- Manual Scrape Trigger ---
@app.post("/api/scrape")
async def trigger_scrape(db: Session = Depends(get_db)):
    """Manually trigger a scrape of all stores."""
    await scrape_all_stores(db)
    return {"success": True, "message": "Scrape completed"}

@app.post("/api/scrape/{store_id}")
async def trigger_store_scrape(store_id: int, db: Session = Depends(get_db)):
    """Manually trigger a scrape for a specific store."""
    store = db.query(Store).filter(Store.id == store_id).first()
    if not store:
        raise HTTPException(status_code=404, detail="Store not found")

    products = await scrape_store_bestsellers(store.url)
    if products:
        update_products_in_db(db, store, products)
    return {"success": True, "products_found": len(products)}

# --- Serve Frontend ---
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")

@app.get("/")
async def serve_root():
    index_path = os.path.join(frontend_path, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>Spy Wizard API</h1><p>Frontend not found.</p>")

@app.get("/{path:path}")
async def serve_static(path: str):
    file_path = os.path.join(frontend_path, path)
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return FileResponse(file_path)
    # Fallback to index.html for SPA routing
    index_path = os.path.join(frontend_path, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Not found")
