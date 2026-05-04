"""First-run seeding of the Store table.

CRITICAL: this runs ONLY when the Store table is empty (first-ever
deploy or after a manual wipe). After that, the DB is the source of
truth — the user can add new stores via POST /api/stores and delete
them via DELETE /api/stores/{id}, and those edits MUST survive every
Railway redeploy.

Previously the loop iterated INITIAL_STORES on every startup and
re-created any missing rows, which clobbered user deletes. The bug:
delete a store, redeploy, store comes back. Fixed here by gating the
whole upsert behind `db.query(Store).count() == 0`.
"""
import logging

from database import SessionLocal, engine
from models import Base, Store

logger = logging.getLogger(__name__)

INITIAL_STORES = [
    {"name": "Novi Good", "url": "https://novigood.com/", "monthly_visitors": "24K", "niche": "Fashion", "country": "AUS"},
    {"name": "Luna Mae", "url": "https://www.luna-mae.com/", "monthly_visitors": "52K", "niche": "Fashion", "country": "AUS, USA"},
    {"name": "Zoe Sydney", "url": "https://zoesydney.com/", "monthly_visitors": "108K", "niche": "Fashion", "country": "AUS, CA, USA, UK, NZ"},
    {"name": "Hudson Grace", "url": "https://hudsonclaye.com/", "monthly_visitors": "314K", "niche": "Fashion", "country": "UK"},
    {"name": "Breuermode", "url": "https://breuermode.de/", "monthly_visitors": "50K", "niche": "Fashion & HD", "country": "DE"},
    {"name": "Gents of Brittain", "url": "https://gentsofbritain.com/", "monthly_visitors": "24K", "niche": "Fashion", "country": "UK"},
    {"name": "Heidi Mode", "url": "https://heidi-mode.de/", "monthly_visitors": "1.9M", "niche": "Fashion", "country": "DE"},
    {"name": "Mullers Modehaus", "url": "https://www.mullers-modehaus.com/", "monthly_visitors": "79K", "niche": "Fashion", "country": "DE"},
    {"name": "Falken Stein", "url": "https://falken-stein.de/", "monthly_visitors": "75K", "niche": "Fashion", "country": "DE"},
    {"name": "Alexander Hampton", "url": "https://alexander-hampton.com/", "monthly_visitors": "190K", "niche": "Fashion & General", "country": "MultiMarket. Mostly USA"},
    {"name": "Ancienne Monde", "url": "https://anciennemonde.fr/", "monthly_visitors": "68K", "niche": "Fashion", "country": "MultiMarket. Mostly UK & USA"},
    {"name": "Lumenrosa", "url": "https://www.lumenrosa.de/", "monthly_visitors": "54K", "niche": "Fashion", "country": "DE"},
]


def seed_stores(db=None) -> int:
    """Insert INITIAL_STORES ONLY when the Store table is empty.
    Returns the number of rows inserted (0 when the table already has
    data, meaning this is a redeploy and the user has existing stores
    we MUST NOT touch).

    Pass `db` for testing; otherwise a session is created and closed
    locally.
    """
    Base.metadata.create_all(bind=engine)
    own_session = db is None
    if own_session:
        db = SessionLocal()
    try:
        existing_count = db.query(Store).count()
        if existing_count > 0:
            logger.info(
                "seed_stores: skipping — Store table already has %d rows; "
                "user-managed catalog is the source of truth from here",
                existing_count,
            )
            return 0
        logger.info(
            "seed_stores: empty Store table — first-run seeding %d initial stores",
            len(INITIAL_STORES),
        )
        for store_data in INITIAL_STORES:
            db.add(Store(**store_data))
        db.commit()
        return len(INITIAL_STORES)
    finally:
        if own_session:
            db.close()


if __name__ == "__main__":
    inserted = seed_stores()
    print(f"Seeded {inserted} stores (0 = table was already populated)")
