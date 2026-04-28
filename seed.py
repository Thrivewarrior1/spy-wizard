"""Seed the database with initial competitor stores."""
from database import SessionLocal, engine
from models import Base, Store

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

def seed_stores():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        for store_data in INITIAL_STORES:
            existing = db.query(Store).filter(Store.url == store_data["url"]).first()
            if not existing:
                store = Store(**store_data)
                db.add(store)
                print(f"Added: {store_data['name']}")
            else:
                print(f"Already exists: {store_data['name']}")
        db.commit()
        print(f"\nTotal stores: {db.query(Store).count()}")
    finally:
        db.close()

if __name__ == "__main__":
    seed_stores()
