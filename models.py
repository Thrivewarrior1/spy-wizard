from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Enum as SQLEnum, Text, Boolean
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime
import enum

Base = declarative_base()

class LabelEnum(str, enum.Enum):
    HERO = "hero"
    VILLAIN = "villain"
    NORMAL = "normal"

class Store(Base):
    __tablename__ = "stores"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    url = Column(String(500), nullable=False, unique=True)
    monthly_visitors = Column(String(50), default="0")
    niche = Column(String(100), default="Fashion")
    country = Column(String(255), default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    products = relationship(
        "Product",
        back_populates="store",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    store_id = Column(
        Integer, ForeignKey("stores.id", ondelete="CASCADE"), nullable=False,
    )
    # Shopify product handles can be 200+ chars on stores with long German /
    # Dutch / French SEO-friendly product names. Use Text (unlimited) for
    # any field that holds a handle or a derived value to avoid silent
    # transaction rollbacks on long values.
    shopify_id = Column(Text)
    title = Column(Text, nullable=False)
    handle = Column(Text)
    image_url = Column(Text)
    price = Column(String(50))
    vendor = Column(String(255))
    product_type = Column(String(255))
    product_url = Column(Text)

    current_position = Column(Integer, default=0)
    previous_position = Column(Integer, default=0)
    label = Column(String(20), default=LabelEnum.NORMAL.value)

    ai_tags = Column(Text, default="")
    is_fashion = Column(Boolean, default=True)
    # Subniche category for non-fashion items shown on the General tab.
    # Gemini picks ONE of: jewelry, accessories, electronics, home, beauty,
    # health, food, toys-books, services, other. Empty string for products
    # that haven't been (re-)classified since this column was added — the
    # General feed filters those out.
    subniche = Column(String(50), default="")
    # Fine-grained product category from the curated PRODUCT_CATEGORIES
    # catalog (chandelier, table-lamp, sneaker, dress, bra, smartwatch,
    # phone-case, etc.). Used by the hybrid search so a query for
    # "chandelier" exhaustively returns chandelier products even when
    # the title is opaque ("Aurora Crystal Pendant"). Empty string for
    # products the regex classifier couldn't categorise — those still
    # match via title/ai_tags but don't surface in strict-category
    # searches.
    product_category = Column(String(64), default="", index=True)

    last_scraped = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    store = relationship("Store", back_populates="products")
    history = relationship(
        "PositionHistory",
        back_populates="product",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

class PositionHistory(Base):
    __tablename__ = "position_history"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(
        Integer, ForeignKey("products.id", ondelete="CASCADE"), nullable=False,
    )
    position = Column(Integer, nullable=False)
    date = Column(DateTime, default=datetime.utcnow)

    product = relationship("Product", back_populates="history")
