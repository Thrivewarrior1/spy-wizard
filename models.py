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

    products = relationship("Product", back_populates="store", cascade="all, delete-orphan")

class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    store_id = Column(Integer, ForeignKey("stores.id"), nullable=False)
    shopify_id = Column(String(100))
    title = Column(String(500), nullable=False)
    handle = Column(String(500))
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

    last_scraped = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    store = relationship("Store", back_populates="products")
    history = relationship("PositionHistory", back_populates="product", cascade="all, delete-orphan")

class PositionHistory(Base):
    __tablename__ = "position_history"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    position = Column(Integer, nullable=False)
    date = Column(DateTime, default=datetime.utcnow)

    product = relationship("Product", back_populates="history")
