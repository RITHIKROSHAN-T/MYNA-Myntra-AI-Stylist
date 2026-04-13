# models.py
# Defines all 4 database table structures for MYNA

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

# ── Table 1: Users ─────────────────────────────────────────────
@dataclass
class User:
    user_id: str          # UUID primary key
    name: str             # full name
    email: str            # email address
    created_at: datetime  # account creation time
    last_login: datetime  # last login time

# ── Table 2: Outfits ───────────────────────────────────────────
@dataclass
class Outfit:
    outfit_id: str        # UUID primary key
    user_id: str          # FK → Users
    photo_url: str        # S3 URL of uploaded/generated photo
    style_type: str       # e.g. Nano-Banana, Casual, Party
    top_label: str        # detected top type
    bottom_label: str     # detected bottom type
    accessory_labels: str # comma-separated accessories
    created_at: datetime  # when outfit was generated

# ── Table 3: Product Catalog ───────────────────────────────────
@dataclass
class Product:
    product_id: str       # unique product identifier
    name: str             # product name
    category: str         # Top / Bottom / Footwear / Accessories
    price: float          # selling price
    image_url: str        # product image URL
    product_url: str      # Myntra product link
    brand: str            # brand name
    color: str            # primary color
    created_at: datetime  # when added

# ── Table 4: Cart Recommendations ──────────────────────────────
@dataclass
class CartRecommendation:
    entry_id: str          # UUID primary key
    user_id: str           # FK → Users
    outfit_id: str         # FK → Outfits
    product_id: str        # FK → Product Catalog
    component_type: str    # Top / Bottom / Accessories / Footwear
    added_to_cart: bool    # True if user added to cart
    recommended_at: datetime
    purchased: bool        # True if purchased