# db_manager.py
# Step 3: PostgreSQL persistence layer
# Implements the 4-table schema from the project spec using psycopg2.
# Reads credentials from .env — swap DB_HOST to your AWS RDS endpoint for prod.

import uuid
import os
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

# Load .env from project root (two levels above this file: src/database/ → root)
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

_DB_CONFIG = {
    "host":     os.getenv("DB_HOST",     "localhost"),
    "port":     int(os.getenv("DB_PORT", "5432")),
    "dbname":   os.getenv("DB_NAME",     "myna_db"),
    "user":     os.getenv("DB_USER",     "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}


# ─────────────────────────────────────────────────────────────────────────────
# Connection helper
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def _get_conn():
    """Yield a psycopg2 connection, auto-commit on success, rollback on error."""
    conn = psycopg2.connect(**_DB_CONFIG)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Table initialisation  (run once on startup)
# ─────────────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """
    Create all four project tables if they do not already exist.

    Tables match the schema defined in the project spec:
        users, outfits, product_catalog, cart_recommendations
    """
    sql = """
    CREATE TABLE IF NOT EXISTS users (
        user_id    TEXT PRIMARY KEY,
        name       VARCHAR(100)  NOT NULL DEFAULT 'Guest',
        email      VARCHAR(100)  NOT NULL DEFAULT '',
        created_at TIMESTAMP     NOT NULL DEFAULT NOW(),
        last_login TIMESTAMP     NOT NULL DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS outfits (
        outfit_id        TEXT PRIMARY KEY,
        user_id          TEXT          NOT NULL,
        photo_url        VARCHAR(500)  NOT NULL DEFAULT '',
        style_type       VARCHAR(100)  NOT NULL DEFAULT 'AI-Generated',
        top_label        VARCHAR(50)   NOT NULL DEFAULT '',
        bottom_label     VARCHAR(50)   NOT NULL DEFAULT '',
        accessory_labels TEXT          NOT NULL DEFAULT '',
        created_at       TIMESTAMP     NOT NULL DEFAULT NOW(),
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    );

    CREATE TABLE IF NOT EXISTS product_catalog (
        product_id  VARCHAR(50)  PRIMARY KEY,
        name        VARCHAR(200) NOT NULL DEFAULT '',
        category    VARCHAR(50)  NOT NULL DEFAULT '',
        price       DECIMAL(10,2) NOT NULL DEFAULT 0,
        image_url   VARCHAR(500) NOT NULL DEFAULT '',
        product_url VARCHAR(500) NOT NULL DEFAULT '',
        brand       VARCHAR(100) NOT NULL DEFAULT '',
        color       VARCHAR(50)  NOT NULL DEFAULT '',
        created_at  TIMESTAMP    NOT NULL DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS cart_recommendations (
        entry_id       TEXT PRIMARY KEY,
        user_id        TEXT         NOT NULL,
        outfit_id      TEXT         NOT NULL DEFAULT '',
        product_id     VARCHAR(50)  NOT NULL,
        component_type VARCHAR(50)  NOT NULL DEFAULT '',
        added_to_cart  BOOLEAN      NOT NULL DEFAULT FALSE,
        recommended_at TIMESTAMP    NOT NULL DEFAULT NOW(),
        purchased      BOOLEAN      NOT NULL DEFAULT FALSE,
        FOREIGN KEY (user_id)   REFERENCES users(user_id),
        FOREIGN KEY (outfit_id) REFERENCES outfits(outfit_id)
    );
    """
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
    print("[DB] Tables initialised.")


# ─────────────────────────────────────────────────────────────────────────────
# Users
# ─────────────────────────────────────────────────────────────────────────────

def get_or_create_user(session_id: str, name: str = "Guest") -> str:
    """
    Return user_id for this session, inserting a new row if needed.
    `session_id` (generated from timestamp in the app) doubles as user_id.
    """
    now = datetime.now()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id FROM users WHERE user_id = %s",
                (session_id,)
            )
            row = cur.fetchone()

            if row:
                cur.execute(
                    "UPDATE users SET last_login = %s WHERE user_id = %s",
                    (now, session_id)
                )
            else:
                cur.execute(
                    """INSERT INTO users (user_id, name, email, created_at, last_login)
                       VALUES (%s, %s, %s, %s, %s)""",
                    (session_id, name, "", now, now)
                )
    return session_id


# ─────────────────────────────────────────────────────────────────────────────
# Outfits
# ─────────────────────────────────────────────────────────────────────────────

def save_outfit(
    user_id:          str,
    detection_result: dict,
    photo_url:        str = "",
    style_type:       str = "AI-Generated",
) -> str:
    """
    Persist a detected outfit record to the outfits table.

    Args:
        user_id:          session identifier (FK → users)
        detection_result: dict from component_detector.analyze_outfit()
        photo_url:        S3 URL or local path of the photo (stored for Step 7)
        style_type:       style label or prompt text (max 100 chars)

    Returns:
        outfit_id (str UUID)
    """
    outfit_id = detection_result.get("outfit_id", str(uuid.uuid4()))
    now       = datetime.now()

    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO outfits
                   (outfit_id, user_id, photo_url, style_type,
                    top_label, bottom_label, accessory_labels, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (outfit_id) DO UPDATE
                       SET style_type = EXCLUDED.style_type,
                           top_label  = EXCLUDED.top_label,
                           bottom_label = EXCLUDED.bottom_label""",
                (
                    outfit_id,
                    user_id,
                    photo_url,
                    style_type[:100],
                    detection_result.get("top_label", ""),
                    detection_result.get("bottom_label", ""),
                    detection_result.get("accessory_labels", ""),
                    now,
                ),
            )
    return outfit_id


def get_recent_outfits(user_id: str, limit: int = 5) -> list:
    """Return the most recent outfits for a given user."""
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """SELECT outfit_id, style_type, top_label, bottom_label,
                          accessory_labels, created_at
                   FROM   outfits
                   WHERE  user_id = %s
                   ORDER BY created_at DESC
                   LIMIT  %s""",
                (user_id, limit),
            )
            return [dict(r) for r in cur.fetchall()]


# ─────────────────────────────────────────────────────────────────────────────
# Recommendations & Cart
# ─────────────────────────────────────────────────────────────────────────────

def save_recommendations(
    user_id:   str,
    outfit_id: str,
    results:   dict,
) -> None:
    """
    Persist recommended products and create cart_recommendations rows.

    Args:
        user_id:   session identifier
        outfit_id: FK to outfits table
        results:   {component_type: [product_dict, ...]} from recommender.py
    """
    now = datetime.now()

    with _get_conn() as conn:
        with conn.cursor() as cur:
            for component_type, products in results.items():
                for p in products[:5]:
                    pid = str(p.get("product_id", "")).strip()
                    if not pid:
                        continue

                    # Upsert into product_catalog
                    cur.execute(
                        """INSERT INTO product_catalog
                           (product_id, name, category, price,
                            image_url, product_url, brand, color, created_at)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                           ON CONFLICT (product_id) DO NOTHING""",
                        (
                            pid,
                            str(p.get("name", ""))[:200],
                            component_type,
                            float(p.get("price", 0)),
                            str(p.get("image_url", "")),
                            str(p.get("product_url", "")),
                            str(p.get("brand", ""))[:100],
                            str(p.get("color", ""))[:50],
                            now,
                        ),
                    )

                    # Insert recommendation record (skip duplicates)
                    cur.execute(
                        """INSERT INTO cart_recommendations
                           (entry_id, user_id, outfit_id, product_id,
                            component_type, added_to_cart, recommended_at, purchased)
                           VALUES (%s, %s, %s, %s, %s, FALSE, %s, FALSE)
                           ON CONFLICT DO NOTHING""",
                        (
                            str(uuid.uuid4()),
                            user_id,
                            outfit_id,
                            pid,
                            component_type,
                            now,
                        ),
                    )


def update_cart_status(user_id: str, product_id: str, added: bool = True) -> None:
    """Mark a recommended product as added-to-cart (or removed)."""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE cart_recommendations
                   SET added_to_cart = %s
                   WHERE user_id = %s AND product_id = %s""",
                (added, user_id, product_id),
            )


def mark_purchased(user_id: str, product_id: str) -> None:
    """Mark a product as purchased."""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE cart_recommendations
                   SET purchased = TRUE
                   WHERE user_id = %s AND product_id = %s""",
                (user_id, product_id),
            )


# ─────────────────────────────────────────────────────────────────────────────
# Analytics helpers  (used in Step 8)
# ─────────────────────────────────────────────────────────────────────────────

def get_session_stats(user_id: str) -> dict:
    """Return basic metrics for one session."""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM outfits WHERE user_id = %s",
                (user_id,)
            )
            outfits_count = cur.fetchone()[0]

            cur.execute(
                "SELECT COUNT(*) FROM cart_recommendations WHERE user_id = %s",
                (user_id,)
            )
            recs_count = cur.fetchone()[0]

            cur.execute(
                """SELECT COUNT(*) FROM cart_recommendations
                   WHERE user_id = %s AND added_to_cart = TRUE""",
                (user_id,)
            )
            cart_count = cur.fetchone()[0]

    return {
        "outfits_generated":    outfits_count,
        "products_recommended": recs_count,
        "items_in_cart":        cart_count,
    }


def get_all_stats() -> dict:
    """Aggregate metrics across all users (for the analytics dashboard in Step 8)."""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(DISTINCT user_id) FROM users")
            total_users = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM outfits")
            total_outfits = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM cart_recommendations")
            total_recs = cur.fetchone()[0]

            cur.execute(
                "SELECT COUNT(*) FROM cart_recommendations WHERE added_to_cart = TRUE"
            )
            total_cart = cur.fetchone()[0]

            cur.execute(
                "SELECT COUNT(*) FROM cart_recommendations WHERE purchased = TRUE"
            )
            total_purchases = cur.fetchone()[0]

            cur.execute(
                """SELECT component_type, COUNT(*) AS cnt
                   FROM   cart_recommendations
                   GROUP  BY component_type
                   ORDER  BY cnt DESC"""
            )
            top_components = [
                {"type": r[0], "count": r[1]} for r in cur.fetchall()
            ]

            cur.execute(
                """SELECT style_type, COUNT(*) AS cnt
                   FROM   outfits
                   GROUP  BY style_type
                   ORDER  BY cnt DESC
                   LIMIT  5"""
            )
            top_styles = [
                {"style": r[0], "count": r[1]} for r in cur.fetchall()
            ]

    return {
        "total_users":       total_users,
        "total_outfits":     total_outfits,
        "total_recommended": total_recs,
        "total_in_cart":     total_cart,
        "total_purchased":   total_purchases,
        "top_components":    top_components,
        "top_styles":        top_styles,
        "cart_rate":         round(total_cart / total_recs * 100, 1) if total_recs else 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Smoke-test  (run directly: python db_manager.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    print("Initialising DB tables …")
    init_db()

    uid = get_or_create_user("test-session-001", "Test User")
    print(f"User OK: {uid}")

    fake_detection = {
        "outfit_id":        "outfit-test-001",
        "component_types":  ["Topwear", "Bottomwear"],
        "top_label":        "blue top",
        "bottom_label":     "black bottom",
        "footwear_label":   "",
        "accessory_labels": "",
    }
    oid = save_outfit(uid, fake_detection, style_type="Casual Test")
    print(f"Outfit saved: {oid}")

    fake_results = {
        "Top": [{
            "product_id":  "P-TEST-001",
            "name":        "Test Blue Shirt",
            "price":       999,
            "image_url":   "https://example.com/img.jpg",
            "product_url": "https://myntra.com/p001",
            "brand":       "TestBrand",
            "color":       "Blue",
        }]
    }
    save_recommendations(uid, oid, fake_results)
    print("Recommendations saved.")

    print("Session stats:", json.dumps(get_session_stats(uid), indent=2))
    print("All stats:",     json.dumps(get_all_stats(),         indent=2))
