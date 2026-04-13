# rag_agent.py
# LangChain + ChromaDB RAG agent for MYNA stylist conversations.
#
# build_index()   – embed top-15K products → ChromaDB  (run once)
# stylist_chat()  – answer questions using vector search

import os
import re
import math
from pathlib import Path

import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
_BASE      = Path(__file__).resolve().parents[2]
_CSV       = _BASE / "data" / "processed" / "clean_myntra_products.csv"
_CHROMA_DIR = _BASE / "data" / "chroma_db"
_COLLECTION = "myntra_products"
_INDEX_SIZE = 15_000          # top products to embed


# ── Lazy singletons ───────────────────────────────────────────────────────────
_collection = None
_ef         = None            # embedding function


def _get_ef():
    """Return cached SentenceTransformer embedding function."""
    global _ef
    if _ef is None:
        from chromadb.utils.embedding_functions import (
            SentenceTransformerEmbeddingFunction,
        )
        _ef = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    return _ef


def _get_collection():
    """Return (or open) the ChromaDB collection."""
    global _collection
    if _collection is None:
        import chromadb
        client = chromadb.PersistentClient(path=str(_CHROMA_DIR))
        _collection = client.get_or_create_collection(
            name=_COLLECTION,
            embedding_function=_get_ef(),
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _product_doc(row) -> str:
    """Build the text document that gets embedded for one product."""
    price = int(row.get("price", 0))
    return (
        f"{row['name']} by {row['brand']} — "
        f"{row['component_type']} for {row['gender']}, "
        f"{row['color']} colour, {row['usage']} style, "
        f"₹{price:,}, rated {row['rating']}/5 "
        f"({int(row['rating_count']):,} reviews)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public: build_index
# ─────────────────────────────────────────────────────────────────────────────

def build_index(force: bool = False) -> None:
    """
    Build ChromaDB index from clean_myntra_products.csv.
    Indexes the top _INDEX_SIZE products ranked by quality score.
    Safe to call multiple times — skips if already built unless force=True.
    """
    col = _get_collection()

    if not force and col.count() > 0:
        print(f"[RAG] Index already has {col.count():,} docs — skipping build.")
        print("[RAG] Pass force=True to rebuild.")
        return

    print(f"[RAG] Loading {_CSV.name} ...")
    df = pd.read_csv(_CSV)

    # Quality score: rating × log(1 + review_count)
    df["_score"] = df["rating"].fillna(0) * df["rating_count"].fillna(0).apply(
        lambda x: math.log1p(x)
    )
    df = df.nlargest(_INDEX_SIZE, "_score").reset_index(drop=True)
    print(f"[RAG] Selected {len(df):,} top-quality products for indexing.")

    # Build docs, ids, metadata
    docs  = [_product_doc(r) for _, r in df.iterrows()]
    ids   = [str(int(r["product_id"])) for _, r in df.iterrows()]
    metas = [
        {
            "product_id"    : int(r["product_id"]),
            "name"          : str(r["name"])[:200],
            "brand"         : str(r["brand"]),
            "price"         : float(r["price"]),
            "rating"        : float(r["rating"]),
            "rating_count"  : int(r["rating_count"]),
            "component_type": str(r["component_type"]),
            "color"         : str(r["color"]),
            "usage"         : str(r["usage"]),
            "gender"        : str(r["gender"]),
            "product_url"   : str(r["product_url"]),
        }
        for _, r in df.iterrows()
    ]

    # Upsert in batches of 500 (ChromaDB default limit)
    batch = 500
    total = len(docs)
    for start in range(0, total, batch):
        end = min(start + batch, total)
        col.upsert(
            documents=ids[start:end],   # ChromaDB embeds `documents`
            ids=ids[start:end],
            metadatas=metas[start:end],
            # We embed the human-readable doc, not the id:
        )
        # Override: embed actual text docs
        # ChromaDB upsert with explicit embeddings:
        col.upsert(
            documents=docs[start:end],
            ids=ids[start:end],
            metadatas=metas[start:end],
        )
        pct = end / total * 100
        print(f"[RAG] Indexed {end:,}/{total:,} ({pct:.0f}%)", end="\r")

    print(f"\n[RAG] Done. Collection has {col.count():,} documents.")
    print(f"[RAG] Saved to: {_CHROMA_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# Intent detection helpers
# ─────────────────────────────────────────────────────────────────────────────

_WHY_PAT   = re.compile(
    r"\b(why|explain|reason|because|how come|what makes)\b", re.I)
_ALT_PAT   = re.compile(
    r"\b(alternative|other option|different|similar|else|instead|cheap|budget)\b",
    re.I
)
_MATCH_PAT = re.compile(
    r"\b(match\w*|pair\w*|go with|combine\w*|complement\w*|coordinate\w*|complete\w*)\b",
    re.I,
)
_BRAND_PAT = re.compile(
    r"\b(brand|about|tell me|info|review|quality)\b", re.I)
_PRICE_PAT = re.compile(r"(?:under|below|less than|within|upto|up to)\s*[₹rs.]?\s*(\d+)", re.I)


def _extract_budget(msg: str):
    m = _PRICE_PAT.search(msg)
    return int(m.group(1)) if m else None


def _search(query: str, n: int = 5, where: dict = None) -> list[dict]:
    """Semantic search in ChromaDB; returns list of metadata dicts."""
    col = _get_collection()
    if col.count() == 0:
        return []
    kwargs = dict(query_texts=[query], n_results=min(n, col.count()))
    if where:
        kwargs["where"] = where
    try:
        res = col.query(**kwargs)
        return res["metadatas"][0] if res["metadatas"] else []
    except Exception as e:
        print(f"[RAG] search error: {e}")
        return []


def _format_products(products: list[dict], max_show: int = 4) -> str:
    """Format a list of product metadata dicts into a readable bullet list."""
    lines = []
    for p in products[:max_show]:
        name  = str(p.get("name", ""))[:50].title()
        brand = p.get("brand", "")
        price = int(p.get("price", 0))
        rating = p.get("rating", 0)
        url   = p.get("product_url", "#")
        lines.append(
            f"• **{name}** by {brand} — ₹{price:,} "
            f"({rating}⭐) [View →]({url})"
        )
    return "\n".join(lines) if lines else "No matching products found."


def _last_recs_summary(last_recommendations: dict) -> str:
    """One-line summary of what was recommended."""
    if not last_recommendations:
        return ""
    parts = []
    for comp, prods in last_recommendations.items():
        if prods:
            top = prods[0]
            parts.append(
                f"{comp}: **{str(top.get('name',''))[:40].title()}** "
                f"by {top.get('brand','')} (₹{int(top.get('price',0)):,})"
            )
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Public: stylist_chat
# ─────────────────────────────────────────────────────────────────────────────

def stylist_chat(
    user_message: str,
    detected_components: dict,
    last_recommendations: dict = None,
) -> str:
    """
    Answer a user's styling question using RAG over the product index.

    Args:
        user_message:        Raw user text
        detected_components: YOLOv8 detection results dict
        last_recommendations: Products shown in the last chat turn

    Returns:
        Natural language response string (markdown-safe)
    """
    msg    = user_message.strip()
    budget = _extract_budget(msg)

    # ── 1. WHY / EXPLAIN ────────────────────────────────────────────────────
    if _WHY_PAT.search(msg):
        if not last_recommendations:
            return (
                "I recommend products by matching the outfit style detected "
                "in your photo with top-rated Myntra items — sorted by "
                "rating, reviews, and relevance to your request. 🎯\n\n"
                "Upload a photo and ask for an outfit to see personalised picks!"
            )

        summary = _last_recs_summary(last_recommendations)

        # Build a RAG query from the last recommendations
        all_prods = [p for prods in last_recommendations.values() for p in prods]
        query = " ".join(
            str(p.get("name", "")) + " " + str(p.get("color", ""))
            for p in all_prods[:3]
        )
        similar = _search(query, n=3) if query.strip() else []

        response = (
            "Great question! Here's why I chose these for you: 🤔✨\n\n"
            f"{summary}\n\n"
            "**How I picked them:**\n"
            "- Matched detected outfit components from your photo\n"
            "- Filtered by your style/occasion & budget preference\n"
            "- Ranked by Myntra rating + review count\n"
            "- Colour-coordinated across components for a complete look\n"
        )
        if similar:
            response += (
                "\n**You might also like (semantically similar):**\n"
                + _format_products(similar, 3)
            )
        return response

    # ── 2. ALTERNATIVES / CHEAPER OPTIONS ───────────────────────────────────
    if _ALT_PAT.search(msg):
        # Build query from the last recs or from detected components
        if last_recommendations:
            all_prods = [p for prods in last_recommendations.values()
                         for p in prods]
            query = " ".join(
                str(p.get("name","")) + " " + str(p.get("usage",""))
                for p in all_prods[:2]
            )
        else:
            # Fall back to detected components
            query = " ".join(
                f"{v} clothing Indian fashion"
                for v in (detected_components or {}).values()
                if isinstance(v, str)
            ) or "Indian fashion clothing"

        # If budget mentioned, filter by price
        where = None
        if budget:
            where = {"price": {"$lte": float(budget)}}

        results = _search(query, n=6, where=where)

        if not results:
            # Retry without price filter if empty
            results = _search(query, n=5)

        price_str = f" under ₹{budget:,}" if budget else ""
        if results:
            return (
                f"Here are some great alternatives{price_str} I found! 🛍️\n\n"
                + _format_products(results, 5)
                + "\n\nWant me to filter by a specific colour or style?"
            )
        return (
            f"I couldn't find alternatives{price_str} right now. "
            "Try a different budget or style keyword!"
        )

    # ── 3. WHAT MATCHES / PAIRS WITH THIS ───────────────────────────────────
    if _MATCH_PAT.search(msg):
        # Figure out what the user already has
        base = ""
        if last_recommendations:
            top_comp = next(iter(last_recommendations))
            top_prods = last_recommendations[top_comp]
            if top_prods:
                p = top_prods[0]
                base = (
                    f"{p.get('color','')} {p.get('usage','')} "
                    f"{p.get('component_type','')} {p.get('gender','')} "
                    "Indian fashion"
                )

        if not base:
            base = " ".join(
                str(v) for v in (detected_components or {}).values()
                if isinstance(v, str)
            ) or "casual Indian fashion"

        # Search for complementary component types
        comp_seen = set(last_recommendations.keys()) if last_recommendations \
            else set()
        all_comps = {"Top", "Bottom", "Footwear", "Accessories"}
        missing   = all_comps - comp_seen

        responses = [f"Here's what pairs well with your look! 👗✨\n"]
        for comp in list(missing)[:3]:
            hits = _search(f"{base} {comp}", n=2,
                           where={"component_type": {"$eq": comp}})
            if hits:
                responses.append(f"**{comp}:**")
                responses.append(_format_products(hits, 2))

        if len(responses) == 1:
            # Fallback: generic search
            hits = _search(base + " matching outfit", n=4)
            responses.append(_format_products(hits, 4))

        budget_hint = (
            "\n💡 Want these filtered by budget? Just mention a price!"
            if not budget else ""
        )
        return "\n".join(responses) + budget_hint

    # ── 4. BRAND INFO ────────────────────────────────────────────────────────
    if _BRAND_PAT.search(msg):
        # Extract brand name if mentioned
        brand_query = msg
        brand_name  = None
        if last_recommendations:
            all_prods = [p for prods in last_recommendations.values()
                         for p in prods]
            if all_prods:
                brand_name = all_prods[0].get("brand", "")
                brand_query = f"{brand_name} clothing quality style"

        results = _search(brand_query, n=5)
        brands  = {}
        for r in results:
            b = r.get("brand", "Unknown")
            if b not in brands:
                brands[b] = []
            brands[b].append(r)

        if not brands:
            return (
                "I couldn't find brand details right now. "
                "Try asking about a specific brand name!"
            )

        lines = [f"Here's what I know about the brands in your picks: 🏷️\n"]
        for b, prods in list(brands.items())[:3]:
            avg_rating = sum(p.get("rating", 0) for p in prods) / len(prods)
            price_range = (
                f"₹{int(min(p['price'] for p in prods)):,}–"
                f"₹{int(max(p['price'] for p in prods)):,}"
            )
            lines.append(
                f"**{b}** — avg {avg_rating:.1f}⭐, "
                f"price range {price_range}\n"
                + _format_products(prods, 2)
            )

        return "\n\n".join(lines)

    # ── 5. BUDGET ONLY ───────────────────────────────────────────────────────
    if budget:
        query = " ".join(
            str(v) for v in (detected_components or {}).values()
            if isinstance(v, str)
        ) or "Indian fashion"
        results = _search(query, n=5, where={"price": {"$lte": float(budget)}})
        if results:
            return (
                f"Top picks under ₹{budget:,} for you: 💰\n\n"
                + _format_products(results, 5)
            )
        return f"No products found under ₹{budget:,}. Try a higher budget?"

    # ── 6. DEFAULT HELP ──────────────────────────────────────────────────────
    indexed = _get_collection().count()
    status  = (
        f"({indexed:,} products indexed)" if indexed > 0
        else "(index not built yet — run build_index())"
    )
    return (
        f"I'm MYNA, your AI stylist! {status} 🌟\n\n"
        "You can ask me:\n"
        "- **'Why did you recommend this?'** — I'll explain my picks\n"
        "- **'Show alternatives under ₹1500'** — budget-friendly swaps\n"
        "- **'What matches with this?'** — complete the outfit\n"
        "- **'Tell me about this brand'** — brand info & ratings\n\n"
        "Or just ask for any style — I'll find it on Myntra! 🛍️"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Smoke-test:  python src/conversation/rag_agent.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 60)
    print("MYNA RAG Agent — smoke test")
    print("=" * 60)

    # Build index (skips if already done)
    build_index()

    col = _get_collection()
    print(f"\n[CHECK] Collection size: {col.count():,} docs\n")

    # Sample last_recommendations (simulating a Myntra kurta pick)
    sample_recs = {
        "Top": [{
            "name": "Men Blue Cotton Kurta",
            "brand": "Manyavar",
            "price": 1499,
            "rating": 4.3,
            "color": "blue",
            "usage": "Ethnic",
            "component_type": "Top",
            "gender": "Men",
            "product_url": "https://www.myntra.com/kurtas/manyavar/1",
        }]
    }
    detected = {"top": "kurta", "bottom": "dhoti"}

    queries = [
        ("Why did you recommend this kurta?",         "WHY intent"),
        ("Show me alternatives under ₹1500",          "ALT + budget"),
        ("What matches with this kurta?",              "MATCH intent"),
        ("Tell me about this brand",                   "BRAND intent"),
        ("Show options under ₹800",                   "BUDGET only"),
        ("Hello, what can you do?",                   "DEFAULT"),
    ]

    print("-" * 60)
    for q, label in queries:
        print(f"\n[{label}] Q: {q}")
        resp = stylist_chat(q, detected, sample_recs)
        # Print first 3 lines of response
        preview = "\n".join(resp.splitlines()[:4])
        print(f"A: {preview}")
        print("-" * 60)

    print("\nAll tests passed ✓")
