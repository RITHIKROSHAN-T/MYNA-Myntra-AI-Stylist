# streamlit_app.py
import streamlit as st
from PIL import Image
import sys
import time
from pathlib import Path
import io
import requests as req

sys.path.append(str(Path(__file__).resolve().parents[1]))
from recommender.recommender import get_recommendations
from outfit_generation.outfit_generator import (
    generate_outfit_from_prompt,
    virtual_tryon,
    is_api_available
)
try:
    from database.db_manager import (
        init_db, get_or_create_user,
        save_outfit, save_recommendations,
        get_session_stats, update_cart_status
    )
    DB_AVAILABLE = True
except Exception:
    DB_AVAILABLE = False

try:
    from conversation.rag_agent import (
        stylist_chat, build_index)
    RAG_AVAILABLE = True
except Exception:
    RAG_AVAILABLE = False

# vision modules are lazy-loaded at point of use
# to avoid 25s YOLO startup penalty on every rerender
SEG_AVAILABLE = True   # confirmed importable

# ── Cached API check (ttl=30s, avoids HF ping every render)
@st.cache_data(ttl=30)
def _cached_api_check() -> bool:
    return is_api_available()

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MYNA — AI Stylist",
    page_icon="👗",
    layout="wide"
)

# ══════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
    header    { visibility: hidden; }
    .stApp    { background-color: #f8f8f8; }

    .myna-header {
        background: linear-gradient(135deg,#FF3F6C,#FF7043);
        padding: 20px 30px; border-radius: 16px;
        color: white; text-align: center; margin-bottom: 20px;
    }
    .myna-header h1 {
        margin:0; font-size:2.2rem; font-weight:800;
    }
    .myna-header p {
        margin:4px 0 0; font-size:1rem; opacity:0.92;
    }
    .chat-user {
        background:#FF3F6C; color:white;
        padding:12px 18px;
        border-radius:20px 20px 4px 20px;
        margin:8px 0 8px auto; max-width:78%;
        font-size:0.95rem;
    }
    .chat-myna {
        background:white; color:#222;
        padding:14px 18px;
        border-radius:20px 20px 20px 4px;
        margin:8px auto 8px 0; max-width:88%;
        border:1px solid #eee; font-size:0.95rem;
        box-shadow:0 2px 8px rgba(0,0,0,0.05);
    }
    .section-label {
        background:#fff0f4;
        border-left:4px solid #FF3F6C;
        padding:9px 16px; border-radius:0 8px 8px 0;
        font-weight:700; color:#FF3F6C;
        font-size:1rem; margin:18px 0 10px;
    }
    .generated-label {
        background:#f0fff4;
        border-left:4px solid #28a745;
        padding:9px 16px; border-radius:0 8px 8px 0;
        font-weight:700; color:#28a745;
        font-size:1rem; margin:18px 0 10px;
    }
    .product-card {
        background:white; border-radius:14px;
        padding:14px 12px; border:1px solid #f0f0f0;
        text-align:center;
        box-shadow:0 2px 10px rgba(0,0,0,0.06);
        height:100%;
    }
    .product-name {
        font-weight:600; font-size:0.82rem;
        color:#222; margin:8px 0 4px; min-height:36px;
    }
    .product-brand {
        color:#888; font-size:0.74rem;
        text-transform:uppercase;
    }
    .product-price {
        color:#FF3F6C; font-weight:800;
        font-size:1.05rem; margin:6px 0 2px;
    }
    .product-rating { color:#f4a523; font-size:0.78rem; }
    .buy-btn {
        background:#FF3F6C; color:white !important;
        padding:7px 16px; border-radius:20px;
        text-decoration:none !important;
        font-size:0.78rem; font-weight:700;
        display:inline-block; margin-top:6px;
    }
    .api-on {
        background:#d4edda; color:#155724;
        padding:8px 12px; border-radius:8px;
        font-size:0.85rem; margin-bottom:10px;
    }
    .api-off {
        background:#fff3cd; color:#856404;
        padding:8px 12px; border-radius:8px;
        font-size:0.85rem; margin-bottom:10px;
    }
    .upload-info {
        background:#fff5f7; border:2px dashed #FF3F6C;
        border-radius:12px; padding:20px;
        text-align:center; color:#888;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════
defaults = {
    'chat_history'        : [],
    'photo_analyzed'      : False,
    'user_image'          : None,
    'session_id'          : str(int(time.time())),
    'last_generated_image': None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Init DB once per session
if DB_AVAILABLE and not st.session_state.get(
        'db_initialized'):
    try:
        init_db()
        get_or_create_user(st.session_state.session_id)
        st.session_state['db_initialized'] = True
    except Exception:
        pass

# Build RAG index once per session
if RAG_AVAILABLE and not st.session_state.get(
        'rag_initialized'):
    try:
        build_index()
        st.session_state['rag_initialized'] = True
    except Exception:
        pass

# Preload heavy modules in background thread
import threading
def _preload():
    try:
        from outfit_generation.outfit_generator import (
            is_api_available)
        is_api_available()
    except Exception:
        pass

if not st.session_state.get('preloaded'):
    threading.Thread(
        target=_preload, daemon=True).start()
    st.session_state['preloaded'] = True


# ══════════════════════════════════════════════════════════════
# MYNA RESPONSE
# ══════════════════════════════════════════════════════════════
def get_myna_response(user_msg: str) -> tuple:
    msg = user_msg.lower().strip()

    # Greetings
    if any(g in msg for g in
           ['hi', 'hello', 'hey', 'namaste']) \
            and len(msg.split()) <= 4:
        return (
            "Hey! 👋 I'm **MYNA**, your AI stylist! ✨\n\n"
            "Tell me what you're looking for:\n"
            "- *'Traditional kurta for Pongal under ₹2000'*\n"
            "- *'Party dress for girl under ₹5000'*\n"
            "- *'Casual jeans for college under ₹3000'*",
            {}
        )

    # Thanks
    if any(w in msg for w in
           ['thank', 'thanks', 'great', 'perfect']):
        return (
            "So glad I could help! 😊 Want me to find "
            "accessories or footwear to complete "
            "the look? 👠👜",
            {}
        )

    # Style change iteration (re-run generation on last image)
    style_changes = [
        'make it', 'change to', 'change color',
        'make the', 'add floral', 'sleeveless',
        'change color to', 'make it red',
        'add prints', 'darker', 'lighter'
    ]
    if (any(k in msg for k in style_changes) and
            st.session_state.get('user_image') and
            st.session_state.get('last_generated_image')):
        result = generate_outfit_from_prompt(
            user_image  = st.session_state.user_image,
            user_prompt = user_msg,
            session_id  = st.session_state.session_id
        )
        if result['success']:
            st.session_state.last_generated_image = \
                result['image']
            return (
                "Here's your updated look! 🎨✨\n\n"
                "Keep tweaking — tell me what to change next!",
                {'_generated': result['image']}
            )

    # RAG-only queries (why/explain/alternatives/match/brand)
    rag_keywords = [
        'why', 'explain', 'reason', 'alternative',
        'budget', 'cheaper', 'match', 'pair',
        'brand', 'tell me about'
    ]
    if any(k in msg for k in rag_keywords):
        if RAG_AVAILABLE:
            try:
                rag_response = stylist_chat(
                    user_message=user_msg,
                    detected_components={},
                    last_recommendations={}
                )
                return rag_response, {}
            except Exception:
                pass

    # Get recommendations
    intent, results = get_recommendations(user_msg, top_n=5)

    if not results:
        return (
            "I couldn't find exact matches. 🤔 Try:\n"
            "- Mention clothing type (kurta, jeans, dress)\n"
            "- Add budget (under ₹2000)\n"
            "- Mention occasion (party, festival, office)",
            {}
        )

    gender_str = intent['gender'] or 'your'
    price_str  = (f"under ₹{intent['max_price']:,}"
                  if intent['max_price'] else "")
    if intent.get('min_price') and intent.get('max_price'):
        price_str = (f"₹{intent['min_price']:,}–"
                     f"₹{intent['max_price']:,}")
    occ_str  = (f"for {intent['occasion']}"
                if intent['occasion'] else "")
    comp_str = " & ".join(results.keys())
    col_str  = (", ".join(intent['colors']) + " "
                if intent['colors'] else "")

    response = (
        f"Fabulous taste! 🌟 Here are my top picks for "
        f"**{gender_str} {col_str}{comp_str}** "
        f"{occ_str} {price_str} — "
        f"curated just for you! 👇"
    )

    rag_reply = ""
    if RAG_AVAILABLE and results:
        try:
            rag_reply = stylist_chat(
                user_message=user_msg,
                detected_components={},
                last_recommendations=results
            )
        except Exception:
            rag_reply = ""
    if rag_reply:
        response = response + "\n\n💡 " + rag_reply

    # Save to DB
    if DB_AVAILABLE:
        try:
            save_recommendations(
                st.session_state.session_id,
                "no-outfit",
                results
            )
        except Exception:
            pass

    return response, results


# ══════════════════════════════════════════════════════════════
# RENDER PRODUCT CARDS
# ══════════════════════════════════════════════════════════════
def render_products(results: dict, msg_idx: int = 0):
    icons = {
        'Top'        : '👕',
        'Bottom'     : '👖',
        'Footwear'   : '👟',
        'Accessories': '👜'
    }
    api_ok = _cached_api_check()

    for comp_type, products in results.items():
        icon = icons.get(comp_type, '🛍️')
        st.markdown(
            f'<div class="section-label">'
            f'{icon} {comp_type} Recommendations'
            f'</div>',
            unsafe_allow_html=True
        )
        n    = min(len(products), 5)
        cols = st.columns(n)

        for i, p in enumerate(products[:5]):
            # CTR: track product views
            st.session_state.setdefault(
                'viewed_products', [])
            if p.get('product_id'):
                st.session_state.viewed_products.append(
                    p.get('product_id'))

            with cols[i]:
                img_url = p.get('image_url', '')
                if img_url:
                    try:
                        st.image(img_url,
                                 use_container_width=True)
                    except Exception:
                        st.markdown("🖼️")

                name   = str(p.get('name',  ''))[:55].title()
                brand  = str(p.get('brand', ''))
                price  = int(p.get('price', 0))
                rating = p.get('rating', 0)
                purl   = p.get('product_url', '#')
                rev    = int(p.get('rating_count', 0))
                usage  = str(p.get('usage',  'Casual'))
                gender = str(p.get('gender', 'Unisex'))

                try:
                    stars = '⭐' * int(float(rating))
                except Exception:
                    stars = ''

                st.markdown(f"""
<div class="product-card">
    <div class="product-name">{name}</div>
    <div class="product-brand">{brand}</div>
    <div class="product-price">₹{price:,}</div>
    <div class="product-rating">
        {stars} {rating}
        <br><small>({rev:,} reviews)</small>
    </div>
    <a href="{purl}" target="_blank" class="buy-btn">
        Buy on Myntra →
    </a>
</div>""", unsafe_allow_html=True)

                # ── Try On / Style Preview Button ──────────
                # ── Try On (Top only) ──────────────────────
                if (comp_type == 'Top' and
                        api_ok and
                        st.session_state.user_image and
                        img_url):
                    btn_key = (
                        f"tryon_{msg_idx}_{comp_type}_{i}_"
                        f"{name[:8].replace(' ','_')}"
                    )
                    if st.button(
                        "👔 Try On",
                        key=btn_key,
                        use_container_width=True
                    ):
                        with st.spinner(
                            f"Trying on "
                            f"{name[:20]}... ✨"
                            f" (60-90 sec)"
                        ):
                            try:
                                gr = req.get(
                                    img_url,
                                    timeout=30,
                                    headers={
                                        'User-Agent':
                                        'Mozilla/5.0 '
                                        'Chrome/120',
                                        'Referer':
                                        'https://www'
                                        '.myntra.com/',
                                    }
                                )
                                if gr.status_code == 200:
                                    gimg = Image.open(
                                        io.BytesIO(
                                            gr.content)
                                    ).convert('RGB')
                                    result = virtual_tryon(
                                        person_image=st.session_state.user_image,
                                        garment_image=gimg,
                                        garment_name=name,
                                        session_id=st.session_state.session_id
                                    )
                                else:
                                    result = generate_outfit_from_prompt(
                                        user_image=st.session_state.user_image,
                                        user_prompt=f"{name} Indian fashion top",
                                        session_id=st.session_state.session_id
                                    )
                            except Exception:
                                result = generate_outfit_from_prompt(
                                    user_image=st.session_state.user_image,
                                    user_prompt=f"{name} Indian fashion top",
                                    session_id=st.session_state.session_id
                                )

                        if result['success']:
                            if DB_AVAILABLE:
                                try:
                                    save_recommendations(
                                        st.session_state.session_id,
                                        "tryon-interest",
                                        {comp_type: [p]}
                                    )
                                except Exception:
                                    pass
                            st.session_state.chat_history.append({
                                'role'    : 'myna',
                                'content' : (
                                    f"Here's how you'd "
                                    f"look in **{name}**!"
                                    f" 🎨✨\n\nLike it? "
                                    f"Click 'Buy on "
                                    f"Myntra'!"
                                ),
                                'results' : {},
                                'generated': result['image']
                            })
                            try:
                                from vision.segmentation import (
                                    segment_clothing)

                                seg = segment_clothing(result['image'])
                                regions = seg.get('regions', {})

                                ORDER = ['Topwear', 'Bottomwear']
                                tried_comp = str(comp_type).lower()

                                def get_emoji(cat):
                                    c = str(cat).lower()
                                    if 'top' in c or 'upper' in c:
                                        return '👕'
                                    elif 'bottom' in c or 'lower' in c:
                                        return '👖'
                                    elif 'foot' in c or 'shoe' in c:
                                        return '👟'
                                    elif 'acc' in c:
                                        return '👜'
                                    return '👗'

                                def get_garment_label(cat, data,
                                                       prod_name=None):
                                    if prod_name:
                                        return prod_name
                                    raw = str(data.get(
                                        'raw_label', '')).lower()
                                    label_map = {
                                        'pants'        : 'Jeans / Pants',
                                        'jean'         : 'Jeans',
                                        'trouser'      : 'Trousers',
                                        'skirt'        : 'Skirt',
                                        'shorts'       : 'Shorts',
                                        'dress'        : 'Dress',
                                        'upper-clothes': 'Top',
                                        'upper_clothes': 'Top',
                                        'left-shoe'    : 'Shoes',
                                        'right-shoe'   : 'Shoes',
                                        'left_shoe'    : 'Shoes',
                                        'right_shoe'   : 'Shoes',
                                        'boot'         : 'Boots',
                                        'sandal'       : 'Sandals',
                                        'sneaker'      : 'Sneakers',
                                        'formal'       : 'Formal Shoes',
                                        'hat'          : 'Hat',
                                        'belt'         : 'Belt',
                                        'bag'          : 'Bag',
                                    }
                                    for key, val in label_map.items():
                                        if key in raw:
                                            return val
                                    return cat

                                detection_lines = []

                                for cat in ORDER:
                                    cat_lower = cat.lower()

                                    is_tried = (
                                        ('top' in tried_comp and
                                         'top' in cat_lower) or
                                        ('bottom' in tried_comp and
                                         'bottom' in cat_lower) or
                                        ('foot' in tried_comp and
                                         'foot' in cat_lower)
                                    )

                                    if is_tried:
                                        label = get_garment_label(
                                            cat, {}, prod_name=name)
                                        detection_lines.append(
                                            f"{get_emoji(cat)} {cat}"
                                            f" — {label}"
                                        )
                                    elif cat in regions:
                                        data = regions[cat]
                                        mask = data.get('mask')
                                        if mask is not None and mask.any():
                                            label = get_garment_label(
                                                cat, data)
                                            detection_lines.append(
                                                f"{get_emoji(cat)} {cat}"
                                                f" — {label}"
                                            )

                                if detection_lines:
                                    detection_text = (
                                        "🔍 I detected these components"
                                        " in your outfit:\n" +
                                        "\n".join(
                                            f"• {l}"
                                            for l in detection_lines)
                                    )
                                else:
                                    detection_text = (
                                        f"🔍 Outfit component detected:\n"
                                        f"{get_emoji(comp_type)} "
                                        f"{comp_type} — {name}"
                                    )

                                st.session_state.chat_history.append({
                                    'role'    : 'myna',
                                    'content' : detection_text,
                                    'results' : {},
                                    'generated': None
                                })
                            except Exception as e:
                                print(f"[DETECTION] Skipped: {e}")
                        else:
                            err = str(result.get(
                                'error', 'Unknown error'))
                            if '⏳' in err or 'busy' in err.lower():
                                st.session_state.chat_history.append({
                                    'role'    : 'myna',
                                    'content' : err,
                                    'results' : {},
                                    'generated': None
                                })
                                st.rerun()
                            elif 'quota' in err.lower():
                                st.session_state.chat_history.append({
                                    'role'    : 'myna',
                                    'content' : (
                                        "⏳ GPU is busy right now.\n\n"
                                        "Please wait 1-2 minutes and "
                                        "click Try On again."
                                    ),
                                    'results' : {},
                                    'generated': None
                                })
                                st.rerun()
                            else:
                                st.error(
                                    f"Try-on failed: {err[:100]}")
                        st.rerun()

                # ── Try On Full Body (Bottom/Footwear/Acc) ──
                if (comp_type != 'Top' and
                        api_ok and
                        st.session_state.user_image and
                        img_url):
                    view_key = (
                        f"view_{msg_idx}_{comp_type}_{i}_"
                        f"{name[:8].replace(' ','_')}"
                    )
                    if st.button(
                        "👗 Try On (Full Body)",
                        key=view_key,
                        use_container_width=True
                    ):
                        if comp_type == 'Footwear':
                            st.session_state.chat_history.append({
                                'role'    : 'myna',
                                'content' : (
                                    f"👟 Footwear try-on is not available "
                                    f"— here's how **{name}** looks on model.\n\n"
                                    f"Click 'View on Myntra' to see all angles "
                                    f"and size guide.\n\n"
                                    f'<a href="{purl}" target="_blank" '
                                    f'style="background:#FF3F6C;color:white;'
                                    f'padding:8px 16px;border-radius:6px;'
                                    f'text-decoration:none;font-weight:700;">'
                                    f'🔗 View on Myntra</a>'
                                ),
                                'results' : {},
                                'generated': img_url,
                            })
                            st.rerun()
                        else:
                            with st.spinner(
                                f"Trying on {name[:20]}"
                                f"... ✨ (60-90 sec)"
                            ):
                                try:
                                    gr = req.get(
                                        img_url,
                                        timeout=30,
                                        headers={
                                            'User-Agent':
                                            'Mozilla/5.0 '
                                            'Chrome/120',
                                            'Referer':
                                            'https://www'
                                            '.myntra.com/',
                                        }
                                    )
                                    if gr.status_code == 200:
                                        gimg = Image.open(
                                            io.BytesIO(
                                                gr.content)
                                        ).convert('RGB')
                                        gtype_map = {
                                            'Bottom'     : 'lowerbody',
                                            'Footwear'   : 'lowerbody',
                                            'Accessories': 'upperbody',
                                        }
                                        gtype = gtype_map.get(
                                            comp_type, 'upperbody')
                                        from outfit_generation.outfit_generator import (
                                            virtual_tryon_full_body)
                                        result = virtual_tryon_full_body(
                                            person_image=st.session_state.user_image,
                                            garment_image=gimg,
                                            garment_name=name,
                                            garment_type=gtype,
                                            session_id=st.session_state.session_id
                                        )
                                    else:
                                        result = generate_outfit_from_prompt(
                                            user_image=st.session_state.user_image,
                                            user_prompt=f"{name} {comp_type} Indian fashion",
                                            session_id=st.session_state.session_id
                                        )
                                except Exception as e:
                                    result = {
                                        'success': False,
                                        'image'  : None,
                                        'error'  : str(e)
                                    }

                            if result['success']:
                                st.session_state.chat_history.append({
                                    'role'    : 'myna',
                                    'content' : (
                                        f"Here's how you'd "
                                        f"look in **{name}**!"
                                        f" 🎨✨\n\nLike it? "
                                        f"Click 'Buy on "
                                        f"Myntra'!"
                                    ),
                                    'results' : {},
                                    'generated': result['image']
                                })
                                try:
                                    from vision.segmentation import (
                                        segment_clothing)

                                    seg = segment_clothing(result['image'])
                                    regions = seg.get('regions', {})

                                    ORDER = ['Topwear', 'Bottomwear']
                                    tried_comp = str(comp_type).lower()

                                    def get_emoji(cat):
                                        c = str(cat).lower()
                                        if 'top' in c or 'upper' in c:
                                            return '👕'
                                        elif 'bottom' in c or 'lower' in c:
                                            return '👖'
                                        elif 'foot' in c or 'shoe' in c:
                                            return '👟'
                                        elif 'acc' in c:
                                            return '👜'
                                        return '👗'

                                    def get_garment_label(cat, data,
                                                           prod_name=None):
                                        if prod_name:
                                            return prod_name
                                        raw = str(data.get(
                                            'raw_label', '')).lower()
                                        label_map = {
                                            'pants'        : 'Jeans / Pants',
                                            'jean'         : 'Jeans',
                                            'trouser'      : 'Trousers',
                                            'skirt'        : 'Skirt',
                                            'shorts'       : 'Shorts',
                                            'dress'        : 'Dress',
                                            'upper-clothes': 'Top',
                                            'upper_clothes': 'Top',
                                            'left-shoe'    : 'Shoes',
                                            'right-shoe'   : 'Shoes',
                                            'left_shoe'    : 'Shoes',
                                            'right_shoe'   : 'Shoes',
                                            'boot'         : 'Boots',
                                            'sandal'       : 'Sandals',
                                            'sneaker'      : 'Sneakers',
                                            'formal'       : 'Formal Shoes',
                                            'hat'          : 'Hat',
                                            'belt'         : 'Belt',
                                            'bag'          : 'Bag',
                                        }
                                        for key, val in label_map.items():
                                            if key in raw:
                                                return val
                                        return cat

                                    detection_lines = []

                                    for cat in ORDER:
                                        cat_lower = cat.lower()

                                        is_tried = (
                                            ('top' in tried_comp and
                                             'top' in cat_lower) or
                                            ('bottom' in tried_comp and
                                             'bottom' in cat_lower) or
                                            ('foot' in tried_comp and
                                             'foot' in cat_lower)
                                        )

                                        if is_tried:
                                            label = get_garment_label(
                                                cat, {}, prod_name=name)
                                            detection_lines.append(
                                                f"{get_emoji(cat)} {cat}"
                                                f" — {label}"
                                            )
                                        elif cat in regions:
                                            data = regions[cat]
                                            mask = data.get('mask')
                                            if mask is not None and mask.any():
                                                label = get_garment_label(
                                                    cat, data)
                                                detection_lines.append(
                                                    f"{get_emoji(cat)} {cat}"
                                                    f" — {label}"
                                                )

                                    if detection_lines:
                                        detection_text = (
                                            "🔍 I detected these components"
                                            " in your outfit:\n" +
                                            "\n".join(
                                                f"• {l}"
                                                for l in detection_lines)
                                        )
                                    else:
                                        detection_text = (
                                            f"🔍 Outfit component detected:\n"
                                            f"{get_emoji(comp_type)} "
                                            f"{comp_type} — {name}"
                                        )

                                    st.session_state.chat_history.append({
                                        'role'    : 'myna',
                                        'content' : detection_text,
                                        'results' : {},
                                        'generated': None
                                    })
                                except Exception as e:
                                    print(f"[DETECTION] Skipped: {e}")
                            else:
                                err = str(result.get(
                                    'error', 'Unknown error'))
                                if '⏳' in err or 'busy' in err.lower():
                                    st.session_state.chat_history.append({
                                        'role'    : 'myna',
                                        'content' : err,
                                        'results' : {},
                                        'generated': None
                                    })
                                    st.rerun()
                                elif 'quota' in err.lower():
                                    st.session_state.chat_history.append({
                                        'role'    : 'myna',
                                        'content' : (
                                            "⏳ GPU is busy right now.\n\n"
                                            "Please wait 1-2 minutes and "
                                            "click Try On again."
                                        ),
                                        'results' : {},
                                        'generated': None
                                    })
                                    st.rerun()
                                else:
                                    st.error(
                                        f"Try-on failed: {err[:100]}")
                            st.rerun()

                # ── Add to Cart Button ─────────────────────
                cart_key = (
                    f"cart_{msg_idx}_{comp_type}_{i}_"
                    f"{name[:8].replace(' ','_')}"
                )
                if st.button(
                    "🛒 Add to Cart",
                    key=cart_key,
                    use_container_width=True
                ):
                    try:
                        _raw_pid = p.get('product_id', '')
                        prod_id  = str(_raw_pid).strip() \
                            if (_raw_pid and
                                str(_raw_pid) not in ('', 'nan', 'None')) \
                            else str(p.get('name', name))[:30].strip()
                        if not prod_id:
                            prod_id = f"{name[:20]}_{i}"
                        print(f"[Cart] product={name[:30]} "
                              f"price={p.get('price')} "
                              f"brand={p.get('brand')} "
                              f"prod_id={prod_id}")

                        if DB_AVAILABLE:
                            import uuid
                            from datetime import datetime
                            from database.db_manager import _get_conn

                            # Ensure product exists in catalog
                            try:
                                with _get_conn() as conn:
                                    with conn.cursor() as cur:
                                        cur.execute("""
                                            INSERT INTO product_catalog
                                            (product_id, name, category,
                                             price, image_url, product_url,
                                             brand, color, created_at)
                                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                                            ON CONFLICT (product_id) DO UPDATE SET
                                                name        = EXCLUDED.name,
                                                price       = EXCLUDED.price,
                                                brand       = EXCLUDED.brand,
                                                image_url   = EXCLUDED.image_url,
                                                product_url = EXCLUDED.product_url,
                                                color       = EXCLUDED.color
                                        """, (
                                            prod_id,
                                            str(p.get('name', ''))[:200],
                                            comp_type,
                                            float(p.get('price', 0) or 0),
                                            str(p.get('image_url', ''))[:500],
                                            str(p.get('product_url', ''))[:500],
                                            str(p.get('brand', 'Unknown'))[:100],
                                            str(p.get('color', ''))[:50],
                                            datetime.now().isoformat(),
                                        ))
                            except Exception as db_e:
                                print(f"[Cart] Catalog insert: {db_e}")

                            # Ensure 'direct-cart' outfit stub exists
                            # (satisfies outfit_id FK constraint)
                            try:
                                with _get_conn() as conn:
                                    with conn.cursor() as cur:
                                        cur.execute("""
                                            INSERT INTO outfits
                                            (outfit_id, user_id,
                                             photo_url, style_type,
                                             top_label, bottom_label,
                                             accessory_labels, created_at)
                                            VALUES (%s,%s,'','direct-cart',
                                                    '','','', NOW())
                                            ON CONFLICT (outfit_id) DO NOTHING
                                        """, (
                                            'direct-cart',
                                            st.session_state.session_id,
                                        ))
                            except Exception:
                                pass

                            # Add to cart
                            try:
                                with _get_conn() as conn:
                                    with conn.cursor() as cur:
                                        cur.execute("""
                                            INSERT INTO cart_recommendations
                                            (entry_id, user_id, outfit_id,
                                             product_id, component_type,
                                             added_to_cart, recommended_at,
                                             purchased)
                                            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                                            ON CONFLICT DO NOTHING
                                        """, (
                                            str(uuid.uuid4()),
                                            st.session_state.session_id,
                                            'direct-cart',
                                            prod_id,
                                            comp_type,
                                            True,
                                            datetime.now(),
                                            False,
                                        ))
                                        cur.execute("""
                                            UPDATE cart_recommendations
                                            SET added_to_cart = TRUE
                                            WHERE user_id = %s
                                            AND product_id = %s
                                        """, (
                                            st.session_state.session_id,
                                            prod_id
                                        ))
                            except Exception as cart_e:
                                print(f"[Cart] Insert error: {cart_e}")

                        st.success(f"✅ Added {name[:25]} to cart!")
                        st.session_state['cart_items_cache'] = None
                        crosssell = {
                            'Top'        : 'bottom wear or footwear',
                            'Bottom'     : 'a matching top or belt',
                            'Footwear'   : 'a matching bag or watch',
                            'Accessories': 'a complete outfit',
                        }
                        st.info(
                            f"💡 Complete your look with "
                            f"{crosssell.get(comp_type,'accessories')}!"
                        )
                        st.rerun()

                    except Exception as e:
                        print(f"[Cart] Unexpected: {e}")
                        st.error(f"Cart error: {str(e)[:80]}")


# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="myna-header">
    <h1>👗 MYNA for Myntra</h1>
    <p>See It. Style It. Shop It. — Your AI Personal Stylist</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["👗 MYNA Stylist", "📊 Analytics"])

with tab1:
 col_left, col_right = st.columns([1, 2], gap="large")


# ══════════════════════════════════════════════════════════════
# LEFT COLUMN — Photo Upload + Generate
# ══════════════════════════════════════════════════════════════
with col_left:
    st.markdown("### 📸 Your Photo")

    uploaded = st.file_uploader(
        "Upload photo",
        type=['jpg', 'jpeg', 'png', 'webp'],
        label_visibility='collapsed'
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.session_state.user_image = image
        st.image(image,
                 use_container_width=True,
                 caption="Your uploaded photo")

        if not st.session_state.photo_analyzed:
            with st.spinner("MYNA is ready... ✨"):
                time.sleep(1)
            st.session_state.photo_analyzed = True

            # Run YOLOv8 detection + save to DB
            if DB_AVAILABLE:
                try:
                    from vision.component_detector import (
                        analyze_outfit)
                    detection = analyze_outfit(image)
                    outfit_id = save_outfit(
                        st.session_state.session_id,
                        detection,
                        style_type="Uploaded Photo"
                    )
                    st.session_state['last_outfit_id'] = \
                        outfit_id
                except Exception:
                    pass

            st.session_state.chat_history.append({
                'role'     : 'myna',
                'content'  : (
                    "Love the photo! 📸 You look amazing! 😍\n\n"
                    "I'm **MYNA**, your AI stylist!\n\n"
                    "What kind of look are you going for today?\n"
                    "1. 🎉 Casual everyday wear\n"
                    "2. 👔 Formal/Office wear\n"
                    "3. 🎊 Party/Celebration look\n"
                    "4. 🙏 Festival/Ethnic wear\n"
                    "5. 💪 Sports/Gym wear\n\n"
                    "Or just tell me what you want! 🛍️"
                ),
                'results'  : {},
                'generated': None
            })
            st.rerun()

        st.success("✅ Photo ready!")

        if SEG_AVAILABLE:
            with st.expander(
                "🔍 View Clothing Segmentation",
                expanded=False
            ):
                with st.spinner(
                    "Detecting clothing regions..."
                ):
                    from vision.segmentation import (
                        segment_clothing)
                    seg_result = segment_clothing(
                        image)
                st.image(
                    seg_result['annotated_image'],
                    caption=(
                        "Detected regions: "
                        + ", ".join(
                            seg_result['labels'])
                    ),
                    use_container_width=True
                )

    else:
        st.markdown("""
        <div class="upload-info">
            📷 Upload your photo to get started!<br>
            <small>JPG, PNG, WEBP supported</small>
        </div>""", unsafe_allow_html=True)
        st.info(
            "💡 Use a clear full-body photo "
            "for best outfit generation results!")

    # ── API Status ─────────────────────────────────────────
    st.markdown("---")
    api_ok = _cached_api_check()
    if api_ok:
        st.markdown(
            '<div class="api-on">'
            '🟢 AI Generation: Online</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="api-off">'
            '🟡 AI Generation: Offline '
            '(HF Space loading...)</div>',
            unsafe_allow_html=True)

    # ── Generate Outfit Prompt Box ─────────────────────────
    st.markdown("### 🎨 Generate Outfit on Your Photo")
    st.markdown("*Describe exactly what outfit you want:*")

    gen_prompt = st.text_area(
        "Prompt:",
        placeholder=(
            "e.g. change outfit to blue silk kurta "
            "with white dhoti for pongal festival"
        ),
        height=90,
        label_visibility='collapsed'
    )

    if st.button(
        "✨ Generate Outfit",
        use_container_width=True,
        type="primary",
        disabled=not api_ok
    ):
        if not gen_prompt.strip():
            st.warning("Please enter a style prompt!")
        elif not st.session_state.user_image:
            st.warning("Please upload your photo first!")
        else:
            st.session_state.chat_history.append({
                'role'     : 'user',
                'content'  : f"🎨 {gen_prompt}",
                'results'  : {},
                'generated': None
            })

            with st.spinner(
                "MYNA is generating your outfit... "
                "✨ (30-60 seconds)"
            ):
                result = generate_outfit_from_prompt(
                    user_image  = st.session_state.user_image,
                    user_prompt = gen_prompt,
                    session_id  = st.session_state.session_id
                )

            if result['success']:
                # Save generated outfit to DB
                if DB_AVAILABLE:
                    try:
                        from vision.component_detector import (
                            analyze_outfit)
                        detection = analyze_outfit(
                            result['image'])
                        save_outfit(
                            st.session_state.session_id,
                            detection,
                            style_type=gen_prompt[:100]
                        )
                    except Exception:
                        pass

                st.session_state.last_generated_image = \
                    result['image']
                st.session_state.chat_history.append({
                    'role'     : 'myna',
                    'content'  : (
                        "Here's your AI-generated outfit! "
                        "🎨✨\n\n"
                        "Love it? Ask me to find similar "
                        "real products from Myntra! 🛍️\n\n"
                        "💡 *Tip: Say 'make it darker' or "
                        "'change to red' to tweak the look!*"
                    ),
                    'results'  : {},
                    'generated': result['image']
                })
            else:
                st.session_state.chat_history.append({
                    'role'     : 'myna',
                    'content'  : (
                        f"Generation failed: "
                        f"{result['error']} 😔\n\n"
                        "HF Space may be sleeping — try again in 30s."
                    ),
                    'results'  : {},
                    'generated': None
                })
            st.rerun()

    # ── Quick Style Buttons ────────────────────────────────
    if st.session_state.photo_analyzed:
        st.markdown("---")
        st.markdown("**What's your vibe today?**")
        style_options = [
            ("🎉 Casual",  "Casual everyday wear under ₹2000"),
            ("👔 Formal",  "Formal office wear for men under ₹3000"),
            ("🎊 Party",   "Party outfit for girl under ₹5000"),
            ("🙏 Festival","Traditional ethnic festival wear under ₹2500"),
            ("💪 Sports",  "Sports gym wear under ₹1500"),
        ]
        for label, query in style_options:
            if st.button(
                label,
                use_container_width=True,
                key=f"style_{label[:6]}"
            ):
                st.session_state.chat_history.append({
                    'role'   : 'user',
                    'content': query,
                    'results': {}, 'generated': None
                })
                reply, results = get_myna_response(query)
                _gen = results.pop('_generated', None)
                st.session_state.chat_history.append({
                    'role'     : 'myna',
                    'content'  : reply,
                    'results'  : results,
                    'generated': _gen,
                })
                st.rerun()

    # ── DB Session Stats ───────────────────────────────────
    if DB_AVAILABLE:
        try:
            stats = get_session_stats(
                st.session_state.session_id)
            st.markdown("---")
            st.markdown("**📊 Your Session:**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Outfits",
                      stats['outfits_generated'])
            c2.metric("Picks",
                      stats['products_recommended'])
            c3.metric("Cart",
                      stats['items_in_cart'])
        except Exception:
            pass

    # ── Cart Display ───────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🛒 Your Cart")
    if st.button(
        "🔄 Refresh Cart",
        use_container_width=True,
        key="refresh_cart_btn"
    ):
        st.session_state['cart_items_cache'] = None
        st.rerun()
    try:
        from database.db_manager import _get_conn
        cart_items = st.session_state.get('cart_items_cache', None)
        if cart_items is None:
            with _get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT
                            cr.product_id,
                            COALESCE(pc.name,
                                cr.product_id) AS name,
                            COALESCE(pc.price, 0) AS price,
                            COALESCE(pc.brand,
                                'Unknown') AS brand,
                            COALESCE(pc.product_url,
                                '#') AS product_url
                        FROM cart_recommendations cr
                        LEFT JOIN product_catalog pc
                            ON cr.product_id = pc.product_id
                        WHERE cr.user_id = %s
                          AND cr.added_to_cart = TRUE
                        ORDER BY cr.recommended_at DESC
                    """, (st.session_state.session_id,))
                    cart_items = cur.fetchall()
            st.session_state['cart_items_cache'] = cart_items

        if cart_items:
            total = sum(
                float(row[2] or 0)
                for row in cart_items
            )
            ca, cb = st.columns(2)
            ca.metric("🛒 Items", len(cart_items))
            cb.metric("💰 Total", f"₹{total:,.0f}")
            st.markdown("---")

            for row in cart_items:
                pid    = str(row[0] or '')
                iname  = str(row[1] or '')[:35].title()
                iprice = float(row[2] or 0)
                ibrand = str(row[3] or '')
                ipurl  = str(row[4] or '#')

                col1, col2 = st.columns([4, 1])
                col1.markdown(
                    f"**{iname}**  \n"
                    f"{ibrand} — ₹{iprice:,.0f}"
                )
                with col2:
                    rm_key = (
                        f"rm_{pid[:12]}"
                        f"_{hash(pid) % 9999}"
                    )
                    if st.button("❌", key=rm_key):
                        try:
                            with _get_conn() as conn:
                                with conn.cursor() as cur:
                                    cur.execute("""
                                        UPDATE cart_recommendations
                                        SET added_to_cart = FALSE
                                        WHERE user_id = %s
                                        AND product_id = %s
                                    """, (
                                        st.session_state.session_id,
                                        pid
                                    ))
                            st.session_state['cart_items_cache'] = None
                            st.rerun()
                        except Exception:
                            pass

            st.markdown("---")
            if st.button(
                "🛍️ Checkout on Myntra",
                use_container_width=True,
                type="primary"
            ):
                st.success(
                    "Visit myntra.com to complete "
                    "your purchase! 🛍️"
                )
        else:
            st.info(
                "🛒 Cart is empty\n"
                "Click 'Add to Cart' on any product!"
            )

    except Exception as e:
        print(f"[Cart display] {e}")
        st.caption("Cart loading...")

    # ── Start Fresh ────────────────────────────────────────
    st.markdown("---")
    if st.button("🔄 Start Fresh",
                 use_container_width=True):
        for k in defaults:
            st.session_state[k] = defaults[k]
        st.session_state.session_id = str(int(time.time()))
        st.rerun()


# ══════════════════════════════════════════════════════════════
# RIGHT COLUMN — Chat Interface
# ══════════════════════════════════════════════════════════════
with col_right:
    st.markdown("### 💬 Chat with MYNA")

    with st.container():
        if not st.session_state.chat_history:
            st.markdown("""
            <div class="chat-myna">
                👋 Hi! I'm <b>MYNA</b>!<br><br>
                📸 Upload your photo on the left,
                then:<br><br>
                1. 💬 Ask for outfit recommendations<br>
                2. 🎨 Generate outfit with AI prompt<br>
                3. 👔 Click Try On on any product!
            </div>""", unsafe_allow_html=True)

        for msg_idx, msg in enumerate(st.session_state.chat_history):
            if msg['role'] == 'user':
                st.markdown(
                    f'<div class="chat-user">'
                    f'👤 {msg["content"]}</div>',
                    unsafe_allow_html=True)
            else:
                content = msg['content'].replace(
                    '\n', '<br>')
                st.markdown(
                    f'<div class="chat-myna">'
                    f'👗 <b>MYNA:</b><br>{content}'
                    f'</div>',
                    unsafe_allow_html=True)

                # Show generated image
                if msg.get('generated') is not None:
                    st.markdown(
                        '<div class="generated-label">'
                        '🎨 AI Generated Outfit</div>',
                        unsafe_allow_html=True)
                    st.image(
                        msg['generated'],
                        use_container_width=True,
                        caption="Your AI-styled outfit"
                    )

                # Show product cards
                if msg.get('results'):
                    render_products(msg['results'], msg_idx)

    # ── Chat Input ─────────────────────────────────────────
    st.markdown("---")
    with st.form(key='chat_form', clear_on_submit=True):
        user_input = st.text_input(
            "Ask MYNA...",
            placeholder=(
                "e.g. Traditional kurta for "
                "Pongal under ₹2000"
            )
        )
        send = st.form_submit_button(
            "Send 💬",
            use_container_width=True
        )

    if send and user_input.strip():
        if not st.session_state.photo_analyzed:
            st.warning("👆 Upload your photo first!")
        else:
            st.session_state.chat_history.append({
                'role'   : 'user',
                'content': user_input.strip(),
                'results': {}, 'generated': None
            })
            with st.spinner(
                "MYNA is styling you... ✨"
            ):
                reply, results = get_myna_response(
                    user_input.strip())
                time.sleep(0.3)
            _gen = results.pop('_generated', None)
            st.session_state.chat_history.append({
                'role'     : 'myna',
                'content'  : reply,
                'results'  : results,
                'generated': _gen,
            })
            st.rerun()

# ══════════════════════════════════════════════════════════════
# TAB 2 — ANALYTICS
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 📊 MYNA Analytics Dashboard")

    try:
        from database.db_manager import get_all_stats
        import pandas as pd
        stats = get_all_stats()

        # ── Top metrics row ────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("👥 Users",      stats['total_users'])
        c2.metric("👗 Outfits",    stats['total_outfits'])
        c3.metric("🛍️ Picks",     stats['total_recommended'])
        c4.metric("🛒 Cart",      stats['total_in_cart'])

        # ── Secondary metrics ──────────────────────────────
        st.markdown("---")
        d1, d2, d3 = st.columns(3)
        d1.metric("🏷️ Purchased",  stats.get('total_purchased', 0))
        d2.metric("👁️ Viewed",     stats.get('total_recommended', 0))
        d3.metric("📈 Cart Rate",   f"{stats['cart_rate']}%")

        # ── Conversion progress bar ────────────────────────
        st.markdown("#### Cart Conversion Rate")
        rate = min(float(stats['cart_rate']), 100.0)
        st.progress(int(rate), text=f"{rate:.1f}% of picks added to cart")

        st.markdown("---")

        # ── Side-by-side bar charts ────────────────────────
        ch1, ch2 = st.columns(2)

        with ch1:
            st.markdown("### Top Categories")
            if stats['top_components']:
                df_comp = pd.DataFrame(stats['top_components'])
                st.bar_chart(
                    df_comp.set_index('type')['count'],
                    use_container_width=True
                )
            else:
                st.caption("No data yet.")

        with ch2:
            st.markdown("### Top Style Types")
            if stats['top_styles']:
                df_style = pd.DataFrame(stats['top_styles'])
                st.bar_chart(
                    df_style.set_index('style')['count'],
                    use_container_width=True
                )
            else:
                st.caption("No data yet.")

    except Exception as e:
        st.error(f"Analytics error: {e}")
        st.info(
            "Start using MYNA to see analytics! "
            "Data will appear here after interactions."
        )