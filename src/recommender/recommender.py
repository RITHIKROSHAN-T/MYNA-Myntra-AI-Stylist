# recommender.py
import pandas as pd
import numpy as np
import re
from pathlib import Path

BASE_DIR     = Path(__file__).resolve().parents[2]
CATALOG_PATH = BASE_DIR / "data" / "processed" / "clean_myntra_products.csv"

import functools

@functools.lru_cache(maxsize=1)
def _load_catalog() -> pd.DataFrame:
    print("Loading product catalog...")
    df = pd.read_csv(CATALOG_PATH)
    df['name']           = df['name'].fillna('').str.lower()
    df['brand']          = df['brand'].fillna('Unknown')
    df['color']          = df['color'].fillna('Unknown')
    df['gender']         = df['gender'].fillna('Unisex')
    df['component_type'] = df['component_type'].fillna('Other')
    df['usage']          = df['usage'].fillna('Casual')
    df['price']          = pd.to_numeric(
                               df['price'],
                               errors='coerce').fillna(0)
    df['rating']         = pd.to_numeric(
                               df['rating'],
                               errors='coerce').fillna(0)
    df['rating_count']   = pd.to_numeric(
                               df['rating_count'],
                               errors='coerce').fillna(0)
    print(f"Catalog ready: {len(df):,} products")
    return df

CATALOG = _load_catalog()


# ── Women-only brands ──────────────────────────────────────────
WOMENS_BRANDS = {
    'van heusen woman', 'annabelle by pantaloons', 'femella',
    'sassafras', 'wishful by w', 'w for woman', 'global desi',
    'aurelia', 'rangmanch', 'wishful', 'oxolloxo', 'berrylush',
    '20dresses', 'athena', 'pspeaches', 'lemon & pepper',
    'biba', 'libas', 'nayo', 'soch', 'indo era', 'rain & rainbow',
}

# ── Men-only brands ────────────────────────────────────────────
MENS_ONLY_BRANDS = {
    'arrow', 'peter england', 'louis philippe', 'van heusen',
    'raymond', 'park avenue', 'blackberrys', 'john players',
    'invictus', 'mufti', 'locomotive', 'dennis lingo',
    'zodiac', 'svanik', 'solemio',
}

# ── Occasion → usage mapping ───────────────────────────────────
OCCASION_USAGE = {
    'festival' : 'Ethnic',
    'party'    : 'Party',
    'college'  : 'Casual',
    'wedding'  : 'Ethnic',
    'office'   : 'Formal',
    'sports'   : 'Sports',
}


# ══════════════════════════════════════════════════════════════
# PART 1 — PARSE USER MESSAGE
# ══════════════════════════════════════════════════════════════
def parse_message(message: str) -> dict:

    msg = str(message).lower().strip()

    intent = {
        'gender'          : None,
        'component_types' : [],
        'max_price'       : None,
        'min_price'       : None,
        'colors'          : [],
        'occasion'        : None,
        'keywords'        : [],
        'specific_item'   : None,
    }

    # ── Gender ─────────────────────────────────────────────────
    if any(w in msg for w in ['women', 'woman', 'girl', 'female',
                               'lady', 'ladies', 'her', 'she']):
        intent['gender'] = 'Women'
    elif any(w in msg for w in ['men', 'man', 'male', 'gents',
                                 'his', 'he', 'husband']):
        intent['gender'] = 'Men'

    # ── Specific item ──────────────────────────────────────────
    specific_items = [
        'dhoti', 'kurta', 'kurti', 'saree', 'lehenga',
        'sherwani', 'blazer', 'hoodie', 'jacket', 'jeans',
        'trouser', 'shorts', 'trackpant', 'jogger',
        'sneaker', 'sandal', 'heels', 'loafer',
        'watch', 'wallet', 'belt', 'bag', 'sunglass',
        'bodycon', 'gown', 'dress', 'skirt', 'palazzo',
        'anarkali', 'salwar', 'churidar', 'dupatta',
        'polo', 'sweatshirt', 'cardigan', 'vest',
        'shirt', 'tshirt',
    ]
    for item in specific_items:
        if item in msg:
            intent['specific_item'] = item
            break

    # ── Component types ────────────────────────────────────────
    component_keywords = {
        'Top': [
            'shirt', 'kurta', 'kurti', 'top', 'tshirt', 't-shirt',
            'blouse', 'saree', 'lehenga', 'hoodie', 'jacket',
            'sweater', 'blazer', 'coat', 'shrug', 'cardigan',
            'vest', 'tunic', 'jumpsuit', 'romper', 'dress',
            'gown', 'sweatshirt', 'polo', 'camisole', 'kaftan',
            'anarkali', 'salwar', 'kameez', 'ethnic', 'traditional',
            'bodycon', 'dungaree', 'tracksuit', 'crop',
            'sherwani', 'nehru', 'indo western',
        ],
        'Bottom': [
            'jeans', 'trouser', 'pant', 'dhoti', 'skirt', 'shorts',
            'legging', 'palazzo', 'jogger', 'chino', 'capri',
            'bermuda', 'cargo', 'culotte', 'tights', 'trackpant',
            'lungi', 'churidar', 'patiyala', 'bottom', 'sarong',
        ],
        'Footwear': [
            # Standard spellings
            'shoe', 'shoes', 'sneaker', 'sneakers', 'footwear', 'footware',
            'sandal', 'sandals', 'boot', 'boots', 'slipper', 'slippers',
            'heel', 'heels', 'loafer', 'loafers', 'flip flop', 'flipflop',
            'moccasin', 'wedge', 'platform', 'oxford', 'derby', 'espadrille',
            # Common misspellings / typos
            'shaker', 'shakers', 'shekers', 'shoues', 'shoess', 'sandle',
            'sandles', 'sneker', 'snekers', 'heeel', 'snadal',
            # Indian English / Myntra common terms
            'sports shoes', 'running shoes', 'casual shoes', 'formal shoes',
            'ethnic footwear', 'kolhapuri', 'mojari', 'juttis', 'jutti',
            'chappals', 'chappal',
            # Remaining original keywords
            'flat', 'mule', 'clog', 'stiletto',
        ],
        'Accessories': [
            'bag', 'handbag', 'tote', 'backpack', 'clutch', 'wallet',
            'purse', 'watch', 'bracelet', 'necklace', 'earring',
            'ring', 'bangle', 'anklet', 'chain', 'pendant',
            'sunglass', 'belt', 'cap', 'hat', 'scarf', 'stole',
            'muffler', 'sock', 'gloves', 'dupatta', 'shawl',
            'bra', 'mask', 'tie', 'brooch', 'jewel', 'jewelry',
        ],
    }
    for comp, keywords in component_keywords.items():
        if any(k in msg for k in keywords):
            intent['component_types'].append(comp)

    # ── Price ──────────────────────────────────────────────────
    range_patterns = [
        r'(\d+)\s*to\s*(\d+)',
        r'between\s*₹?\s*(\d+)\s*(?:and|to)\s*₹?\s*(\d+)',
        r'₹?\s*(\d+)\s*-\s*₹?\s*(\d+)',
        r'range\s*(?:of)?\s*₹?\s*(\d+)\s*(?:to|-)\s*₹?\s*(\d+)',
        r'from\s*₹?\s*(\d+)\s*to\s*₹?\s*(\d+)',
    ]
    range_found = False
    for pattern in range_patterns:
        match = re.search(pattern, msg)
        if match:
            intent['min_price'] = int(match.group(1))
            intent['max_price'] = int(match.group(2))
            range_found = True
            break

    if not range_found:
        price_patterns = [
            r'under\s*₹?\s*(\d+)',
            r'below\s*₹?\s*(\d+)',
            r'less\s*than\s*₹?\s*(\d+)',
            r'within\s*₹?\s*(\d+)',
            r'budget\s*(?:of|is|:)?\s*₹?\s*(\d+)',
            r'max\s*₹?\s*(\d+)',
            r'upto\s*₹?\s*(\d+)',
            r'up\s*to\s*₹?\s*(\d+)',
            r'₹\s*(\d+)',
            r'rs\.?\s*(\d+)',
            r'(\d+)\s*(?:rupees|rs\.?|inr)',
        ]
        prices_found = []
        for pattern in price_patterns:
            matches = re.findall(pattern, msg)
            prices_found.extend([int(m) for m in matches])
        if prices_found:
            intent['max_price'] = max(prices_found)

    # ── Colors ─────────────────────────────────────────────────
    color_list = [
        'black', 'white', 'blue', 'red', 'green', 'yellow',
        'pink', 'grey', 'gray', 'purple', 'orange', 'brown',
        'beige', 'maroon', 'navy', 'cream', 'olive', 'coral',
        'teal', 'gold', 'silver', 'printed', 'floral',
        'multicolor', 'striped', 'checked',
    ]
    intent['colors'] = [c for c in color_list if c in msg]

    # ── Occasion ───────────────────────────────────────────────
    occasion_map = {
        'festival': ['pongal', 'diwali', 'holi', 'eid', 'onam',
                     'navratri', 'festival', 'traditional',
                     'ethnic', 'cultural', 'pooja', 'puja'],
        'party'   : ['party', 'club', 'night out', 'cocktail',
                     'dance', 'celebration', 'birthday'],
        'college' : ['college', 'campus', 'casual', 'everyday',
                     'daily', 'class', 'uni', 'university'],
        'wedding' : ['wedding', 'marriage', 'reception', 'engagement',
                     'bride', 'groom', 'sangeet', 'mehendi'],
        'office'  : ['office', 'work', 'formal', 'business',
                     'professional', 'interview', 'meeting'],
        'sports'  : ['gym', 'sport', 'workout', 'running',
                     'athletic', 'fitness', 'yoga', 'training'],
    }
    for occasion, words in occasion_map.items():
        if any(w in msg for w in words):
            intent['occasion'] = occasion
            break

    # ── Style keywords ─────────────────────────────────────────
    style_words = [
        'cotton', 'silk', 'linen', 'polyester', 'wool', 'denim',
        'printed', 'solid', 'embroidered', 'striped', 'floral',
        'slim fit', 'regular fit', 'oversized', 'bodycon',
        'handloom', 'khadi', 'maxi', 'mini', 'midi',
        'sleeveless', 'full sleeve', 'half sleeve',
    ]
    intent['keywords'] = [w for w in style_words if w in msg]

    return intent


# ══════════════════════════════════════════════════════════════
# PART 2 — FILTER & RECOMMEND
# ══════════════════════════════════════════════════════════════
def recommend(intent: dict, top_n: int = 5) -> dict:

    results = {}
    df      = CATALOG.copy()

    # ── Filter 1: Minimum reviews ──────────────────────────────
    if 'rating_count' in df.columns:
        df = df[df['rating_count'] >= 10]

    # ── Filter 2: Gender ───────────────────────────────────────
    if intent['gender'] == 'Men':
        df = df[df['gender'].isin(['Men', 'Unisex'])]
        df = df[~df['name'].str.contains(
            r'\bwomen\b|\bwomens\b|\bwoman\b|'
            r'\bladies\b|\bgirls\b|\bfemale\b',
            na=False, regex=True)]
        df = df[~df['brand'].str.lower().isin(
            [b.lower() for b in WOMENS_BRANDS])]

    elif intent['gender'] == 'Women':
        df = df[df['gender'].isin(['Women', 'Unisex'])]
        df = df[~df['name'].str.contains(
            r'\bmen\b|\bmens\b|\bboys\b|\bmale\b|\bgents\b',
            na=False, regex=True)]
        df = df[~df['brand'].str.lower().isin(
            [b.lower() for b in MENS_ONLY_BRANDS])]

    else:
        df = df[df['gender'].isin(['Men', 'Women', 'Unisex'])]

    # ── Filter 3: Remove multi-item packs ─────────────────────
    df = df[~df['name'].str.contains(
        r'pack\s*of\s*\d|pack\s*of\s*two|pack\s*of\s*three|'
        r'\d[\s-]+pack\b|set\s*of\s*\d|'
        r'\d\s*piece\s*set|\d\s*pcs\s*set',
        na=False, regex=True)]

    # ── Filter 4: Remove innerwear unless asked ────────────────
    if intent['specific_item'] not in [
            'brief', 'trunk', 'boxer', 'bra', 'innerwear']:
        df = df[~df['name'].str.contains(
            r'\bbrief\b|\bbriefs\b|\btrunk\b|\btrunks\b|'
            r'\bboxer\b|\bboxers\b|\bbra\b|'
            r'hipster|thong|\bundershirt\b',
            na=False, regex=True)]

    # ── Filter 5: Item-specific exclusions ────────────────────
    if intent['specific_item'] == 'shirt':
        # Remove all tshirt variations
        df = df[~df['name'].str.contains(
            r't-shirt|tshirt|\bt\s+shirt|'
            r'\btee\b|hoodie|sweatshirt|'
            r'\bpolo\b|training|cottot',
            na=False, regex=True)]

    elif intent['specific_item'] == 'jeans':
        df = df[~df['name'].str.contains(
            r'\bshorts\b|\btrack\b|\bjogger\b|\bchino\b',
            na=False, regex=True)]

    elif intent['specific_item'] == 'kurta':
        df = df[~df['name'].str.contains(
            r't-shirt|tshirt|\bjeans\b|\bshorts\b',
            na=False, regex=True)]

    # ── Filter 6: Price ────────────────────────────────────────
    if intent['max_price']:
        df = df[df['price'] <= intent['max_price']]
    if intent.get('min_price'):
        df = df[df['price'] >= intent['min_price']]

    # ── Decide components ──────────────────────────────────────
    components = intent['component_types']
    if not components:
        components = ['Top', 'Bottom']

    # ── For each component ─────────────────────────────────────
    for comp in components:
        sub = df[df['component_type'] == comp].copy()

        if len(sub) == 0:
            continue

        # ── Specific item first ────────────────────────────────
        if intent['specific_item']:
            item  = intent['specific_item']
            exact = sub[sub['name'].str.contains(
                item, na=False)]
            rest  = sub[~sub['name'].str.contains(
                item, na=False)]
            if len(exact) >= top_n:
                sub = exact
            elif len(exact) > 0:
                sub = pd.concat([exact, rest])

        # ── Usage-based filter (NEW — uses cleaned column) ─────
        # Maps occasion → usage label in dataset
        # Much more accurate than keyword matching
        if intent['occasion']:
            target_usage = OCCASION_USAGE.get(
                intent['occasion'])

            if target_usage:
                usage_match = sub[
                    sub['usage'] == target_usage]

                if len(usage_match) >= top_n:
                    # Enough matches → use only these
                    sub = usage_match

                elif len(usage_match) > 0:
                    # Some matches → show these first
                    rest = sub[sub['usage'] != target_usage]
                    sub  = pd.concat([usage_match, rest])

                # Zero matches → keep all (show best available)
                # This handles "formal shirt" data limitation

        # ── Color filter ───────────────────────────────────────
        if intent['colors']:
            color_pattern = '|'.join(intent['colors'])
            color_match   = (
                sub['color'].str.lower().str.contains(
                    color_pattern, na=False) |
                sub['name'].str.contains(
                    color_pattern, na=False)
            )
            colored = sub[color_match]
            rest    = sub[~color_match]
            if len(colored) > 0:
                sub = pd.concat([colored, rest])

        # ── Style keyword boost ────────────────────────────────
        if intent['keywords']:
            kw_pattern = '|'.join(intent['keywords'])
            kw_mask    = sub['name'].str.contains(
                kw_pattern, na=False)
            boosted    = sub[kw_mask]
            rest       = sub[~kw_mask]
            sub        = pd.concat([boosted, rest])

        # ── Sort by weighted score ─────────────────────────────
        sub = sub.copy()
        if 'rating_count' in sub.columns:
            sub['score'] = (
                sub['rating'] *
                np.log10(
                    sub['rating_count'].clip(lower=1) + 1)
            )
            sub = sub.sort_values(
                ['score', 'price'],
                ascending=[False, True])
        else:
            sub = sub.sort_values(
                ['rating', 'price'],
                ascending=[False, True])

        # ── Remove duplicate names ─────────────────────────────
        sub = sub.drop_duplicates(
            subset=['name'], keep='first')

        # ── Brand diversity — max 2 per brand ─────────────────
        brand_counts = {}
        diverse_rows = []
        for _, row in sub.iterrows():
            brand = row['brand']
            count = brand_counts.get(brand, 0)
            if count < 2:
                diverse_rows.append(row)
                brand_counts[brand] = count + 1
            if len(diverse_rows) >= top_n:
                break

        if diverse_rows:
            sub = pd.DataFrame(diverse_rows)

        top = sub.head(top_n)

        if len(top) > 0:
            results[comp] = top[[
                'product_id', 'name', 'brand', 'price', 'color',
                'image_url', 'product_url',
                'rating', 'gender', 'rating_count',
                'usage'
            ]].to_dict('records')

    return results


# ══════════════════════════════════════════════════════════════
# PART 3 — MAIN FUNCTION
# ══════════════════════════════════════════════════════════════
def get_recommendations(
        user_message: str,
        top_n: int = 5) -> tuple:
    intent  = parse_message(user_message)
    results = recommend(intent, top_n=top_n)
    return intent, results


# ══════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":

    tests = [
        "traditional kurta for pongal under 2000",
        "party dress for girl in range of 2500 to 5000",
        "casual jeans for college under 3000",
        "gym wear for men under 1500",
        "formal shirt for men under 2000",
        "white dhoti for men under 1500",
        "wedding lehenga for women under 5000",
    ]

    for msg in tests:
        print("\n" + "=" * 55)
        print(f"USER : {msg}")
        print("=" * 55)
        intent, results = get_recommendations(msg)
        print(f"  gender   : {intent['gender']}")
        print(f"  price    : "
              f"min=₹{intent['min_price']} "
              f"max=₹{intent['max_price']}")
        print(f"  item     : {intent['specific_item']}")
        print(f"  occasion : {intent['occasion']}")

        for comp, products in results.items():
            print(f"\n  [{comp}]")
            for p in products:
                name  = str(p['name'])[:50].title()
                usage = p.get('usage', '')
                print(f"    • {name}")
                print(f"      {p['brand']} | "
                      f"₹{int(p['price'])} | "
                      f"⭐{p['rating']} | "
                      f"Usage: {usage}")