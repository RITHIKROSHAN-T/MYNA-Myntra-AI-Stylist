# segmentation.py
# SegFormer B2 — pixel-level clothing segmentation
# Falls back to zone-based if model cannot load.
# Does NOT replace component_detector.py
# Used to VISUALIZE segmentation on photo

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import cv2

BASE_DIR = Path(__file__).resolve().parents[2]

# ── Colour scheme ──────────────────────────────────────────────
REGION_COLORS = {
    'Topwear'    : (0,   210, 210),   # Cyan
    'Bottomwear' : (255, 140,   0),   # Orange
    'Footwear'   : (180,  80, 255),   # Purple
    'Accessories': (255, 215,   0),   # Gold
}

# ── SegFormer label → MYNA category ────────────────────────────
LABEL_MAP = {
    'Upper-clothes': 'Topwear',
    'Dress'        : 'Topwear',
    'Scarf'        : 'Topwear',
    'coat'         : 'Topwear',
    'jacket'       : 'Topwear',
    'Pants'        : 'Bottomwear',
    'Skirt'        : 'Bottomwear',
    'Left-shoe'    : 'Footwear',
    'Right-shoe'   : 'Footwear',
    'Hat'          : 'Accessories',
    'Sunglasses'   : 'Accessories',
    'Belt'         : 'Accessories',
    'Bag'          : 'Accessories',
}

IGNORE_LABELS = {
    'Hair', 'Face', 'Left-leg', 'Right-leg',
    'Left-arm', 'Right-arm', 'Background',
    'skin', 'Left-hand', 'Right-hand',
}

SEG_PIPE = None


def _load_segformer():
    global SEG_PIPE
    if SEG_PIPE is not None:
        return SEG_PIPE
    try:
        from transformers import pipeline
        print("[SEG] Loading SegFormer B2 clothes model…")
        SEG_PIPE = pipeline(
            "image-segmentation",
            model="mattmdjaga/segformer_b2_clothes"
        )
        print("[SEG] SegFormer B2 loaded!")
    except Exception as e:
        print(f"[SEG] SegFormer load failed: {e}")
        SEG_PIPE = None
    return SEG_PIPE


def _draw_label_badge(draw, text, x, y, color, font=None):
    """Draw a filled rounded-rect badge with white text."""
    pad = 6
    if font:
        bbox = draw.textbbox((0, 0), text, font=font)
    else:
        bbox = draw.textbbox((0, 0), text)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    rx0, ry0 = x, y
    rx1, ry1 = x + tw + pad * 2, y + th + pad * 2
    draw.rounded_rectangle(
        [rx0, ry0, rx1, ry1],
        radius=6,
        fill=color
    )
    draw.text(
        (rx0 + pad, ry0 + pad),
        text,
        fill=(255, 255, 255),
        font=font
    )


def _segformer_segment(image: Image.Image):
    """
    Run SegFormer B2 pipeline and return annotated image + regions.
    Returns None if the pipeline could not be loaded (triggers fallback).
    """
    pipe = _load_segformer()
    if pipe is None:
        return None

    rgb  = image.convert('RGB')
    w, h = rgb.size

    # Run segmentation — returns list of
    # {'label': str, 'score': float, 'mask': PIL.Image}
    seg_results = pipe(rgb)

    # Accumulate masks per MYNA category (union of all matching labels)
    category_masks  = {}
    category_labels = {}   # first raw SegFormer label per category
    for item in seg_results:
        label = item['label']
        if label in IGNORE_LABELS:
            continue
        category = LABEL_MAP.get(label) or LABEL_MAP.get(label.lower())
        if category is None:
            continue

        # item['mask'] is a PIL 'L' image, pixel=255 where present
        mask_arr = np.array(item['mask']) > 128  # bool H×W

        if category not in category_masks:
            category_masks[category]  = mask_arr
            category_labels[category] = label
        else:
            category_masks[category] = (
                category_masks[category] | mask_arr
            )

    if not category_masks:
        # SegFormer ran but found nothing — still prefer over zones
        print("[SEG] SegFormer found no clothing; using zone fallback")
        return None

    # Build composited output image
    output = rgb.copy().convert('RGBA')
    regions = {}
    labels  = []

    for category, mask_arr in category_masks.items():
        color_rgb  = REGION_COLORS[category]
        fill_alpha = 90   # semi-transparent fill

        # ── 1. Semi-transparent fill on clothing pixels ──────────
        fill_arr           = np.zeros((h, w, 4), dtype=np.uint8)
        fill_arr[mask_arr] = color_rgb + (fill_alpha,)
        fill_layer         = Image.fromarray(fill_arr, 'RGBA')
        output             = Image.alpha_composite(output, fill_layer)

        # ── 2. Contour outline ───────────────────────────────────
        mask_u8     = mask_arr.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_u8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        outline_layer = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        outline_draw  = ImageDraw.Draw(outline_layer)
        for cnt in contours:
            pts = [tuple(p[0]) for p in cnt]
            if len(pts) >= 2:
                outline_draw.line(
                    pts + [pts[0]],
                    fill=color_rgb + (220,),
                    width=3
                )
        output = Image.alpha_composite(output, outline_layer)

        # ── 3. Label badge at top of detected region ─────────────
        ys, xs = np.where(mask_arr)
        if len(xs):
            bx0, by0 = int(xs.min()), int(ys.min())
            bx1, by1 = int(xs.max()), int(ys.max())
        else:
            bx0, by0, bx1, by1 = 0, 0, w, h

        label_layer = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        label_draw  = ImageDraw.Draw(label_layer)
        badge_x = bx0 + 6
        badge_y = max(by0 + 6, 4)
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except Exception:
            font = None
        _draw_label_badge(
            label_draw, category,
            badge_x, badge_y,
            color_rgb, font
        )
        output = Image.alpha_composite(output, label_layer)

        if category not in regions:
            regions[category] = {
                'mask'     : mask_arr,
                'color'    : color_rgb,
                'bbox'     : (bx0, by0, bx1, by1),
                'raw_label': category_labels.get(
                    category, category),
            }
        else:
            regions[category]['mask'] = (
                regions[category]['mask'] |
                mask_arr
            )
        labels.append(category)

    result_img = output.convert('RGB')
    print(f"[SEG] SegFormer detected: {labels}")
    return {
        'annotated_image': result_img,
        'regions'        : regions,
        'labels'         : labels,
    }


def _zone_fallback(image: Image.Image) -> dict:
    """Original zone-based fallback (used when SegFormer unavailable)."""
    overlay = image.convert('RGBA').copy()
    draw    = ImageDraw.Draw(overlay)
    w, h    = image.size

    zone_colors = {
        'Topwear'   : (0,   210, 210, 100),
        'Bottomwear': (255, 140,   0, 100),
        'Footwear'  : (180,  80, 255, 100),
    }
    zones = {
        'Topwear'   : (0, 0,              w, int(h * 0.50)),
        'Bottomwear': (0, int(h * 0.50),  w, int(h * 0.80)),
        'Footwear'  : (0, int(h * 0.80),  w, h),
    }

    regions = {}
    labels  = []

    for zone_name, bbox in zones.items():
        color = zone_colors[zone_name]
        draw.rectangle(
            bbox,
            fill=color,
            outline=color[:3] + (255,),
            width=2
        )
        draw.text(
            (bbox[0] + 10, bbox[1] + 10),
            zone_name,
            fill=(255, 255, 255, 255)
        )
        regions[zone_name] = {
            'bbox' : bbox,
            'color': color[:3],
        }
        labels.append(zone_name)

    result_img = Image.alpha_composite(
        image.convert('RGBA'), overlay
    ).convert('RGB')

    print(f"[SEG] Zone fallback used: {labels}")
    return {
        'annotated_image': result_img,
        'regions'        : regions,
        'labels'         : labels,
    }


def segment_clothing(image: Image.Image) -> dict:
    """
    Segment clothing parts in person photo using SegFormer B2.
    Falls back to zone-based segmentation if model unavailable.

    Returns:
        {
          'annotated_image': PIL.Image with coloured masks,
          'regions': {
              'Topwear'    : {'bbox': (x0,y0,x1,y1),
                              'color': (r,g,b), 'mask': ndarray},
              'Bottomwear' : {...},
              'Footwear'   : {...},
              'Accessories': {...},
          },
          'labels': ['Topwear', 'Bottomwear', ...]
        }
    """
    try:
        result = _segformer_segment(image)
        if result is not None:
            return result
    except Exception as e:
        print(f"[SEG] SegFormer error: {e}")

    return _zone_fallback(image)


if __name__ == "__main__":
    test_img = Image.new('RGB', (400, 600), (200, 180, 160))
    result   = segment_clothing(test_img)
    result['annotated_image'].save('test_segmentation.png')
    print("Labels:", result['labels'])
    print("Saved test_segmentation.png")
