# component_detector.py
# Step 3: Outfit Component Detection
# Uses YOLOv8 (already downloaded) + pixel-level region analysis.
# No paid API — runs fully locally.

import sys
import uuid
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))
from detection.detector import detect_clothing_components, model as yolo_model


# ─────────────────────────────────────────────────────────────────────────────
# Named-color lookup table  (RGB → label)
# ─────────────────────────────────────────────────────────────────────────────
_COLOR_TABLE = {
    (220, 20,  60):  "red",
    (255, 99,  71):  "red",
    (255, 165,  0):  "orange",
    (255, 215,  0):  "yellow",
    (34,  139, 34):  "green",
    (0,   128, 128): "teal",
    (0,   0,   205): "blue",
    (0,   0,   128): "navy blue",
    (75,  0,   130): "indigo",
    (148, 0,   211): "violet",
    (255, 20,  147): "pink",
    (255, 182, 193): "light pink",
    (0,   0,   0):   "black",
    ( 10,  10,  10): "black",
    ( 15,  20,  30): "black",
    ( 20,  25,  35): "black",
    ( 25,  30,  40): "dark navy",
    ( 10,  15,  25): "black",
    (255, 255, 255): "white",
    (128, 128, 128): "grey",
    (192, 192, 192): "silver",
    (165, 42,  42):  "brown",
    (210, 180, 140): "tan",
    (245, 222, 179): "cream",
    (128, 0,   0):   "maroon",
    (0,   100,  0):  "dark green",
    (184, 134,  11): "golden",
    (218, 165,  32): "golden",
    (205, 133,  63): "golden brown",
    (210, 105,  30): "bronze",
    ( 72, 209, 204): "teal",
    ( 32, 178, 170): "teal",
    (  0, 139, 139): "dark teal",
    ( 70, 130, 180): "steel blue",
    ( 25,  25, 112): "midnight blue",
    ( 44,  62,  80): "dark navy",
    ( 52,  73,  94): "dark navy",
    ( 33,  33,  33): "charcoal",
    ( 64,  64,  64): "dark grey",
    (101,  67,  33): "dark brown",
    (139,  90,  43): "brown",
    (188, 143, 143): "dusty rose",
    (219, 112, 147): "pink",
    (255,  20, 147): "hot pink",
    (128,   0, 128): "purple",
    (153,  50, 204): "purple",
    (  8,  24,  58): "navy blue",
    ( 15,  30,  68): "navy blue",
    ( 20,  20,  60): "navy blue",
    ( 30,  30,  80): "dark navy",
    ( 10,  40,  60): "dark teal blue",
    ( 20,  52,  80): "dark blue",
    ( 84, 160, 134): "teal",
    ( 70, 150, 120): "teal",
    ( 90, 170, 140): "teal",
    ( 65, 145, 115): "teal",
    ( 95, 165, 145): "teal",
}


def _dominant_color(image: Image.Image, region: tuple = None) -> str:
    """
    Return the closest named color for a region of the image.

    Args:
        image:  PIL Image (RGB)
        region: (left, top, right, bottom) — crops before analysing; None = full image
    """
    try:
        img = image.copy().convert("RGB")
        if region:
            img = img.crop(region)
        img = img.resize((40, 40), Image.LANCZOS)
        arr = np.array(img, dtype=float)

        # Filter near-white pixels (plain white background)
        near_white = (
            (arr[:, :, 0] > 220) &
            (arr[:, :, 1] > 220) &
            (arr[:, :, 2] > 220)
        )

        # Filter near-grey neutral pixels (checkered background)
        # Grey pixels have nearly equal R, G, B values
        # and are not very dark (not clothing shadows)
        ch_max = arr.max(axis=2)
        ch_min = arr.min(axis=2)
        saturation_range = ch_max - ch_min
        near_grey = (
            (saturation_range < 25) &
            (arr[:, :, 0] > 100) &
            (arr[:, :, 1] > 100) &
            (arr[:, :, 2] > 100)
        )

        # Skin-tone exclusion — warm pixels where R > G > B
        # Covers all skin tones from light to very dark.
        # Skin signature: red channel dominates over blue,
        # green is between them, warm spread R-B > 20.
        is_skin = (
            (arr[:, :, 0] > arr[:, :, 2]) &
            (arr[:, :, 1] > arr[:, :, 2]) &
            (arr[:, :, 0] - arr[:, :, 2] > 20) &
            (arr[:, :, 0] < 230) &
            (saturation_range < 80)
        )

        # Determine if garment is predominantly light coloured
        # (white shoes, cream shirts) — if so, don't filter
        # near-white pixels, only remove transparent black
        all_px = arr.reshape(-1, 3)
        light_count = (
            (all_px[:, 0] > 200) &
            (all_px[:, 1] > 200) &
            (all_px[:, 2] > 200)
        ).sum()
        garment_is_light = (light_count / len(all_px) > 0.40)
        if garment_is_light:
            # Filter near-black pixels — transparent PNG
            # areas become black (0,0,0) on RGB conversion
            near_black = (
                (arr[:, :, 0] < 30) &
                (arr[:, :, 1] < 30) &
                (arr[:, :, 2] < 30)
            )
            mask = ~near_black
        else:
            # Filter near-black pixels — transparent PNG
            # areas become black (0,0,0) on RGB conversion
            near_black = (
                (arr[:, :, 0] < 30) &
                (arr[:, :, 1] < 30) &
                (arr[:, :, 2] < 30)
            )
            mask = ~(near_white | near_grey |
                     near_black | is_skin)

        if mask.sum() > 10:
            clothing_pixels = arr[mask]
            # Find pixel with highest saturation
            # This reliably finds fabric colour on
            # patterned garments — teal wins over
            # gold wins over background, every time
            ch_max = clothing_pixels.max(axis=1)
            ch_min = clothing_pixels.min(axis=1)
            sat_vals = ch_max - ch_min
            best_pixel_idx = sat_vals.argmax()
            avg = clothing_pixels[best_pixel_idx]
        else:
            avg = arr.mean(axis=(0, 1))

        r, g, b = int(avg[0]), int(avg[1]), int(avg[2])

        best_name, best_dist = "multicolor", float("inf")
        for (cr, cg, cb), name in _COLOR_TABLE.items():
            d = ((r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2) ** 0.5
            if d < best_dist:
                best_dist, best_name = d, name
        return best_name
    except Exception:
        return "multicolor"


def _color_from_mask(
        image: Image.Image,
        mask_array: np.ndarray) -> str:
    try:
        arr = np.array(
            image.convert('RGB'), dtype=float)
        raw_pixels = arr[mask_array]

        # Apply same filters as _dominant_color()
        # to remove background and skin pixels
        r_ch = raw_pixels[:, 0]
        g_ch = raw_pixels[:, 1]
        b_ch = raw_pixels[:, 2]

        ch_max = raw_pixels.max(axis=1)
        ch_min = raw_pixels.min(axis=1)
        sat_range = ch_max - ch_min

        # Check if garment is predominantly light coloured
        # (white/cream shoes, white shirts etc.)
        # If >40% of pixels are near-white,
        # the garment IS light coloured — don't filter
        total = len(raw_pixels)
        light_pixels = (
            (r_ch > 200) & (g_ch > 200) & (b_ch > 200)
        ).sum()
        garment_is_light = (
            total > 0 and
            light_pixels / total > 0.40
        )

        if garment_is_light:
            # Keep all pixels — garment is white/cream
            # Only remove pure transparent-black pixels
            near_black_only = (
                (r_ch < 30) &
                (g_ch < 30) &
                (b_ch < 30)
            )
            clothing_pixels = raw_pixels[
                ~near_black_only
            ] if (~near_black_only).sum() > 10 \
              else raw_pixels
        else:
            # Normal path — apply all filters
            near_white = (
                (r_ch > 220) &
                (g_ch > 220) &
                (b_ch > 220)
            )
            near_grey = (
                (sat_range < 25) &
                (r_ch > 100) &
                (g_ch > 100) &
                (b_ch > 100)
            )
            is_skin = (
                (r_ch > b_ch) & (g_ch > b_ch) &
                (r_ch - b_ch > 20) & (r_ch < 230) &
                (sat_range < 80)
            )
            near_black = (
                (r_ch < 30) &
                (g_ch < 30) &
                (b_ch < 30)
            )
            keep = ~(near_white | near_grey |
                     near_black | is_skin)
            clothing_pixels = (
                raw_pixels[keep]
                if keep.sum() > 10
                else raw_pixels
            )

        if len(clothing_pixels) < 10:
            return "multicolor"
        from sklearn.cluster import KMeans
        import warnings
        n = min(5, max(2,
            len(clothing_pixels) // 50))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            km = KMeans(n_clusters=n, n_init=3,
                max_iter=100, random_state=0)
            km.fit(clothing_pixels)
        best_idx, best_score = 0, -1
        counts = np.bincount(km.labels_)
        for i, center in enumerate(
                km.cluster_centers_):
            r, g, b = center
            mx = max(r, g, b)
            mn = min(r, g, b)
            # Saturation — how colourful is this cluster
            sat = (mx-mn)/mx if mx > 0 else 0
            # Distance from grey — clusters far from
            # grey (128,128,128) are more likely fabric
            grey_dist = (
                ((r-128)**2 +
                 (g-128)**2 +
                 (b-128)**2) ** 0.5
            ) / 221.7   # normalise to 0-1
            # Brightness weight — not too dark, not
            # too bright (clothing sits in mid-range)
            brightness = (r + g + b) / 765.0
            bright_w = 1.0 - abs(brightness - 0.45)
            cw = counts[i] / len(km.labels_)
            score = (
                sat * 0.50 +
                grey_dist * 0.30 +
                bright_w * 0.10 +
                cw * 0.10
            )
            if score > best_score:
                best_score = score
                best_idx = i
        avg = km.cluster_centers_[best_idx]
        r, g, b = (int(avg[0]),
                   int(avg[1]), int(avg[2]))
        best_name, best_dist = (
            "multicolor", float("inf"))
        for (cr, cg, cb), name in (
                _COLOR_TABLE.items()):
            d = ((r-cr)**2 +
                 (g-cg)**2 +
                 (b-cb)**2)**0.5
            if d < best_dist:
                best_dist, best_name = d, name
        return best_name
    except Exception:
        return "multicolor"


# ─────────────────────────────────────────────────────────────────────────────
# Region-based label builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_labels(image: Image.Image, component_types: list) -> dict:
    """
    Divide the image into horizontal bands and extract dominant colors.

        Upper band  (0 – 50%)   → Topwear
        Middle band (50 – 80%)  → Bottomwear
        Lower band  (80 – 100%) → Footwear
        Full image              → Accessories (handbag, tie, etc.)
    """
    w, h = image.size
    bands = {
        "Topwear":    (0, int(h * 0.20), w, int(h * 0.58)),
        "Bottomwear": (0, int(h * 0.50), w, int(h * 0.80)),
        "Footwear":   (0, int(h * 0.80), w, h),
        "Accessories":(0, 0,            w, h),
    }

    labels = {}

    if "Topwear" in component_types:
        color = _dominant_color(image, bands["Topwear"])
        labels["top_label"]  = f"{color} top"
        labels["top_color"]  = color

    if "Bottomwear" in component_types:
        color = _dominant_color(image, bands["Bottomwear"])
        labels["bottom_label"]  = f"{color} bottom"
        labels["bottom_color"]  = color

    if "Footwear" in component_types:
        color = _dominant_color(image, bands["Footwear"])
        labels["footwear_label"] = f"{color} footwear"

    if "Accessories" in component_types:
        labels["accessory_labels"] = "accessories"

    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Per-bounding-box detail  (best-effort, non-critical)
# ─────────────────────────────────────────────────────────────────────────────

def _yolo_box_details(image: Image.Image) -> dict:
    """
    Run YOLOv8 inference and return per-detection metadata.
    Each entry: { class_name: {confidence, bbox, color} }
    Only detections above 0.30 confidence are included.
    """
    img_arr = np.array(image)
    results  = yolo_model(img_arr, verbose=False)
    details  = {}

    for r in results:
        for box in r.boxes:
            cls_name = yolo_model.names[int(box.cls)].lower()
            conf     = float(box.conf)
            if conf < 0.30:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            color = _dominant_color(image, (x1, y1, x2, y2))
            details[cls_name] = {
                "confidence": round(conf, 2),
                "bbox":       (x1, y1, x2, y2),
                "color":      color,
            }

    return details


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def analyze_outfit(image: Image.Image) -> dict:
    """
    Full detection pipeline for one image.

    Steps:
      1. YOLOv8  → high-level categories (Topwear / Bottomwear / Footwear / Accessories)
      2. Fallback → [Topwear, Bottomwear] when nothing is detected
      3. Region   → dominant-color extraction per band
      4. Boxes    → per-detection detail (best-effort)

    Returns dict with keys:
        outfit_id, component_types, top_label, bottom_label,
        footwear_label, accessory_labels, box_details, detected_at
    """
    # Step 1 — SegFormer pixel-level masks
    try:
        import sys
        from pathlib import Path
        sys.path.append(str(
            Path(__file__).resolve().parents[1]))
        from vision.segmentation import (
            segment_clothing)
        seg   = segment_clothing(image)
        regs  = seg.get('regions', {})
        if regs:
            labels = {}
            for cat, data in regs.items():
                mask = data.get('mask')
                if mask is None or not mask.any():
                    continue
                colour = _color_from_mask(
                    image, mask)
                if cat == 'Topwear':
                    labels['top_label'] = (
                        f"{colour} top")
                    labels['top_color'] = colour
                elif cat == 'Bottomwear':
                    labels['bottom_label'] = (
                        f"{colour} bottom")
                elif cat == 'Footwear':
                    labels['footwear_label'] = (
                        f"{colour} footwear")
                elif cat == 'Accessories':
                    labels['accessory_labels'] = (
                        "accessories")
            comp = seg.get('labels', [])
            if labels:
                print(f"[DETECTOR] SegFormer: "
                      f"{labels}")
                return {
                  "outfit_id": str(uuid.uuid4()),
                  "component_types": comp,
                  "top_label": labels.get(
                      "top_label",""),
                  "bottom_label": labels.get(
                      "bottom_label",""),
                  "footwear_label": labels.get(
                      "footwear_label",""),
                  "accessory_labels": labels.get(
                      "accessory_labels",""),
                  "box_details": {},
                  "detected_at": (
                      datetime.now().isoformat()),
                }
    except Exception as e:
        print(f"[DETECTOR] SegFormer failed: {e}")

    # Step 2 — Zone-based fallback
    print("[DETECTOR] Zone fallback")
    comp = detect_clothing_components(image)
    if not comp:
        comp = ["Topwear", "Bottomwear"]
    labels = _build_labels(image, comp)
    try:
        box_details = _yolo_box_details(image)
    except Exception:
        box_details = {}
    return {
        "outfit_id": str(uuid.uuid4()),
        "component_types": comp,
        "top_label": labels.get(
            "top_label",""),
        "bottom_label": labels.get(
            "bottom_label",""),
        "footwear_label": labels.get(
            "footwear_label",""),
        "accessory_labels": labels.get(
            "accessory_labels",""),
        "box_details": box_details,
        "detected_at": (
            datetime.now().isoformat()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Smoke-test  (run directly: python component_detector.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    # Synthetic test image: blue upper half, dark lower half
    test_img = Image.new("RGB", (300, 600))
    px = test_img.load()
    for y in range(600):
        for x in range(300):
            px[x, y] = (30, 80, 180) if y < 300 else (20, 20, 20)

    result = analyze_outfit(test_img)
    printable = {k: v for k, v in result.items() if k != "box_details"}
    print(json.dumps(printable, indent=2))
    print("Box details:", result["box_details"])
