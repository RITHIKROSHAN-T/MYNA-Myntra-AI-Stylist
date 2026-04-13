# outfit_generator.py
# AI Outfit Generation via HuggingFace IDM-VTON Space (no Colab needed)
#
# virtual_tryon()            – person photo + garment image  → IDM-VTON try-on
# generate_outfit_from_prompt() – text prompt → PIL garment  → virtual_tryon()
# is_api_available()         – lightweight check: HF Space reachable?

import os
import random
import requests
from PIL import Image, ImageDraw
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Try multiple .env locations
_root = Path(__file__).resolve().parents[2]
_env_paths = [
    _root / ".env",
    _root / "src" / ".env",
    Path(".env"),
]
for _ep in _env_paths:
    if _ep.exists():
        load_dotenv(_ep, override=True)
        break

BASE_DIR   = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "data" / "generated_outfits"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "").strip()
# Token is loaded at call time — module-level check is intentionally skipped
HF_SPACE = "yisol/IDM-VTON"

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_client():
    from gradio_client import Client
    token = os.getenv("HUGGINGFACE_TOKEN", "").strip()
    if not token:
        # Try reading directly from .env file
        root = Path(__file__).resolve().parents[2]
        env_path = root / ".env"
        if env_path.exists():
            for line in env_path.read_text(
                    encoding='utf-8-sig').splitlines():
                if line.startswith("HUGGINGFACE_TOKEN="):
                    token = line.split("=", 1)[1].strip()
                    break
    print(f"[IDM-VTON] Token: {'YES '+token[:8] if token else 'NOT FOUND'}")
    client = Client(
        "yisol/IDM-VTON",
        token=token if token else None,
        verbose=False,
    )
    return client


def _save_image(image: Image.Image, session_id: str) -> str:
    """Save PIL image to OUTPUT_DIR and return the path string."""
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"outfit_{session_id}_{ts}.png"
    path     = OUTPUT_DIR / filename
    image.save(path)
    return str(path)


def _save_temp(image: Image.Image, name: str) -> str:
    """Save a temporary file for gradio_client upload; return path string."""
    path = OUTPUT_DIR / f"_tmp_{name}.png"
    image.save(path)
    return str(path)


def _cleanup(*paths):
    """Delete temporary files silently."""
    for p in paths:
        try:
            Path(p).unlink(missing_ok=True)
        except Exception:
            pass


def _fresh_seed() -> int:
    """
    Return a random seed on every call.
    Prevents HuggingFace Spaces from reusing cached denoising
    state across multiple try-on requests in the same session.
    seed=42 hardcoded causes smoky/degraded texture on 2nd/3rd click.
    """
    return random.randint(0, 2_147_483_647)


def _tmp_name(prefix: str, session_id: str) -> str:
    """
    Unique temp filename using session_id + microsecond timestamp.
    Prevents stale temp files being re-read on rapid successive calls.
    """
    ts = datetime.now().strftime("%H%M%S%f")
    return str(OUTPUT_DIR / f"_{prefix}_{session_id}_{ts}.png")


# ─────────────────────────────────────────────────────────────────────────────
# Color / garment helpers for prompt-based generation
# ─────────────────────────────────────────────────────────────────────────────

_COLOR_MAP = {
    "red":    (210, 40,  40),
    "blue":   (40,  90,  200),
    "green":  (40,  150, 70),
    "yellow": (230, 200, 40),
    "orange": (230, 120, 40),
    "pink":   (230, 90,  150),
    "white":  (245, 245, 245),
    "black":  (35,  35,  35),
    "grey":   (150, 150, 150),
    "gray":   (150, 150, 150),
    "brown":  (140, 85,  45),
    "purple": (130, 55,  180),
    "navy":   (25,  45,  110),
    "maroon": (128, 25,  45),
    "cream":  (240, 225, 185),
    "golden": (200, 160, 40),
    "teal":   (30,  140, 140),
    "indigo": (60,  50,  170),
    "violet": (145, 50,  180),
}

_FORMAL_KEYWORDS = {"formal", "office", "suit", "blazer", "shirt", "trousers"}
_ETHNIC_KEYWORDS = {"kurta", "kurti", "saree", "ethnic", "traditional", "salwar",
                    "lehenga", "anarkali", "dhoti", "sherwani", "pongal",
                    "diwali", "festival"}
_BOTTOM_KEYWORDS = {"jeans", "trousers", "skirt", "shorts", "dhoti", "palazzo",
                    "leggings", "pant", "saree"}
_DRESS_KEYWORDS  = {"dress", "gown", "lehenga", "saree", "anarkali"}


def _parse_prompt(prompt: str) -> dict:
    """Extract color, garment style, and type hint from a user prompt."""
    p = prompt.lower()

    color = (180, 180, 200)          # default: light grey-blue
    for name, rgb in _COLOR_MAP.items():
        if name in p:
            color = rgb
            break

    is_ethnic  = bool(_ETHNIC_KEYWORDS  & set(p.split()))
    is_formal  = bool(_FORMAL_KEYWORDS  & set(p.split()))
    is_bottom  = bool(_BOTTOM_KEYWORDS  & set(p.split()))
    is_dress   = bool(_DRESS_KEYWORDS   & set(p.split()))

    return {
        "color":     color,
        "is_ethnic": is_ethnic,
        "is_formal": is_formal,
        "is_bottom": is_bottom,
        "is_dress":  is_dress,
    }


def _make_garment_image(prompt: str) -> Image.Image:
    """
    Create a simple garment silhouette image from prompt keywords.
    Used when the user types a style prompt instead of selecting a product.
    IDM-VTON needs a real garment image — this PIL sketch acts as a placeholder.
    """
    info  = _parse_prompt(prompt)
    color = info["color"]
    W, H  = 512, 512
    bg    = (255, 255, 255)

    img  = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)

    if info["is_dress"]:
        # Dress / anarkali shape
        draw.polygon(
            [(W//2 - 80, 120), (W//2 + 80, 120),
             (W//2 + 160, H - 60), (W//2 - 160, H - 60)],
            fill=color
        )
        # neckline
        draw.ellipse([W//2 - 35, 100, W//2 + 35, 145], fill=bg)

    elif info["is_bottom"]:
        # Trouser / jeans shape
        draw.rectangle([W//2 - 90, 80, W//2 + 90, 300], fill=color)
        # left leg
        draw.rectangle([W//2 - 90, 280, W//2 - 10, H - 40], fill=color)
        # right leg
        draw.rectangle([W//2 + 10, 280, W//2 + 90, H - 40], fill=color)

    elif info["is_ethnic"]:
        # Kurta / kurti shape (straight cut)
        draw.rectangle([W//2 - 90, 100, W//2 + 90, H - 60], fill=color)
        # short sleeves
        draw.rectangle([W//2 - 130, 100, W//2 - 90, 200], fill=color)
        draw.rectangle([W//2 + 90,  100, W//2 + 130, 200], fill=color)
        # placket line
        draw.line([(W//2, 100), (W//2, 220)], fill=bg, width=4)
        # neckline
        draw.ellipse([W//2 - 35, 80, W//2 + 35, 125], fill=bg)

    elif info["is_formal"]:
        # Formal shirt / blazer shape
        draw.rectangle([W//2 - 95, 100, W//2 + 95, H - 50], fill=color)
        # lapels
        draw.polygon(
            [(W//2, 120), (W//2 - 95, 100), (W//2 - 50, 230)],
            fill=(max(color[0]-30, 0), max(color[1]-30, 0), max(color[2]-30, 0))
        )
        draw.polygon(
            [(W//2, 120), (W//2 + 95, 100), (W//2 + 50, 230)],
            fill=(max(color[0]-30, 0), max(color[1]-30, 0), max(color[2]-30, 0))
        )
        # collar
        draw.ellipse([W//2 - 35, 80, W//2 + 35, 125], fill=bg)

    else:
        # Default: T-shirt shape
        draw.rectangle([W//2 - 90, 140, W//2 + 90, H - 60], fill=color)
        # sleeves
        draw.polygon(
            [(W//2 - 90, 140), (W//2 - 160, 230), (W//2 - 90, 230)],
            fill=color
        )
        draw.polygon(
            [(W//2 + 90, 140), (W//2 + 160, 230), (W//2 + 90, 230)],
            fill=color
        )
        # neckline
        draw.ellipse([W//2 - 40, 110, W//2 + 40, 155], fill=bg)

    return img


# ─────────────────────────────────────────────────────────────────────────────
# Public API  (signatures unchanged — streamlit_app.py needs no change)
# ─────────────────────────────────────────────────────────────────────────────

def is_api_available() -> bool:
    """
    Return True if the IDM-VTON HuggingFace Space is reachable.
    Uses a lightweight HTTP HEAD check — no client initialisation.
    """
    try:
        url = f"https://huggingface.co/api/spaces/{HF_SPACE}"
        r   = requests.get(url, timeout=8)
        if r.status_code == 200:
            stage = (
                r.json()
                 .get("runtime", {})
                 .get("stage", "")
            )
            return stage == "RUNNING"
        return False
    except Exception:
        return False


def virtual_tryon(
    person_image:  Image.Image,
    garment_image: Image.Image,
    garment_name:  str = "clothing item",
    session_id:    str = "user",
) -> dict:
    """
    Apply a garment onto a person photo using IDM-VTON.

    Args:
        person_image:  Full-body or upper-body photo of the user (PIL Image)
        garment_image: Product image (downloaded from Myntra URL) (PIL Image)
        garment_name:  Product name — used as garment description for the model
        session_id:    Used to name the saved output file

    Returns:
        {'success': bool, 'image': PIL|None, 'image_path': str|None, 'error': str|None}
    """
    from gradio_client import handle_file

    person_tmp  = _save_temp(person_image.convert("RGB"),  f"person_{session_id}")
    garment_tmp = _save_temp(garment_image.convert("RGB"), f"garment_{session_id}")

    try:
        client = _get_client()
        print(f"[IDM-VTON] Calling HF Space for: {garment_name[:60]}")

        result = client.predict(
            dict={
                "background": handle_file(person_tmp),
                "layers":     [],
                "composite":  None,
            },
            garm_img     = handle_file(garment_tmp),
            garment_des  = garment_name[:200],
            is_checked   = True,    # auto-mask person
            is_checked_crop = False,
            denoise_steps   = 30,
            seed            = _fresh_seed(),
            api_name        = "/tryon",
        )

        # result is a tuple: (output_image_path, masked_image_path)
        # or sometimes a single path string depending on gradio version
        if isinstance(result, (list, tuple)):
            out_path = result[0]
        else:
            out_path = result
        result_img = Image.open(out_path).convert("RGB")

        save_path = _save_image(result_img, session_id)
        print(f"[IDM-VTON] Saved: {save_path}")

        try:
            from storage.s3_manager import (
                upload_image_file, is_s3_available)
            if is_s3_available():
                s3_url = upload_image_file(
                    save_path, session_id)
                if s3_url:
                    print(f"[S3] Uploaded: {s3_url}")
        except Exception as e:
            print(f"[S3] Skipped: {e}")

        return {
            "success":    True,
            "image":      result_img,
            "image_path": save_path,
            "error":      None,
        }

    except Exception as e:
        error_msg = str(e)
        print(f"[IDM-VTON] Error: {error_msg}")
        if "quota" in error_msg.lower():
            return {
                "success":    False,
                "image":      None,
                "image_path": None,
                "error": (
                    "Daily GPU quota reached. "
                    "Try again tomorrow or "
                    "create a new HF account at "
                    "huggingface.co"
                ),
            }
        return {
            "success":    False,
            "image":      None,
            "image_path": None,
            "error":      error_msg,
        }

    finally:
        _cleanup(person_tmp, garment_tmp)


def virtual_tryon_full_body(
        person_image  : Image.Image,
        garment_image : Image.Image,
        garment_name  : str = "clothing",
        garment_type  : str = "upperbody",
        session_id    : str = "user") -> dict:
    """
    Full body virtual try-on.
    Chain (both confirmed UP as of 2026-04):
      1. OOTDiffusion /process_dc — category-aware
         (Upper-body / Lower-body / Dress)
      2. FitDiT — 2-step mask+process fallback

    garment_type: 'upperbody' | 'lowerbody' | 'dress'
    """
    # ── Try Leffa first (best quality, confirmed UP) ───
    _leffa_map = {
        'upperbody': 'upper_body',
        'lowerbody': 'lower_body',
        'dress'    : 'dresses',
    }
    leffa_type = _leffa_map.get(
        garment_type.lower(), 'upper_body')

    print(
        f"[VTON] Trying Leffa first... "
        f"type={leffa_type}"
    )
    leffa_result = virtual_tryon_leffa(
        person_image  = person_image,
        garment_image = garment_image,
        garment_name  = garment_name,
        garment_type  = leffa_type,
        session_id    = session_id
    )
    if leffa_result['success']:
        print("[VTON] Leffa succeeded!")
        return leffa_result

    # Leffa failed — check reason
    err_msg = str(
        leffa_result.get('error', ''))

    if 'quota' in err_msg.lower():
        import time
        # Retry up to 3 times with increasing
        # wait — Leffa Space queue clears fast
        for attempt in range(1, 4):
            wait = attempt * 10  # 10s, 20s, 30s
            print(f"[VTON] Quota hit — "
                  f"retry {attempt}/3 "
                  f"in {wait}s...")
            time.sleep(wait)
            retry = virtual_tryon_leffa(
                person_image  = person_image,
                garment_image = garment_image,
                garment_name  = garment_name,
                garment_type  = leffa_type,
                session_id    = session_id
            )
            if retry['success']:
                print(f"[VTON] Retry {attempt} "
                      f"succeeded!")
                return retry
            if 'quota' not in str(
                    retry.get('error', '')).lower():
                break  # different error — stop
        # All retries failed
        return {
            'success'   : False,
            'image'     : None,
            'image_path': None,
            'error'     : (
                "⏳ Leffa GPU is currently "
                "busy with other users.\n\n"
                "Please click Try On again "
                "in 1-2 minutes."
            )
        }

    # Non-quota failure — try OOTDiffusion
    # only for upper body (it works better
    # for upper body than lower body)
    if 'lower' in leffa_type or \
       'dress' in leffa_type:
        print("[VTON] Lower body Leffa "
              "non-quota failure — "
              "returning error")
        return {
            'success'   : False,
            'image'     : None,
            'image_path': None,
            'error'     : (
                "Try-on failed. "
                "Please try again."
            )
        }

    # Upper body only — try OOTDiffusion
    print("[VTON] Trying OOTDiffusion "
          "for upper body fallback...")

    from gradio_client import Client, handle_file

    # ── Token ──────────────────────────────────────────
    token = ""
    root  = Path(__file__).resolve().parents[2]
    for ep in [root / '.env', Path('.env')]:
        if ep.exists():
            for line in ep.read_text(
                    encoding='utf-8-sig'
            ).splitlines():
                if line.startswith(
                        "HUGGINGFACE_TOKEN="):
                    token = line.split(
                        "=", 1)[1].strip()
                    break

    # ── Shared image prep ─────────────────────────────
    person_tmp  = _tmp_name("vton_p", session_id)
    garment_tmp = _tmp_name("vton_g", session_id)

    # White background — prevents pose estimation issues
    person_rgb = Image.new(
        'RGB', person_image.size, (255, 255, 255))
    if person_image.mode in ('RGBA', 'LA'):
        person_rgb.paste(
            person_image.convert('RGBA'),
            mask=person_image.convert('RGBA').split()[3]
        )
    else:
        person_rgb = person_image.convert('RGB')

    person_rgb      = person_rgb.resize(
        (384, 512), Image.LANCZOS)
    # Fix transparent PNG garment images —
    # paste on white background before resize
    garment_white = Image.new(
        'RGB', (384, 512), (255, 255, 255))
    if garment_image.mode in ('RGBA', 'LA'):
        g = garment_image.convert(
            'RGBA').resize(
            (384, 512), Image.LANCZOS)
        garment_white.paste(
            g, mask=g.split()[3])
    else:
        g = garment_image.convert(
            'RGB').resize(
            (384, 512), Image.LANCZOS)
        garment_white.paste(g)
    garment_resized = garment_white

    person_rgb.save(person_tmp)
    garment_resized.save(garment_tmp)

    # Category string for /process_dc
    _cat_map = {
        'upperbody': 'Upper-body',
        'lowerbody': 'Lower-body',
        'dress'    : 'Dress',
    }
    category = _cat_map.get(
        garment_type.lower(), 'Upper-body')

    # ── Attempt 1: OOTDiffusion /process_dc ───────────
    try:
        print(
            f"[OOTD] Trying /process_dc "
            f"garment={garment_name} "
            f"category={category}..."
        )
        client = Client(
            "levihsu/OOTDiffusion",
            token=token if token else None,
            verbose=False
        )
        result = client.predict(
            vton_img    = handle_file(person_tmp),
            garm_img    = handle_file(garment_tmp),
            category    = category,
            n_samples   = 1,
            n_steps     = 30,
            image_scale = 2.5,
            seed        = _fresh_seed(),
            api_name    = "/process_dc"
        )
        if isinstance(result, list) and result:
            out_path = result[0].get(
                'image', result[0])
        else:
            out_path = result

        out_img   = Image.open(
            out_path).convert('RGB')
        save_path = _save_image(out_img, session_id)
        print(f"[OOTD] Saved: {save_path}")
        print(f"[VTON] Result from: OOTDiffusion")
        for f in [person_tmp, garment_tmp]:
            try:
                Path(f).unlink(missing_ok=True)
            except Exception:
                pass
        return {
            'success'   : True,
            'image'     : out_img,
            'image_path': save_path,
            'error'     : None
        }

    except Exception as e1:
        print(f"[OOTD] Failed: {str(e1)[:80]}")
        if "quota" in str(e1).lower():
            return {
                'success'   : False,
                'image'     : None,
                'image_path': None,
                'error'     : (
                    "Daily GPU quota reached. "
                    "Try again tomorrow!"
                )
            }

    # ── Attempt 2: FitDiT (2-step) ────────────────────
    try:
        print("[FitDiT] Trying 2-step try-on...")
        _fitdit_cat = {
            'upperbody': 'Upper-body',
            'lowerbody': 'Lower-body',
            'dress'    : 'Dresses',
        }.get(garment_type.lower(), 'Upper-body')

        client2 = Client(
            "BoyuanJiang/FitDiT",
            token=token if token else None,
            verbose=False
        )
        # Step 1: generate mask + pose
        mask_res = client2.predict(
            vton_img  = handle_file(person_tmp),
            category  = _fitdit_cat,
            api_name  = "/generate_mask"
        )
        masked_img = mask_res[0]
        pose_img   = mask_res[1]

        # Step 2: try-on
        fitdit_res = client2.predict(
            vton_img             = handle_file(person_tmp),
            garm_img             = handle_file(garment_tmp),
            pre_mask             = masked_img,
            pose_image           = pose_img,
            n_steps              = 20,
            image_scale          = 2.0,
            seed                 = _fresh_seed(),
            num_images_per_prompt= 1,
            resolution           = "768x1024",
            api_name             = "/process"
        )
        # Result: list of dict(image: dict, caption)
        if isinstance(fitdit_res, list) and fitdit_res:
            img_info = fitdit_res[0]
            if isinstance(img_info, dict):
                img_data = img_info.get('image', img_info)
                if isinstance(img_data, dict):
                    out_path2 = img_data.get('path')
                else:
                    out_path2 = img_data
            else:
                out_path2 = img_info
        else:
            out_path2 = fitdit_res

        out_img2   = Image.open(
            out_path2).convert('RGB')
        save_path2 = _save_image(out_img2, session_id)
        print(f"[FitDiT] Saved: {save_path2}")
        print(f"[VTON] Result from: FitDiT")
        for f in [person_tmp, garment_tmp]:
            try:
                Path(f).unlink(missing_ok=True)
            except Exception:
                pass
        return {
            'success'   : True,
            'image'     : out_img2,
            'image_path': save_path2,
            'error'     : None
        }

    except Exception as e2:
        print(f"[FitDiT] Failed: {str(e2)[:80]}")
        return {
            'success'   : False,
            'image'     : None,
            'image_path': None,
            'error'     : (
                f"Try-on unavailable. "
                f"OOTD: {str(e1)[:40]} | "
                f"FitDiT: {str(e2)[:40]}"
            )
        }


def virtual_tryon_leffa(
        person_image  : Image.Image,
        garment_image : Image.Image,
        garment_name  : str = "clothing",
        garment_type  : str = "upper_body",
        session_id    : str = "user") -> dict:
    """
    Virtual try-on using Leffa model.
    Supports upper_body, lower_body, dresses.
    Free HuggingFace Space: franciszzj/Leffa
    """
    try:
        from gradio_client import Client, handle_file

        token = ""
        root  = Path(__file__).resolve().parents[2]
        for ep in [root / '.env', Path('.env')]:
            if ep.exists():
                for line in ep.read_text(
                        encoding='utf-8-sig'
                ).splitlines():
                    if line.startswith(
                            "HUGGINGFACE_TOKEN="):
                        token = line.split(
                            "=", 1)[1].strip()
                        break

        person_tmp  = _tmp_name("leffa_p", session_id)
        garment_tmp = _tmp_name("leffa_g", session_id)

        print(f"[Leffa] Garment mode: "
              f"{garment_image.mode} "
              f"size: {garment_image.size}")
        print(f"[Leffa] Person mode: "
              f"{person_image.mode} "
              f"size: {person_image.size}")

        # White background fix + 768×1024 for Leffa
        person_white = Image.new(
            'RGB', (768, 1024), (255, 255, 255))
        if person_image.mode in ('RGBA', 'LA'):
            p = person_image.convert('RGBA').resize(
                (768, 1024), Image.LANCZOS)
            person_white.paste(p, mask=p.split()[3])
        else:
            p = person_image.convert('RGB').resize(
                (768, 1024), Image.LANCZOS)
            person_white.paste(p)

        # Fix transparent PNG garment images —
        # paste on white background before resize
        # Same approach used for person_image above
        garment_white = Image.new(
            'RGB', (768, 1024), (255, 255, 255))
        if garment_image.mode in ('RGBA', 'LA'):
            g = garment_image.convert(
                'RGBA').resize(
                (768, 1024), Image.LANCZOS)
            garment_white.paste(
                g, mask=g.split()[3])
        else:
            g = garment_image.convert(
                'RGB').resize(
                (768, 1024), Image.LANCZOS)
            garment_white.paste(g)
        garment_rgb = garment_white

        person_white.save(person_tmp)
        garment_rgb.save(garment_tmp)

        print(
            f"[Leffa] Connecting... "
            f"type={garment_type}"
        )
        client = Client(
            "franciszzj/Leffa",
            token=token if token else None,
            verbose=False
        )

        # Tune params per garment type
        if garment_type == "lower_body":
            model_type = "dress_code"
            repaint    = True
            scale_val  = 4.0
            step_val   = 30
        elif garment_type == "dresses":
            model_type = "dress_code"
            repaint    = True
            scale_val  = 5.0
            step_val   = 30    # reduced from 40 to preserve GPU quota
        else:  # upper_body
            model_type = "viton_hd"
            repaint    = False
            scale_val  = 2.5
            step_val   = 30

        print(
            f"[Leffa] Running: {garment_name} | "
            f"model={model_type} repaint={repaint} "
            f"scale={scale_val} steps={step_val}"
        )
        result = client.predict(
            src_image_path   = handle_file(person_tmp),
            ref_image_path   = handle_file(garment_tmp),
            ref_acceleration = False,
            step             = step_val,
            scale            = scale_val,
            seed             = _fresh_seed(),
            vt_model_type    = model_type,
            vt_garment_type  = garment_type,
            vt_repaint       = repaint,
            api_name         = "/leffa_predict_vt"
        )

        # Returns tuple: (generated_image, mask, densepose)
        # Each is dict with 'path' key
        if isinstance(result, (list, tuple)):
            out_dict = result[0]
        else:
            out_dict = result

        if isinstance(out_dict, dict):
            out_path = (out_dict.get('path') or
                        out_dict.get('url'))
        else:
            out_path = out_dict

        out_img   = Image.open(
            out_path).convert('RGB')
        save_path = _save_image(out_img, session_id)
        print(f"[Leffa] Saved: {save_path}")

        for f in [person_tmp, garment_tmp]:
            try:
                Path(f).unlink(missing_ok=True)
            except Exception:
                pass

        return {
            'success'   : True,
            'image'     : out_img,
            'image_path': save_path,
            'error'     : None
        }

    except Exception as e:
        err = str(e)
        print(f"[Leffa] Error: {err}")
        return {
            'success'   : False,
            'image'     : None,
            'image_path': None,
            'error'     : (
                "Daily GPU quota reached. "
                "Try again tomorrow!"
                if "quota" in err.lower()
                else err
            )
        }


def virtual_tryon_stable(
        person_image  : Image.Image,
        garment_image : Image.Image,
        garment_name  : str = "clothing",
        session_id    : str = "user") -> dict:
    """
    StableVITON-HD virtual try-on.
    Better quality than OOTDiffusion for lower body.
    Free HuggingFace Space.
    """
    try:
        from gradio_client import Client, handle_file

        token = ""
        root  = Path(__file__).resolve().parents[2]
        for ep in [root / '.env', Path('.env')]:
            if ep.exists():
                for line in ep.read_text(
                        encoding='utf-8-sig'
                ).splitlines():
                    if line.startswith(
                            "HUGGINGFACE_TOKEN="):
                        token = line.split(
                            "=", 1)[1].strip()
                        break

        person_tmp  = _tmp_name("stable_p", session_id)
        garment_tmp = _tmp_name("stable_g", session_id)

        # White background fix + 768×1024 for StableVITON
        person_rgb = Image.new(
            'RGB', (768, 1024), (255, 255, 255))
        if person_image.mode in ('RGBA', 'LA'):
            person_rgb.paste(
                person_image.convert('RGBA'),
                mask=person_image.convert(
                    'RGBA').split()[3]
            )
        else:
            p = person_image.convert('RGB').resize(
                (768, 1024), Image.LANCZOS)
            person_rgb.paste(p)

        # Fix transparent PNG garment images —
        # paste on white background before resize
        garment_white = Image.new(
            'RGB', (768, 1024), (255, 255, 255))
        if garment_image.mode in ('RGBA', 'LA'):
            g = garment_image.convert(
                'RGBA').resize(
                (768, 1024), Image.LANCZOS)
            garment_white.paste(
                g, mask=g.split()[3])
        else:
            g = garment_image.convert(
                'RGB').resize(
                (768, 1024), Image.LANCZOS)
            garment_white.paste(g)
        garment_rgb = garment_white

        person_rgb.save(person_tmp)
        garment_rgb.save(garment_tmp)

        print("[StableVITON] Connecting...")
        client = Client(
            "rlawjdghek/StableVITON-HD",
            token=token if token else None,
            verbose=False
        )

        print(f"[StableVITON] Running: {garment_name}")
        result = client.predict(
            img      = handle_file(person_tmp),
            garm_img = handle_file(garment_tmp),
            n_steps  = 20,
            api_name = "/process"
        )

        if isinstance(result, (list, tuple)):
            out_path = result[0]
        else:
            out_path = result

        out_img   = Image.open(
            out_path).convert('RGB')
        save_path = _save_image(out_img, session_id)
        print(f"[StableVITON] Saved: {save_path}")

        for f in [person_tmp, garment_tmp]:
            try:
                Path(f).unlink(missing_ok=True)
            except Exception:
                pass

        return {
            'success'   : True,
            'image'     : out_img,
            'image_path': save_path,
            'error'     : None
        }

    except Exception as e:
        print(f"[StableVITON] Error: {e}")
        return {
            'success'   : False,
            'image'     : None,
            'image_path': None,
            'error'     : str(e)
        }


def generate_outfit_from_prompt(
    user_image:  Image.Image,
    user_prompt: str,
    session_id:  str = "user",
) -> dict:
    """
    Generate an outfit on the user's photo from a text style prompt.

    Steps:
      1. Parse color + garment type from the prompt
      2. Draw a garment silhouette with PIL
      3. Feed into virtual_tryon()

    Args:
        user_image:  User-uploaded photo (PIL Image)
        user_prompt: e.g. "blue kurta for Pongal festival"
        session_id:  Used to name the saved output file
    """
    print(f"[GENERATE] Prompt: {user_prompt[:80]}")
    garment_img = _make_garment_image(user_prompt)
    return virtual_tryon(
        person_image  = user_image,
        garment_image = garment_img,
        garment_name  = user_prompt[:200],
        session_id    = session_id,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Smoke-test:  python src/outfit_generation/outfit_generator.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("MYNA - outfit_generator.py smoke test")
    print("=" * 55)

    # 1. API availability
    print("\n[1] Checking HF Space availability...")
    ok = is_api_available()
    print(f"    is_api_available() = {ok}")

    # 2. Garment generation (no network needed)
    print("\n[2] Generating garment images from prompts...")
    prompts = [
        "blue kurta for Pongal festival",
        "red formal shirt for office",
        "black jeans for college",
        "pink party dress",
        "white t-shirt casual",
    ]
    for p in prompts:
        img = _make_garment_image(p)
        out = OUTPUT_DIR / f"test_garment_{p[:20].replace(' ', '_')}.png"
        img.save(out)
        print(f"    OK  {p[:45]:<45}  saved: {out.name}")

    # 3. Virtual try-on connectivity check
    if ok:
        print("\n[3] HF Space is RUNNING - virtual_tryon() is ready.")
        print("    NOTE: Try-on requires a real person photo.")
        print("    Synthetic plain images will fail with IndexError (expected).")
        print("    In the Streamlit app, real user photos work correctly.")
    else:
        print("\n[3] HF Space not RUNNING - try-on will be unavailable.")
        print("    Open in browser to wake the space:")
        print(f"    https://huggingface.co/spaces/{HF_SPACE}")

    print("\nDone.")
