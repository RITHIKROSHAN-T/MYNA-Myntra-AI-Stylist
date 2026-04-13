# s3_manager.py
# Upload generated outfit images to AWS S3.
#
# is_s3_available()    – check credentials present
# upload_pil_image()   – PIL Image → S3 → public URL
# upload_image_file()  – local file path → S3 → public URL

import io
from pathlib import Path


# ── Credential helper (reads .env directly, no dotenv cache issues) ───────────

def _get_env(key: str) -> str:
    root = Path(__file__).resolve().parents[2]
    for ep in [root / ".env", Path(".env")]:
        if ep.exists():
            for line in ep.read_text(encoding="utf-8-sig").splitlines():
                if line.startswith(f"{key}="):
                    return line.split("=", 1)[1].strip()
    return ""


def _boto_client():
    import boto3
    return boto3.client(
        "s3",
        aws_access_key_id     = _get_env("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key = _get_env("AWS_SECRET_ACCESS_KEY"),
        region_name           = _get_env("AWS_REGION") or "ap-south-1",
    )


# ── Public API ────────────────────────────────────────────────────────────────

def is_s3_available() -> bool:
    """Return True if AWS credentials are present in .env."""
    return bool(
        _get_env("AWS_ACCESS_KEY_ID") and
        _get_env("AWS_SECRET_ACCESS_KEY")
    )


def upload_pil_image(
    image,
    filename: str,
    session_id: str = "default",
) -> str:
    """
    Upload a PIL Image to S3.

    Args:
        image:      PIL.Image.Image
        filename:   e.g. "outfit_abc_20240101.png"
        session_id: used to namespace the S3 key

    Returns:
        Public HTTPS URL, or "" on any failure.
    """
    try:
        bucket = _get_env("S3_BUCKET_NAME")
        region = _get_env("AWS_REGION") or "ap-south-1"
        if not bucket:
            print("[S3] S3_BUCKET_NAME not set in .env")
            return ""

        key = f"generated-outfits/{session_id}/{filename}"

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)

        s3 = _boto_client()
        s3.upload_fileobj(
            buf,
            bucket,
            key,
            ExtraArgs={"ContentType": "image/png"},
        )

        url = f"https://{bucket}.s3.{region}.amazonaws.com/{key}"
        return url

    except Exception as e:
        print(f"[S3] upload_pil_image failed: {e}")
        return ""


def upload_image_file(
    local_path: str,
    session_id: str = "default",
) -> str:
    """
    Upload a local image file to S3.

    Args:
        local_path: absolute or relative path to the file
        session_id: used to namespace the S3 key

    Returns:
        Public HTTPS URL, or "" on any failure.
    """
    try:
        bucket = _get_env("S3_BUCKET_NAME")
        region = _get_env("AWS_REGION") or "ap-south-1"
        if not bucket:
            print("[S3] S3_BUCKET_NAME not set in .env")
            return ""

        path = Path(local_path)
        if not path.exists():
            print(f"[S3] File not found: {local_path}")
            return ""

        key = f"generated-outfits/{session_id}/{path.name}"

        s3 = _boto_client()
        s3.upload_file(
            str(path),
            bucket,
            key,
            ExtraArgs={"ContentType": "image/png"},
        )

        url = f"https://{bucket}.s3.{region}.amazonaws.com/{key}"
        return url

    except Exception as e:
        print(f"[S3] upload_image_file failed: {e}")
        return ""


# ── Smoke-test: python src/storage/s3_manager.py ─────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("MYNA - s3_manager.py smoke test")
    print("=" * 55)

    avail = is_s3_available()
    print(f"\n[1] Credentials present : {avail}")
    if avail:
        key_preview = _get_env("AWS_ACCESS_KEY_ID")[:8]
        bucket      = _get_env("S3_BUCKET_NAME")
        region      = _get_env("AWS_REGION")
        print(f"    Access key  : {key_preview}...")
        print(f"    Bucket      : {bucket or '(not set)'}")
        print(f"    Region      : {region or '(not set)'}")
    else:
        print("    AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY not in .env")
        print("    S3 upload will be skipped (non-fatal).")

    if avail and _get_env("S3_BUCKET_NAME"):
        print("\n[2] Attempting test upload...")
        from PIL import Image
        test_img = Image.new("RGB", (64, 64), (255, 100, 100))
        url = upload_pil_image(test_img, "test_upload.png", "smoke-test")
        if url:
            print(f"    Upload OK: {url}")
        else:
            print("    Upload returned empty URL (check bucket/perms).")
    else:
        print("\n[2] Skipping upload test (no bucket configured).")

    print("\nDone.")
