# supabase_storage.py
import os
from typing import Optional
from supabase import create_client, Client

_SUPABASE_URL = os.environ["SUPABASE_URL"]
_SERVICE_ROLE = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
_BUCKET = os.environ.get("SUPABASE_BUCKET", "privatedocs")

_client: Optional[Client] = None


def _sb() -> Client:
    global _client
    if _client is None:
        _client = create_client(_SUPABASE_URL, _SERVICE_ROLE)
    return _client


def upload_file(local_path: str, dest_path: str, upsert: bool = True) -> str:
    """
    Lädt local_path in den Bucket unter dest_path (z. B. 'wunschoele/2025-06/leitfaden.pdf')
    und gibt den Storage-Pfad zurück (bucket/key).
    """
    with open(local_path, "rb") as f:
        _sb().storage.from_(_BUCKET).upload(dest_path, f, {"upsert": upsert})
    return f"{_BUCKET}/{dest_path}"


def create_signed_url(storage_path: str, expires_sec: int = 3600) -> str:
    """
    storage_path Format: '<bucket>/<key>'
    """
    bucket, key = storage_path.split("/", 1)
    res = _sb().storage.from_(bucket).create_signed_url(key, expires_sec)
    return res["signedURL"]


def exists(storage_path: str) -> bool:
    bucket, key = storage_path.split("/", 1)
    try:
        _sb().storage.from_(bucket).list(
            path="/".join(key.split("/")[:-1]), search=key.split("/")[-1]
        )
        return True
    except Exception:
        return False
