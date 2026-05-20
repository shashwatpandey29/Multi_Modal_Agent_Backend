import os
import json
import hashlib
from typing import Optional

import redis

REDIS_URL = os.getenv("CACHE_URL", "redis://localhost:6379/0")
CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", "86400"))  # default 24h

_client = redis.from_url(REDIS_URL, decode_responses=True)


def make_key(*parts) -> str:
    return ":".join(str(p) for p in parts)


def get(key: str) -> Optional[dict]:
    v = _client.get(key)
    if not v:
        return None
    try:
        return json.loads(v)
    except Exception:
        return None


def set(key: str, value, ttl: int = CACHE_TTL):
    _client.set(key, json.dumps(value), ex=ttl)


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
