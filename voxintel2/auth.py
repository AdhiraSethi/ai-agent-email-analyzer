"""
auth.py — API key protection. Set API_SECRET_KEY in .env to enable.
Leave API_SECRET_KEY blank to disable auth (dev mode).
"""

import os
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

API_KEY        = os.getenv("API_SECRET_KEY", "")
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    if not API_KEY:
        return True   # auth disabled in dev
    if not api_key or api_key != API_KEY:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = "Invalid or missing API key. Send it as X-API-Key header."
        )
    return True