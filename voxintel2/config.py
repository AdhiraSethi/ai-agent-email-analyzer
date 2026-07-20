"""
config.py — All settings in one place.
Every other file imports from here instead of calling os.getenv() directly.
"""

import os
from dotenv import load_dotenv
load_dotenv()

# ── Groq ──────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL        = os.getenv("MODEL",       "llama-3.3-70b-versatile")
MODEL_FAST   = os.getenv("MODEL_FAST",  "llama-3.1-8b-instant")

# ── Email ──────────────────────────────────────────────────────────────────────
EMAIL_USER    = os.getenv("EMAIL_USER",      "")
EMAIL_PASS    = os.getenv("EMAIL_PASS",      "")
IMAP_HOST     = os.getenv("EMAIL_IMAP_HOST", "imap.gmail.com")
SMTP_HOST     = os.getenv("EMAIL_SMTP_HOST", "smtp.gmail.com")
SMTP_PORT     = int(os.getenv("EMAIL_SMTP_PORT", 465))
MANAGER_EMAIL = os.getenv("MANAGER_EMAIL",  "")
AUTO_SEND     = os.getenv("AUTO_SEND_EMAIL", "false").lower() == "true"

# ── Thresholds ─────────────────────────────────────────────────────────────────
AUTO_SEND_THRESHOLD = float(os.getenv("AUTO_SEND_THRESHOLD",        0.90))
MANAGER_THRESHOLD   = float(os.getenv("MANAGER_APPROVAL_THRESHOLD", 0.70))
POLL_MINUTES        = int(os.getenv("POLL_INTERVAL_MINUTES", 5))

# ── Database ───────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///voxintel.db")

# ── App ────────────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")