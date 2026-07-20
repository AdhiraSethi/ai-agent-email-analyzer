"""
preprocess.py — Email Preprocessing stage (the box between "Email Connector"
and "VoxIntel AI Engine" in the architecture diagram).

Does the cheap, deterministic work BEFORE the email ever reaches the LLM:
  • Remove HTML          → plain-text body from html-only emails
  • Remove Signature     → strip sign-offs so they don't pollute the prompt
  • Detect Language      → real langdetect call, not an LLM guess
  • Spam Detection       → rule-based pre-filter (cheap, runs before any LLM call)

OCR Attachments is intentionally left as a documented stub — wiring a real OCR
engine (e.g. pytesseract) needs binary system deps outside this project's
scope, but the hook is here so it's obvious where it plugs in.
"""

import re
import logging
from bs4 import BeautifulSoup
from langdetect import detect, DetectorFactory, LangDetectException

DetectorFactory.seed = 0  # deterministic langdetect output
log = logging.getLogger("voxintel.preprocess")

# ── Signature stripping ────────────────────────────────────────────────────
SIGNATURE_MARKERS = [
    r"\n--\s*\n",                      # standard "-- " signature delimiter
    r"\nregards,?\s*\n",
    r"\nthanks,?\s*\n",
    r"\nthank you,?\s*\n",
    r"\nbest,?\s*\n",
    r"\nbest regards,?\s*\n",
    r"\nsincerely,?\s*\n",
    r"\nsent from my iphone",
    r"\nsent from my android",
    r"\nget outlook for",
]

# ── Spam heuristics ─────────────────────────────────────────────────────────
SPAM_KEYWORDS = [
    "click here", "you have won", "lottery", "viagra", "act now",
    "limited time offer", "wire transfer", "nigerian prince",
    "congratulations you", "claim your prize", "100% free", "no cost to you",
    "urgent reply needed", "crypto investment", "guaranteed income",
]


def strip_html(raw_html: str) -> str:
    """Convert an HTML-only email body into plain text."""
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    return text


def remove_signature(text: str) -> str:
    """Cut the body off at the first recognizable sign-off."""
    lowered = text.lower()
    cut_at = len(text)
    for pattern in SIGNATURE_MARKERS:
        m = re.search(pattern, lowered)
        if m:
            cut_at = min(cut_at, m.start())
    return text[:cut_at].strip() or text.strip()


def detect_language(text: str) -> str:
    """Return an ISO-639-1 language code, defaulting to 'en'."""
    sample = text.strip()
    if len(sample) < 3:
        return "en"
    try:
        return detect(sample)
    except LangDetectException:
        return "en"


def is_spam(subject: str, body: str) -> tuple[bool, float]:
    """
    Cheap rule-based spam pre-filter — runs BEFORE any LLM call so obvious
    spam never costs a Groq API call. Returns (is_spam, score 0.0-1.0).
    """
    text = f"{subject} {body}".lower()
    hits = sum(1 for kw in SPAM_KEYWORDS if kw in text)

    score = min(hits * 0.25, 1.0)

    # ALL-CAPS subject with an exclamation point is a classic spam signal
    if subject and subject.isupper() and len(subject) > 8:
        score = min(score + 0.2, 1.0)

    # Excessive exclamation marks
    if body.count("!") >= 5:
        score = min(score + 0.15, 1.0)

    return score >= 0.5, round(score, 2)


def extract_attachment_text(attachments: list) -> str:
    """
    Stub for OCR Attachments (diagram step). Wire in pytesseract / a cloud
    OCR API here. For now, returns empty string and logs what was skipped
    so it's visible in ops rather than silently dropped.
    """
    if attachments:
        log.info("Skipping OCR for %d attachment(s) — OCR not configured", len(attachments))
    return ""


def preprocess_email(raw: dict) -> dict:
    """
    Main entry point. Takes the raw dict from email_connector.fetch_unread_emails()
    (or a manually-constructed one) and returns it enriched with cleaned fields.

    Expected input keys: subject, body, body_html (optional), attachments (optional)
    Added output keys:   body (cleaned), language, is_spam, spam_score
    """
    subject = raw.get("subject", "") or ""
    body    = raw.get("body", "") or ""

    # Fall back to HTML body if no plain-text part was found
    if not body.strip() and raw.get("body_html"):
        body = strip_html(raw["body_html"])

    body = remove_signature(body)

    attachment_text = extract_attachment_text(raw.get("attachments", []))
    if attachment_text:
        body = f"{body}\n\n[Attachment content]\n{attachment_text}"

    spam, spam_score = is_spam(subject, body)

    out = dict(raw)
    out["body"]       = body
    out["language"]   = detect_language(body or subject)
    out["is_spam"]    = spam
    out["spam_score"] = spam_score
    return out