"""
VoxIntel — FastAPI Backend (Clean version)
"""

import time, uuid, json, logging
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime

import config

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

import agent
import email_connector

logging.basicConfig(
    level=config.LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("voxintel")

# ── In-memory stores ──────────────────────────────────────────────────────────
memory: dict[str, list] = defaultdict(list)
stats  = defaultdict(int)
stats["confidence_sum"] = 0.0


# ── Request models ────────────────────────────────────────────────────────────
class EmailIn(BaseModel):
    subject:       str
    body:          str
    sender:        Optional[str]  = ""
    customer_name: Optional[str]  = "Customer"
    customer_id:   Optional[str]  = ""
    crm:           Optional[dict] = None
    manager_email: Optional[str]  = None

class RawEmailIn(BaseModel):
    raw_email:     str
    customer_id:   Optional[str]  = ""
    crm:           Optional[dict] = None
    manager_email: Optional[str]  = None

class CorrectionIn(BaseModel):
    email_id:    str
    ai_reply:    str
    human_reply: str


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not config.GROQ_API_KEY:
        log.error("GROQ_API_KEY not set in .env!")
    else:
        log.info("VoxIntel ready — model: %s", config.MODEL)

    if config.EMAIL_USER and config.EMAIL_PASS:
        from scheduler import start_scheduler, stop_scheduler
        start_scheduler()
        yield
        stop_scheduler()
    else:
        log.info("No email credentials — manual mode only (use /process-raw)")
        yield


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "VoxIntel — AI Email Intelligence Agent",
    description = "Reads emails, detects intent/emotion/priority, drafts reply, routes decision.",
    version     = "3.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ── Global error handler ──────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.exception("Unhandled error on %s", request.url)
    return JSONResponse(
        status_code = 500,
        content     = {
            "error":  type(exc).__name__,
            "detail": str(exc),
            "path":   str(request.url),
        }
    )


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_history(customer_id: str) -> str:
    interactions = memory.get(customer_id, [])[-5:]
    if not interactions:
        return ""
    return "\n".join(
        f"- [{i['time']}] {i['summary']} (resolved: {i['resolved']})"
        for i in interactions
    )


def save_interaction(customer_id: str, summary: str, resolved: bool,
                     email_id: str = "", result: dict = None):
    if not customer_id:
        return
    entry = {
        "email_id": email_id,
        "time":     datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
        "summary":  summary,
        "resolved": resolved,
    }
    if result:
        entry["intent"]         = result["analysis"].get("intent", "")
        entry["emotion"]        = result["analysis"].get("emotion", "")
        entry["priority"]       = result["analysis"].get("priority", "")
        entry["decision"]       = result["decision"]
        entry["confidence"]     = result["reply"]["confidence"]
        entry["outgoing_email"] = result.get("outgoing_email", {})
        entry["manager_alert"]  = result.get("manager_alert")
    memory[customer_id].append(entry)


def track_stats(result: dict):
    stats["total"]            += 1
    stats["confidence_sum"]   += result["reply"]["confidence"]
    stats[result["decision"]] += 1
    stats[f"intent_{result['analysis']['intent']}"] += 1


def _send_emails_if_enabled(result: dict):
    """Send auto-reply and/or manager alert if AUTO_SEND is on."""
    if not config.AUTO_SEND:
        return
    out = result.get("outgoing_email", {})
    if result["decision"] == "AUTO_SEND" and out.get("to"):
        email_connector.send_email(out["to"], out["subject"], out["body"])
        log.info("Auto-reply sent to %s", out["to"])
    alert = result.get("manager_alert")
    if alert and alert.get("to"):
        email_connector.send_email(alert["to"], alert["subject"], alert["body"])
        log.info("Manager alert sent to %s", alert["to"])


def _run_pipeline(subject: str, body: str, sender: str, customer_name: str,
                  customer_id: str, crm: dict, manager_email: str) -> tuple:
    """Shared pipeline logic used by /process and /process-raw."""

    # Input validation
    if not body or len(body.strip()) < 5:
        raise ValueError("Email body is too short to analyze")
    body = body[:10000]   # cap very long emails

    email_id = str(uuid.uuid4())[:8]
    start    = time.perf_counter()
    history  = get_history(customer_id or sender or "")

    log.info("Processing email_id=%s subject='%s'", email_id, subject)

    result = agent.run(
        subject       = subject,
        body          = body,
        sender        = sender,
        customer_name = customer_name,
        customer_id   = customer_id,
        crm           = crm,
        history       = history,
        manager_email = manager_email or config.MANAGER_EMAIL,
    )

    elapsed = round((time.perf_counter() - start) * 1000, 1)

    save_interaction(
        customer_id or sender,
        result["analysis"]["summary"],
        resolved  = (result["decision"] == "AUTO_SEND"),
        email_id  = email_id,
        result    = result,
    )
    track_stats(result)
    _send_emails_if_enabled(result)

    log.info("email_id=%s → %s (%.0f%%) in %sms",
             email_id, result["decision"],
             result["reply"]["confidence"] * 100, elapsed)

    return email_id, result, elapsed


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    return {
        "status":        "ok",
        "model":         config.MODEL,
        "auto_send":     config.AUTO_SEND,
        "email_enabled": bool(config.EMAIL_USER),
    }


@app.post("/process", tags=["Pipeline"])
def process(req: EmailIn):
    """Process a structured JSON email through the full pipeline."""
    email_id, result, elapsed = _run_pipeline(
        req.subject, req.body,
        req.sender        or "",
        req.customer_name or "Customer",
        req.customer_id   or "",
        req.crm, req.manager_email,
    )
    return {"email_id": email_id, "processing_ms": elapsed, **result}


@app.post("/process-raw")
def process_raw(req: RawEmailIn):
    ...
    start = time.perf_counter()
    ...
    elapsed = round((time.perf_counter() - start) * 1000, 1)
    return {
        "email_id":      email_id,
        "parsed":        parsed,
        "processing_ms": elapsed,   # ← NameError, elapsed never computed
        **result
    }

@app.post("/analyze", tags=["Pipeline"])
def analyze_only(req: EmailIn):
    """Analysis only — returns signals without generating a reply."""
    start = time.perf_counter()
    result = agent.analyze(req.subject, req.body, req.crm)
    return {"processing_ms": round((time.perf_counter() - start) * 1000, 1), **result}


@app.get("/emails", tags=["Data"])
def list_emails():
    """List all processed emails stored in memory."""
    all_emails = []
    for customer_id, interactions in memory.items():
        for entry in interactions:
            all_emails.append({
                "email_id":    entry.get("email_id", ""),
                "customer_id": customer_id,
                "time":        entry.get("time", ""),
                "summary":     entry.get("summary", ""),
                "intent":      entry.get("intent", ""),
                "emotion":     entry.get("emotion", ""),
                "priority":    entry.get("priority", ""),
                "decision":    entry.get("decision", ""),
                "confidence":  entry.get("confidence", 0),
            })
    return {"total": len(all_emails), "emails": all_emails}


@app.get("/emails/{email_id}", tags=["Data"])
def get_email(email_id: str):
    """Get full details of a specific processed email including drafted reply."""
    for customer_id, interactions in memory.items():
        for entry in interactions:
            if entry.get("email_id") == email_id:
                return {"customer_id": customer_id, **entry}
    raise HTTPException(status_code=404, detail=f"Email {email_id} not found")


@app.get("/memory/{customer_id}", tags=["Data"])
def get_memory(customer_id: str):
    """Get conversation history for a specific customer."""
    return {
        "customer_id":  customer_id,
        "interactions": memory.get(customer_id, []),
        "count":        len(memory.get(customer_id, [])),
    }


@app.post("/correction", tags=["Learning"])
def log_correction(req: CorrectionIn):
    """Save human-edited reply for future fine-tuning (Step 12)."""
    record = {
        "email_id":    req.email_id,
        "timestamp":   datetime.utcnow().isoformat(),
        "ai_reply":    req.ai_reply,
        "human_reply": req.human_reply,
        "changed":     req.ai_reply.strip() != req.human_reply.strip(),
    }
    with open("corrections.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
    return {"logged": True, "changed": record["changed"]}


@app.get("/stats", tags=["System"])
def get_stats():
    """Dashboard statistics."""
    total = stats["total"] or 1
    return {
        "total_processed":  stats["total"],
        "auto_sent":        stats["AUTO_SEND"],
        "manager_approval": stats["MANAGER_APPROVAL"],
        "human_agent":      stats["HUMAN_AGENT"],
        "escalated":        stats["ESCALATE"],
        "avg_confidence":   round(stats["confidence_sum"] / total, 3),
    }