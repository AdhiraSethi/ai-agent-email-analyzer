"""
VoxIntel — FastAPI Backend
"""

import os, time, uuid, json, logging
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

import agent

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("voxintel")


# ── In-memory stores (swap for Redis/Postgres in production) ──────────────────
memory: dict[str, list] = defaultdict(list)   # customer_id → interactions
stats  = defaultdict(int)
stats["confidence_sum"] = 0.0


# ── Request / response models ─────────────────────────────────────────────────
class EmailIn(BaseModel):
    subject:       str
    body:          str
    sender:        Optional[str]  = ""
    customer_name: Optional[str]  = "Customer"
    customer_id:   Optional[str]  = ""
    crm:           Optional[dict] = None  # lifetime_value, complaint_count, is_vip, etc.

class CorrectionIn(BaseModel):
    email_id:     str
    ai_reply:     str
    human_reply:  str


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.getenv("GROQ_API_KEY"):
        log.error("GROQ_API_KEY not set!")
    else:
        log.info("VoxIntel ready — model: %s", os.getenv("MODEL", "llama-3.3-70b-versatile"))
    yield


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="VoxIntel", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_history(customer_id: str) -> str:
    interactions = memory.get(customer_id, [])[-5:]
    if not interactions:
        return ""
    return "\n".join(f"- [{i['time']}] {i['summary']} (resolved: {i['resolved']})"
                     for i in interactions)

def save_interaction(customer_id: str, summary: str, resolved: bool):
    if customer_id:
        memory[customer_id].append({
            "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
            "summary": summary,
            "resolved": resolved,
        })

def track_stats(result: dict):
    stats["total"] += 1
    stats["confidence_sum"] += result["reply"]["confidence"]
    stats[result["decision"]] += 1
    stats[f"intent_{result['analysis']['intent']}"] += 1


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": os.getenv("MODEL", "llama-3.3-70b-versatile")}


@app.post("/process")
def process(req: EmailIn):
    """Full pipeline: analyze → retrieve → reply → route."""
    start = time.perf_counter()
    email_id = str(uuid.uuid4())[:8]
    log.info("Processing %s | subject='%s'", email_id, req.subject)

    try:
        history = get_history(req.customer_id or "")
        result  = agent.run(
            subject=req.subject,
            body=req.body,
            sender=req.sender or "",
            customer_name=req.customer_name or "Customer",
            customer_id=req.customer_id or "",
            crm=req.crm,
            history=history,
        )
    except Exception as e:
        log.exception("Pipeline failed for %s", email_id)
        raise HTTPException(status_code=500, detail=str(e))

    save_interaction(
        req.customer_id or "",
        result["analysis"]["summary"],
        resolved=(result["decision"] == "AUTO_SEND"),
    )
    track_stats(result)
    elapsed = round((time.perf_counter() - start) * 1000, 1)

    log.info("%s → %s (%.0f%%) in %sms",
             email_id, result["decision"],
             result["reply"]["confidence"] * 100, elapsed)

    return {"email_id": email_id, "processing_ms": elapsed, **result}


@app.post("/analyze")
def analyze_only(req: EmailIn):
    """Analysis only — no reply generation."""
    start = time.perf_counter()
    try:
        result = agent.analyze(req.subject, req.body, req.crm)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"processing_ms": round((time.perf_counter() - start) * 1000, 1), **result}


@app.post("/correction")
def log_correction(req: CorrectionIn):
    """Step 12: Save human-edited reply for future fine-tuning."""
    record = {
        "email_id": req.email_id,
        "timestamp": datetime.utcnow().isoformat(),
        "ai_reply": req.ai_reply,
        "human_reply": req.human_reply,
        "changed": req.ai_reply.strip() != req.human_reply.strip(),
    }
    with open("corrections.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
    return {"logged": True, "changed": record["changed"]}


@app.get("/memory/{customer_id}")
def get_customer_memory(customer_id: str):
    return {"customer_id": customer_id, "interactions": memory.get(customer_id, [])}


@app.get("/stats")
def get_stats():
    total = stats["total"] or 1
    return {
        "total":            stats["total"],
        "auto_sent":        stats["AUTO_SEND"],
        "manager_approval": stats["MANAGER_APPROVAL"],
        "human_agent":      stats["HUMAN_AGENT"],
        "escalated":        stats["ESCALATE"],
        "avg_confidence":   round(stats["confidence_sum"] / total, 3),
    }
