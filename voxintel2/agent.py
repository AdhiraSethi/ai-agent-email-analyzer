"""
VoxIntel Agent — Full 12-step email intelligence pipeline.
One file, two Groq calls per email: analyze → reply.
"""

import os, re, json, logging
from groq import Groq
import config
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

log    = logging.getLogger("voxintel.agent")
client = Groq(api_key=config.GROQ_API_KEY)
MODEL      = config.MODEL
MODEL_FAST = config.MODEL_FAST

# ── Knowledge base (Step 7) ───────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ── Knowledge base documents ──────────────────────────────────────────────────
# Add as many policies as you want here — FAISS handles scale
KB_DOCUMENTS = [
    "Password reset: go to Settings → Security → Reset Password. OTP valid 10 min. If not received, retry after 2 min. Contact Tech Support if issue persists.",
    "Refund policy: eligible within 30 days of purchase. Processed in 5–7 business days. Digital products non-refundable after download. Refund requires order ID.",
    "Fraud policy: account blocked within 15 min of report. Security team investigates and sends case ID within 1 hour. Never share OTP or password with anyone.",
    "Cancellation: go to Account → Subscription → Cancel. Annual plans get pro-rated refund. Service remains active until end of current billing period.",
    "Invoice policy: invoices auto-sent within 24 hrs of payment. Download duplicates from Account → Billing. GST invoices available for Indian customers.",
    "Payment failure: verify card details or try another payment method. Share transaction ID with billing support for manual verification. Retry after 30 minutes.",
    "Complaint escalation: all complaints acknowledged within 2 hours. Unresolved complaints auto-escalated to senior support after 24 hours.",
    "Subscription plans: monthly and annual plans available. Annual plan saves 20%. Plan details at account.voxintel.ai/plans.",
    "Technical support: available 24/7. For app issues clear cache first. For website issues try incognito mode. Raise ticket at support.voxintel.ai.",
    "Legal concerns: all legal notices to be sent to legal@voxintel.ai. Our legal team responds within 48 business hours.",
    "Partnership inquiries: send company profile and proposal to partners@voxintel.ai. Business development team responds within 5 business days.",
    "Sales and pricing: enterprise pricing available for 50+ seats. Contact sales@voxintel.ai for custom quotes and demos.",
]

# ── Build FAISS index at startup ──────────────────────────────────────────────
log.info("Loading sentence transformer model...")
_embedder = SentenceTransformer("all-MiniLM-L6-v2")  # fast, 80MB, very good quality

def _build_index():
    embeddings = _embedder.encode(KB_DOCUMENTS, convert_to_numpy=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # normalize
    index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner Product = cosine similarity
    index.add(embeddings)
    return index

_faiss_index = _build_index()
log.info("FAISS index built with %d documents", len(KB_DOCUMENTS))


# ── Step 7: Semantic retrieval ────────────────────────────────────────────────
def retrieve(intent: str, body: str, top_k: int = 2) -> str:
    """
    Converts query to embedding, finds top_k most similar KB documents.
    Returns them combined as context for reply generation.
    Much better than keyword matching — understands meaning, not just words.
    """
    query = f"{intent}. {body[:300]}"
    query_embedding = _embedder.encode([query], convert_to_numpy=True)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    scores, indices = _faiss_index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if score > 0.3:   # threshold — ignore low-similarity matches
            results.append(KB_DOCUMENTS[idx])

    if not results:
        return "No specific policy found. Use general support best practices."

    return "\n\n".join(results)

# ── Department map ────────────────────────────────────────────────────────────
DEPARTMENT_MAP = {
    "fraud":           "Fraud Team",
    "legal":           "Legal Team",
    "escalation":      "Escalations Team",
    "password reset":  "Tech Support",
    "technical issue": "Tech Support",
    "payment issue":   "Billing Team",
    "billing":         "Billing Team",
    "invoice":         "Billing Team",
    "refund":          "Refunds Team",
    "cancellation":    "Retention Team",
    "subscription":    "Retention Team",
    "sales lead":      "Sales Team",
    "partnership":     "Sales Team",
    "complaint":       "Customer Care",
    "feedback":        "Customer Care",
    "general query":   "Customer Care",
    "support":         "Customer Care",
    "recruitment":     "HR Team",
    "job application": "HR Team",
}

# ── Hard-escalation rules (Step 10) ──────────────────────────────────────────
ESCALATE_INTENTS  = {"fraud", "legal", "escalation"}
ESCALATE_KEYWORDS = ["lawsuit", "court", "police", "ceo", "government", "media", "threat"]

# ── Analysis prompt ───────────────────────────────────────────────────────────
ANALYSIS_PROMPT = """You are an expert email analysis engine for a customer support system with 10 years of experience.

EXAMPLES OF CORRECT OUTPUT:
Email: "My account is hacked please block it"
Output: {{"intent": "fraud", "emotion": "fear", "sentiment": "NEGATIVE", "priority": "Critical"}}

Email: "I want refund for order 1234 it has been 2 weeks"
Output: {{"intent": "refund", "emotion": "frustrated", "sentiment": "NEGATIVE", "priority": "Medium"}}

Email: "Thank you so much your team was very helpful"
Output: {{"intent": "feedback", "emotion": "happy", "sentiment": "POSITIVE", "priority": "Low"}}

Email: "I will take legal action if not resolved today"
Output: {{"intent": "legal", "emotion": "angry", "sentiment": "NEGATIVE", "priority": "Critical"}}

Email: "OTP is not coming to my phone cannot login"
Output: {{"intent": "password reset", "emotion": "confused", "sentiment": "NEGATIVE", "priority": "High"}}

Email: "I want to cancel my subscription immediately"
Output: {{"intent": "cancellation", "emotion": "frustrated", "sentiment": "NEGATIVE", "priority": "Medium"}}

Email: "When will my invoice be generated? I need it for GST filing."
Output: {{"intent": "invoice", "emotion": "neutral", "sentiment": "NEUTRAL", "priority": "Low"}}

Email: "I would like to know about your enterprise pricing plans."
Output: {{"intent": "sales lead", "emotion": "neutral", "sentiment": "NEUTRAL", "priority": "Low"}}

Email: "I am interested in a partnership opportunity with your company."
Output: {{"intent": "partnership", "emotion": "excited", "sentiment": "POSITIVE", "priority": "Low"}}

INTENT RULES:
- fraud: account hacked, unauthorized access, suspicious activity, block account
- legal: lawsuit, court, legal action, police, consumer court
- escalation: already complained, not resolved, demanding manager
- password reset: OTP issue, cannot login, forgot password, locked out
- payment issue: payment failed, money deducted, transaction failed
- refund: wants money back, return product, refund not received
- complaint: unhappy, bad experience, poor service
- technical issue: app not working, bug, error, website down
- cancellation: wants to cancel subscription or service
- invoice: asking for invoice document, GST invoice, receipt download, billing document
- billing: questions about bill amount, charges, billing cycle (NOT invoice document)
- sales lead: enterprise pricing, bulk plans, demo request, pricing inquiry, how much does it cost
- partnership: collaboration, business opportunity, partner program, joint venture, integrate with
- feedback: happy, satisfied, suggestion, compliment, thank you
- general query: simple question that does not fit any category above

EMOTION RULES:
- angry: unacceptable, disgusting, worst, furious, outraged
- frustrated: tried multiple times, no response, been waiting, immediately, still not resolved
- fear: account hacked, security breach, unauthorized, someone accessed
- sad: disappointed, let down, upset, feeling cheated
- confused: does not understand, unclear, how do I, when will
- happy: thank you, great service, satisfied, excellent
- excited: interested in buying, partnership inquiry, new product
- neutral: simple factual question, no emotional language

PRIORITY RULES:
- Critical: fraud, legal, escalation, any threat
- High: password reset, payment issue, technical issue
- Medium: refund, complaint, cancellation, billing
- Low: feedback, general query, sales lead, partnership, invoice

Now analyze this email and return ONLY this JSON, no explanation, no markdown:
{{
  "intent": "one from the list above",
  "emotion": "one from the list above",
  "sentiment": "POSITIVE or NEGATIVE or NEUTRAL",
  "language": "ISO code e.g. en",
  "priority": "Critical or High or Medium or Low",
  "urgency": "Immediate or Within 1 hour or Within 24 hours or Standard queue",
  "department": "exact team name",
  "risk_flagged": true or false,
  "entities": {{"names": [], "order_ids": [], "amounts": [], "dates": []}},
  "summary": "one sentence summary",
  "customer_value": "VIP or Premium Customer or High Revenue Customer or Regular or New Customer or Frequent Complaint Customer",
  "confidence": 0.0 to 1.0
}}

Email subject: {subject}
Email body: {body}
CRM data: {crm_data}"""


# ── Validation lists ──────────────────────────────────────────────────────────
VALID_INTENTS = [
    "refund", "complaint", "invoice", "cancellation", "technical issue",
    "password reset", "fraud", "general query", "sales lead", "subscription",
    "payment issue", "feedback", "escalation", "legal", "billing",
    "support", "partnership", "recruitment", "job application"
]
VALID_EMOTIONS  = ["happy", "angry", "fear", "sad", "excited", "confused", "frustrated", "neutral"]
VALID_PRIORITY  = ["Critical", "High", "Medium", "Low"]
VALID_SENTIMENT = ["POSITIVE", "NEGATIVE", "NEUTRAL"]


# ── Step 7: Knowledge retrieval ───────────────────────────────────────────────
def retrieve(intent: str, body: str, top_k: int = 2) -> str:
    query = f"{intent}. {body[:300]}"
    query_embedding = _embedder.encode([query], convert_to_numpy=True)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    scores, indices = _faiss_index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if score > 0.3:
            results.append(KB_DOCUMENTS[idx])

    if not results:
        return "No specific policy found. Use general support best practices."

    return "\n\n".join(results)


# ── Internal: single LLM call for analysis ───────────────────────────────────
def _call_analyze(subject: str, body: str, crm_str: str) -> dict:
    prompt = ANALYSIS_PROMPT.format(
        subject  = subject,
        body     = body[:3000],
        crm_data = crm_str,
    )
    raw = client.chat.completions.create(
        model       = MODEL,
        messages    = [{"role": "user", "content": prompt}],
        max_tokens  = 600,
        temperature = 0.0,
    ).choices[0].message.content.strip()

    try:
        return json.loads(re.sub(r"```(?:json)?|```", "", raw).strip())
    except json.JSONDecodeError:
        log.warning("JSON parse failed. Raw: %s", raw[:150])
        return _defaults()


# ── Step 1–6: Analyze email ───────────────────────────────────────────────────
def analyze(subject: str, body: str, crm: dict | None = None) -> dict:
    crm_str = json.dumps(crm) if crm else "No CRM data"

    result = _call_analyze(subject, body, crm_str)

    # Retry once if confidence is low
    if float(result.get("confidence", 0)) < 0.70:
        log.info("Low confidence %.2f — retrying...", result.get("confidence", 0))
        result = _call_analyze(subject, body, crm_str)

    # Validate and fix bad outputs
    if result.get("intent")    not in VALID_INTENTS:  result["intent"]    = "general query"
    if result.get("emotion")   not in VALID_EMOTIONS:  result["emotion"]   = "neutral"
    if result.get("priority")  not in VALID_PRIORITY:  result["priority"]  = "Medium"
    if result.get("sentiment") not in VALID_SENTIMENT: result["sentiment"] = "NEUTRAL"

    # Force department from map — always correct
    result["department"] = DEPARTMENT_MAP.get(result["intent"], "Customer Care")

    result["confidence"]     = float(result.get("confidence", 0.75))
    result["confidence_pct"] = f"{round(result['confidence'] * 100, 1)}%"

    entities = result.get("entities", {})
    for k in ["names", "order_ids", "amounts", "dates"]:
        entities.setdefault(k, [])
    result["entities"] = entities

    return result


# ── Steps 8–9: Generate reply + score confidence ──────────────────────────────
def generate_reply(analysis: dict, subject: str, customer_name: str, history: str, customer_id: str = "") -> dict:
    messages = [
        {"role": "system", "content": (
            "You are a professional VoxIntel customer support agent. "
            "Before writing your reply, use tools if you need the exact policy wording "
            "or the customer's past history. Once you have what you need, write the reply directly — "
            "do not call tools more than necessary."
        )},
        {"role": "user", "content": f"""Customer: {customer_name} ({analysis['customer_value']})
Emotion: {analysis['emotion']} | Intent: {analysis['intent']}
Summary: {analysis['summary']}
Customer ID: {customer_id or 'unknown'}
History passed in: {history or 'none'}

Write a 3–4 sentence empathetic reply that directly resolves the issue.
End with: "Warm regards, VoxIntel Support Team"
Then on a NEW LINE write only: CONFIDENCE: <float 0.0-1.0>"""}
    ]

    raw = ""
    for _ in range(3):  # safety cap on tool round-trips
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=350,
            temperature=0.4,
        )
        msg = response.choices[0].message
        messages.append(msg.model_dump())

        if not msg.tool_calls:
            raw = (msg.content or "").strip()
            break

        for call in msg.tool_calls:
            args   = json.loads(call.function.arguments)
            result = _execute_tool(call.function.name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            })
    else:
        raw = messages[-1].get("content") or ""

    match      = re.search(r"CONFIDENCE:\s*([\d.]+)", raw)
    confidence = float(match.group(1)) if match else 0.75
    confidence = round(min(max(confidence, 0.0), 1.0), 2)
    reply      = raw[:match.start()].strip() if match else raw

    return {
        "reply":          reply,
        "confidence":     confidence,
        "confidence_pct": f"{round(confidence * 100, 1)}%",
    }


# ── Steps 9–10: Route decision ────────────────────────────────────────────────
def decide(analysis: dict, confidence: float, body: str) -> tuple[str, str]:
    body_lower = body.lower()

    if analysis["intent"] in ESCALATE_INTENTS:
        return "ESCALATE", f"Intent '{analysis['intent']}' always requires human review"
    if analysis["customer_value"] == "VIP":
        return "ESCALATE", "VIP customer — mandatory human handling"
    if analysis["risk_flagged"]:
        return "ESCALATE", "Risk flag triggered"
    if any(kw in body_lower for kw in ESCALATE_KEYWORDS):
        return "ESCALATE", "Escalation keyword detected in email"

    threshold_auto    = float(os.getenv("AUTO_SEND_THRESHOLD",        0.90))
    threshold_manager = float(os.getenv("MANAGER_APPROVAL_THRESHOLD", 0.70))

    if confidence >= threshold_auto:
        return "AUTO_SEND",        f"Confidence {confidence:.0%} — auto send"
    if confidence >= threshold_manager:
        return "MANAGER_APPROVAL", f"Confidence {confidence:.0%} — manager review"
    return "HUMAN_AGENT",          f"Confidence {confidence:.0%} — human agent needed"


# ── Draft outgoing email ──────────────────────────────────────────────────────
def draft_email(reply_text: str, customer_email: str, subject: str, analysis: dict) -> dict:
    subject_prefixes = {
        "refund":          "Re: Your Refund Request",
        "fraud":           "Re: Account Security Alert — Urgent",
        "technical issue": "Re: Technical Support",
        "password reset":  "Re: Password Reset Assistance",
        "complaint":       "Re: Your Complaint — We're On It",
        "billing":         "Re: Your Billing Inquiry",
        "cancellation":    "Re: Your Cancellation Request",
        "legal":           "Re: Your Concern — Reference Noted",
        "escalation":      "Re: Your Escalation — Priority Handling",
        "payment issue":   "Re: Payment Issue Resolution",
        "invoice":         "Re: Your Invoice Request",
    }

    email_subject = subject_prefixes.get(analysis.get("intent", ""), f"Re: {subject}")

    email_body = f"""Dear {analysis.get('customer_name', 'Customer')},

{reply_text}

---
Case Reference : {analysis.get('case_id', 'N/A')}
Department     : {analysis.get('department', 'Customer Care')}
Priority       : {analysis.get('priority', 'Medium')}

This is an automated response from VoxIntel AI Support.
For urgent issues, reply directly to this email.

VoxIntel Support Team
support@voxintel.ai | Available 24/7
"""

    return {
        "to":            customer_email,
        "subject":       email_subject,
        "body":          email_body,
        "ready_to_send": True,
    }


# ── Manager alert email ───────────────────────────────────────────────────────
def notify_manager(analysis, reply_text, decision_reason, sender, subject, manager_email=None):
    manager_email = manager_email or os.getenv("MANAGER_EMAIL", "manager@voxintel.ai")
    severity_map  = {"Critical": "🔴 CRITICAL", "High": "🟠 HIGH", "Medium": "🟡 MEDIUM", "Low": "🟢 LOW"}
    severity      = severity_map.get(analysis.get("priority", "Medium"), "🟡 MEDIUM")

    alert_body = f"""⚠️  ESCALATION ALERT — Action Required
{'='*50}

SEVERITY     : {severity}
INTENT       : {analysis.get('intent', 'N/A').upper()}
EMOTION      : {analysis.get('emotion', 'N/A').upper()}
DEPARTMENT   : {analysis.get('department', 'N/A')}
CUSTOMER     : {analysis.get('customer_name', 'Unknown')} ({analysis.get('customer_value', 'Regular')})
FROM         : {sender}
SUBJECT      : {subject}
URGENCY      : {analysis.get('urgency', 'N/A')}

REASON FOR ESCALATION:
{decision_reason}

EMAIL SUMMARY:
{analysis.get('summary', 'N/A')}

AI DRAFTED REPLY (not sent — awaiting your approval):
{'-'*40}
{reply_text}
{'-'*40}

ACTION REQUIRED:
1. Review the customer email
2. Edit or approve the AI draft above
3. Send reply or handle directly

— VoxIntel AI System
"""

    return {
        "to":              manager_email,
        "subject":         f"[VoxIntel ESCALATION] {severity} — {analysis.get('intent', '').upper()} from {sender}",
        "body":            alert_body,
        "priority":        analysis.get("priority", "Medium"),
        "requires_action": True,
    }


# ── Master pipeline ───────────────────────────────────────────────────────────
# ── Master pipeline ───────────────────────────────────────────────────────────
def run(
    subject:       str,
    body:          str,
    sender:        str          = "",
    customer_name: str          = "Customer",
    customer_id:   str          = "",
    crm:           dict | None  = None,
    history:       str          = "",
    manager_email: str          = None,
) -> dict:
    analysis                  = analyze(subject, body, crm)
    analysis["customer_name"] = customer_name
    analysis["case_id"]       = f"CASE-{customer_id or 'ANON'}"

    reply_data       = generate_reply(analysis, subject, customer_name, history, customer_id)
    decision, reason = decide(analysis, reply_data["confidence"], body)

    outgoing_email = draft_email(
        reply_text     = reply_data["reply"],
        customer_email = sender,
        subject        = subject,
        analysis       = analysis,
    )

    manager_alert = None
    if decision in ("ESCALATE", "MANAGER_APPROVAL") or analysis.get("priority") == "Critical":
        manager_alert = notify_manager(
            analysis        = analysis,
            reply_text      = reply_data["reply"],
            decision_reason = reason,
            sender          = sender,
            subject         = subject,
            manager_email   = manager_email,
        )

    return {
        "analysis":        analysis,
        "reply":           reply_data,
        "decision":        decision,
        "decision_reason": reason,
        "outgoing_email":  outgoing_email,
        "manager_alert":   manager_alert,
    }

# ── Parse raw pasted email ────────────────────────────────────────────────────
def parse_raw_email(raw_email: str) -> dict:
    lines         = raw_email.strip().splitlines()
    sender        = ""
    subject       = ""
    customer_name = "Customer"
    body_lines    = []
    in_body       = False

    for line in lines:
        stripped = line.strip()
        lower    = stripped.lower()

        if lower.startswith("from:"):
            sender = stripped[5:].strip()
            if "<" in sender:
                customer_name = sender.split("<")[0].strip()
                sender        = sender.split("<")[1].replace(">", "").strip()
        elif lower.startswith("subject:"):
            subject = stripped[8:].strip()
        elif lower.startswith(("to:", "date:", "cc:")):
            continue
        else:
            in_body = True

        if in_body and not any(lower.startswith(h) for h in ["from:", "subject:", "to:", "date:", "cc:"]):
            body_lines.append(stripped)

    body = "\n".join(body_lines).strip()
    return {
        "sender":        sender,
        "customer_name": customer_name,
        "subject":       subject or "No Subject",
        "body":          body or raw_email.strip(),
    }


# ── Safe defaults ─────────────────────────────────────────────────────────────
def _defaults() -> dict:
    return {
        "intent": "general query", "emotion": "neutral", "sentiment": "NEUTRAL",
        "language": "en", "priority": "Medium", "urgency": "Within 24 hours",
        "department": "Customer Care", "risk_flagged": False,
        "entities": {}, "summary": "Email received.", "customer_value": "Regular",
        "confidence": 0.50,
    }

import database as db  # add this near your other imports

# ── Agentic tools ──────────────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_policy",
            "description": "Search VoxIntel's policy knowledge base (refunds, fraud, password reset, billing, etc). Call this if you're unsure what the official policy says before drafting a reply.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for, e.g. 'refund policy' or 'password reset steps'"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_customer_history",
            "description": "Get this customer's past support interactions to check for repeat issues or unresolved complaints before drafting a reply.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "description": "Customer's ID or email"}
                },
                "required": ["customer_id"]
            }
        }
    }
]

def _execute_tool(name: str, args: dict) -> str:
    if name == "lookup_policy":
        return retrieve(args.get("query", ""), "")
    if name == "get_customer_history":
        with db.get_db() as session:
            rows = db.get_memory(session, args.get("customer_id", ""))
        return json.dumps(rows) if rows else "No past interactions found."
    return "Unknown tool"