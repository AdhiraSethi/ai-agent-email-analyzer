# VoxIntel — AI Email Intelligence Agent

VoxIntel is an AI-powered email triage and auto-reply agent built for a fictional D2C e-commerce brand, **Lumora**. It reads incoming customer support emails, classifies them across multiple signals (intent, emotion, sentiment, priority, department), drafts a context-aware reply grounded in a policy knowledge base, and routes each email to the right outcome — auto-send, manager approval, human agent, or escalation — based on model confidence and hard business rules.

Built as a hands-on portfolio project to explore LLM-powered backend systems: prompt-driven classification, retrieval-augmented reply generation, agentic tool use, and the operational plumbing (scheduling, persistence, evaluation) that turns a prompt into a real service.

---

## What it does

1. **Fetches** unread emails from a mailbox via IMAP (or accepts them directly through the API).
2. **Preprocesses** each email — strips HTML and signatures, detects language, runs a cheap rule-based spam filter — before anything reaches the LLM.
3. **Analyzes** the email with an LLM call: intent, emotion, sentiment, priority, urgency, department, risk flags, extracted entities, and a one-line summary. Low-confidence results are retried once; invalid values are corrected against fixed lookup tables and a deterministic department map.
4. **Retrieves relevant policy** from a small knowledge base using semantic search (sentence-transformer embeddings + FAISS), so replies are grounded in Lumora's actual policies rather than the model's guesses.
5. **Drafts a reply** using an agentic loop — the model can call tools (`lookup_policy`, `get_customer_history`) before writing, then produces a reply plus a self-reported confidence score.
6. **Routes the decision**:
   - Hard-escalates fraud, legal, VIP customers, risk flags, and threat keywords regardless of confidence.
   - Otherwise routes by confidence threshold: **auto-send** → **manager approval** → **human agent**.
7. **Persists everything** — email logs, tickets, customer records, and per-customer memory — to SQLite via SQLAlchemy, and sends real emails via SMTP when auto-send is enabled.
8. **Learns from corrections** — human edits to AI-drafted replies are logged for future review/fine-tuning.

---

## Architecture

```
                 ┌─────────────────┐
   IMAP inbox ──▶│  Email Connector │
                 └────────┬─────────┘
                          ▼
                 ┌─────────────────┐
                 │   Preprocess     │  HTML strip · signature strip
                 │                  │  language detect · spam filter
                 └────────┬─────────┘
                          ▼
                 ┌─────────────────┐
                 │  VoxIntel Agent  │
                 │ ─────────────── │
                 │ 1. Analyze (LLM) │  intent/emotion/sentiment/
                 │ 2. Retrieve (KB) │  priority via Groq + FAISS
                 │ 3. Reply  (LLM)  │  agentic, tool-using
                 │ 4. Decide        │  confidence + hard rules
                 └────────┬─────────┘
                          ▼
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
   Auto-send email   Manager approval   Human agent /
   (SMTP)            alert (SMTP)       Escalation queue
                          │
                          ▼
                 ┌─────────────────┐
                 │  SQLite / SQLAlchemy │  email_logs · tickets ·
                 │                      │  customers · memory · corrections
                 └─────────────────┘
```

The FastAPI layer (`main.py`) exposes the pipeline over HTTP; `scheduler.py` runs it automatically on a poll loop for a live inbox.

---

## Tech stack

| Layer               | Choice                                             |
|---------------------|-----------------------------------------------------|
| LLM inference        | [Groq API](https://groq.com) (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`) |
| API framework        | FastAPI + Pydantic                                  |
| Semantic retrieval   | `sentence-transformers` (`all-MiniLM-L6-v2`) + FAISS |
| Persistence          | SQLAlchemy + SQLite (swappable for Postgres)        |
| Scheduling           | APScheduler                                          |
| Email                | `imaplib` / `smtplib` (Gmail-compatible)             |
| Auth                 | Simple API-key header (optional, off by default)     |
| Containerization     | Docker                                               |

---

## Project structure

```
.
├── agent.py                  # Core pipeline: analyze → retrieve → reply → decide
├── main.py                   # FastAPI app and routes
├── scheduler.py               # Background inbox polling (APScheduler)
├── email_connector.py         # IMAP fetch / SMTP send
├── preprocess.py               # HTML/signature stripping, language detect, spam filter
├── database.py                 # SQLAlchemy models + persistence helpers
├── auth.py                     # Optional API-key auth
├── config.py                   # Centralized settings, loaded from .env
├── evaluate.py                  # Accuracy/latency/confidence evaluation harness
├── test_emails.json              # Labeled test set for evaluate.py
├── evaluation_results.json        # Latest evaluation output
├── read_and_reply.py               # CLI script: fetch live inbox, review, send interactively
├── test_email_connection.py         # Standalone IMAP/SMTP connectivity check
├── corrections.jsonl                 # Logged human corrections to AI replies
├── requirements.txt
└── Dockerfile
```

---

## Setup

### 1. Clone and install

```bash
pip install -r requirements.txt
```

Requires Python 3.11+ (developed on Python 3.14).

### 2. Configure environment

Create a `.env` file in the project root:

```env
# Groq
GROQ_API_KEY=your_groq_api_key
MODEL=llama-3.3-70b-versatile
MODEL_FAST=llama-3.1-8b-instant

# Email (optional — omit to run in manual /process-raw mode only)
EMAIL_USER=you@gmail.com
EMAIL_PASS=your_app_password
EMAIL_IMAP_HOST=imap.gmail.com
EMAIL_SMTP_HOST=smtp.gmail.com
EMAIL_SMTP_PORT=465
MANAGER_EMAIL=manager@example.com
AUTO_SEND_EMAIL=false

# Decision thresholds
AUTO_SEND_THRESHOLD=0.90
MANAGER_APPROVAL_THRESHOLD=0.70
POLL_INTERVAL_MINUTES=5

# Database
DATABASE_URL=sqlite:///voxintel.db

# Optional API key protection (leave blank to disable)
API_SECRET_KEY=

LOG_LEVEL=INFO
```

For Gmail, use an [App Password](https://support.google.com/accounts/answer/185833) — not your regular account password — and make sure IMAP is enabled in Gmail settings.

### 3. Run the API

```bash
uvicorn main:app --reload
```

The app starts the background poller automatically if `EMAIL_USER`/`EMAIL_PASS` are set; otherwise it runs in manual mode (`/process-raw` and `/process` only).

### 4. Or run with Docker

```bash
docker build -t voxintel .
docker run -p 8000:8000 --env-file .env voxintel
```

---

## API reference

| Method | Route                | Description                                                   |
|--------|-----------------------|-----------------------------------------------------------------|
| GET    | `/health`               | Service status, active model, auto-send flag                    |
| POST   | `/process`               | Run the full pipeline on a structured JSON email                 |
| POST   | `/process-raw`            | Run the full pipeline on a raw pasted email (headers + body)      |
| POST   | `/analyze`                 | Run analysis only (no reply generation)                            |
| GET    | `/emails`                    | List all processed emails (in-memory)                               |
| GET    | `/emails/{email_id}`           | Full detail for one processed email                                   |
| GET    | `/memory/{customer_id}`          | Conversation history for a customer                                     |
| POST   | `/correction`                      | Log a human edit to an AI-drafted reply                                   |
| GET    | `/stats`                             | Aggregate dashboard stats (volume, decisions, avg confidence)              |

Example request to `/process`:

```json
POST /process
{
  "subject": "Refund request",
  "body": "I want a refund for order #1234, it's been 2 weeks.",
  "sender": "customer@example.com",
  "customer_name": "Jane Doe",
  "customer_id": "cust_001"
}
```

Interactive docs are available at `/docs` once the server is running.

---

## Standalone scripts

- **`read_and_reply.py`** — connects to a live Gmail inbox, runs the full pipeline on each unread email, prints the analysis/decision/drafted reply, and asks for confirmation before sending.
- **`test_email_connection.py`** — verifies IMAP/SMTP credentials and sends a test email to yourself.
- **`evaluate.py`** — runs the analysis pipeline against `test_emails.json` and reports per-field accuracy, intent precision/recall, latency, and confidence. Results are written to `evaluation_results.json`.

```bash
python evaluate.py
python test_email_connection.py
python read_and_reply.py
```

## Evaluation results (latest run)

| Field       | Accuracy |
|-------------|----------|
| Intent       | 100%     |
| Emotion      | 90%      |
| Sentiment    | 100%     |
| Priority     | 100%     |
| Department   | 100%     |
| **Overall**  | **98%**  |

Average confidence: 91% · Average latency: ~7.4s per email (varies with Groq load).

---

## Decision logic

Every email is force-escalated regardless of confidence if:
- Intent is `fraud`, `legal`, or `escalation`
- Customer is flagged VIP
- The analysis sets `risk_flagged: true`
- The body contains an escalation keyword (`lawsuit`, `court`, `police`, `ceo`, `government`, `media`, `threat`)

Otherwise, routing follows the reply's confidence score:

| Confidence         | Decision            |
|---------------------|-----------------------|
| ≥ 90% (`AUTO_SEND_THRESHOLD`)          | Auto-send reply           |
| ≥ 70% (`MANAGER_APPROVAL_THRESHOLD`)     | Manager approval needed     |
| < 70%                                      | Routed to a human agent       |

Both thresholds are configurable via `.env`.

---

## Notes on the fictional scenario

Lumora and its policies (refund windows, fraud SLAs, support hours) are fictional and exist purely to give the knowledge base and department routing a coherent, realistic context for demo and evaluation purposes.

## Known limitations / next steps

- In-memory conversation store in `main.py` (`memory` dict) doesn't persist across restarts — the scheduler-driven path uses the database instead, but the pure-API path does not yet.
- `/process-raw` has an incomplete implementation and needs its pipeline call wired up.
- OCR for email attachments is stubbed out (`preprocess.extract_attachment_text`) but not implemented.
- No automated test suite beyond `evaluate.py`'s labeled accuracy check.
