"""
scheduler.py — polls inbox every N minutes, runs full pipeline automatically
"""

import os, logging, uuid, time
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

import agent, database as db, email_connector as ec

log           = logging.getLogger("voxintel.scheduler")
POLL_MINUTES  = int(os.getenv("POLL_INTERVAL_MINUTES", 5))
MANAGER_EMAIL = os.getenv("MANAGER_EMAIL", "")
AUTO_SEND     = os.getenv("AUTO_SEND_EMAIL", "false").lower() == "true"


def poll_and_process():
    log.info("Polling inbox...")
    emails = ec.fetch_unread_emails()
    if not emails:
        return

    for raw in emails:
        email_id = str(uuid.uuid4())[:8]
        start    = time.perf_counter()
        try:
            with db.get_db() as session:
                history_rows = db.get_memory(session, raw["sender"])
            history = "\n".join(f"- [{r['time']}] {r['summary']}" for r in history_rows)

            result  = agent.run(
                subject       = raw["subject"],
                body          = raw["body"],
                sender        = raw["sender"],
                customer_name = raw["customer_name"],
                customer_id   = raw["sender"],
                history       = history,
                manager_email = MANAGER_EMAIL,
            )
            elapsed = round((time.perf_counter() - start) * 1000, 1)

            with db.get_db() as session:
                db.save_email_log(session, email_id, raw, result, elapsed)
                db.create_ticket(session, email_id, raw, result)
                db.upsert_customer(session, raw["sender"], raw["sender"], raw["customer_name"])
                db.save_memory(session, raw["sender"], result["analysis"]["summary"],
                               resolved=(result["decision"] == "AUTO_SEND"))

            if AUTO_SEND:
                out = result.get("outgoing_email", {})
                if result["decision"] == "AUTO_SEND" and out.get("to"):
                    ec.send_email(out["to"], out["subject"], out["body"])
                alert = result.get("manager_alert")
                if alert and alert.get("to"):
                    ec.send_email(alert["to"], alert["subject"], alert["body"])

        except Exception as e:
            log.error("Failed on email from %s: %s", raw.get("sender"), e)


scheduler = BackgroundScheduler()

def start_scheduler():
    scheduler.add_job(poll_and_process, IntervalTrigger(minutes=POLL_MINUTES),
                      id="email_poll", replace_existing=True)
    scheduler.start()
    log.info("Scheduler started — every %d min", POLL_MINUTES)

def stop_scheduler():
    scheduler.shutdown(wait=False)