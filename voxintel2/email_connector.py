"""
email_connector.py — fetch emails via IMAP, send via SMTP
Works with Gmail, Outlook, any provider.
"""

import os, imaplib, smtplib, email, logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import decode_header

log = logging.getLogger("voxintel.email")

import config
IMAP_HOST  = config.IMAP_HOST
SMTP_HOST  = config.SMTP_HOST
SMTP_PORT  = config.SMTP_PORT
EMAIL_USER = config.EMAIL_USER
EMAIL_PASS = config.EMAIL_PASS

def fetch_unread_emails() -> list[dict]:
    if not EMAIL_USER or not EMAIL_PASS:
        log.warning("EMAIL_USER or EMAIL_PASS not set — skipping fetch")
        return []
    try:
        mail = imaplib.IMAP4_SSL(IMAP_HOST)
        mail.login(EMAIL_USER, EMAIL_PASS)
        mail.select("inbox")
        _, data = mail.search(None, "UNSEEN")
        email_ids = data[0].split()
        if not email_ids:
            return []

        emails = []
        for eid in email_ids:
            _, msg_data = mail.fetch(eid, "(RFC822)")
            msg     = email.message_from_bytes(msg_data[0][1])
            sender  = _decode(msg.get("From", ""))
            subject = _decode(msg.get("Subject", "No Subject"))
            body    = _get_body(msg)

            customer_name = "Customer"
            sender_email  = sender
            if "<" in sender:
                customer_name = sender.split("<")[0].strip().strip('"')
                sender_email  = sender.split("<")[1].replace(">", "").strip()

            emails.append({
                "message_id":    msg.get("Message-ID", ""),
                "sender":        sender_email,
                "customer_name": customer_name,
                "subject":       subject,
                "body":          body,
            })
            mail.store(eid, "+FLAGS", "\\Seen")

        mail.logout()
        return emails
    except Exception as e:
        log.error("Fetch failed: %s", e)
        return []


def send_email(to: str, subject: str, body: str) -> bool:
    if not EMAIL_USER or not EMAIL_PASS:
        log.warning("Email creds not set — cannot send")
        return False
    if not to:
        return False
    try:
        msg = MIMEMultipart()
        msg["From"]    = EMAIL_USER
        msg["To"]      = to
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(msg)
        log.info("Sent email to %s", to)
        return True
    except Exception as e:
        log.error("Send failed to %s: %s", to, e)
        return False


def _decode(value: str) -> str:
    parts = decode_header(value)
    result = []
    for part, charset in parts:
        if isinstance(part, bytes):
            result.append(part.decode(charset or "utf-8", errors="ignore"))
        else:
            result.append(part)
    return " ".join(result)


def _get_body(msg) -> str:
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    return payload.decode("utf-8", errors="ignore").strip()
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            return payload.decode("utf-8", errors="ignore").strip()
    return ""