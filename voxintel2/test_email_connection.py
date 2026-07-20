import os
from dotenv import load_dotenv
load_dotenv()

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")

print("=" * 50)
print("VoxIntel Email Connection Test")
print("=" * 50)
print(f"Email configured: {EMAIL_USER}")
print()

# ── Test 1: Check credentials are set ────────────────────────────────────────
if not EMAIL_USER or not EMAIL_PASS:
    print("❌ EMAIL_USER or EMAIL_PASS not set in .env")
    exit()
else:
    print("✅ Credentials found in .env")

# ── Test 2: IMAP connection (can we READ emails?) ─────────────────────────────
print("\nTesting IMAP connection (reading emails)...")
try:
    import imaplib
    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(EMAIL_USER, EMAIL_PASS)
    mail.select("inbox")
    _, data = mail.search(None, "UNSEEN")
    unread_count = len(data[0].split()) if data[0] else 0
    mail.logout()
    print(f"✅ IMAP connected — {unread_count} unread emails in inbox")
except Exception as e:
    print(f"❌ IMAP failed: {e}")
    print("   Fix: Check EMAIL_USER, EMAIL_PASS, and IMAP enabled in Gmail settings")

# ── Test 3: SMTP connection (can we SEND emails?) ─────────────────────────────
print("\nTesting SMTP connection (sending emails)...")
try:
    import smtplib
    server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
    server.login(EMAIL_USER, EMAIL_PASS)
    server.quit()
    print("✅ SMTP connected — can send emails")
except Exception as e:
    print(f"❌ SMTP failed: {e}")

# ── Test 4: Send a real test email to yourself ────────────────────────────────
print("\nSending test email to yourself...")
try:
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    msg = MIMEMultipart()
    msg["From"]    = EMAIL_USER
    msg["To"]      = EMAIL_USER
    msg["Subject"] = "VoxIntel — Email Connection Test ✅"
    msg.attach(MIMEText("""
Hello,

This is a test email from VoxIntel.
If you receive this, your email connection is working correctly.

VoxIntel Support Team
    """, "plain"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)

    print(f"✅ Test email sent to {EMAIL_USER}")
    print("   Check your inbox in 30 seconds")
except Exception as e:
    print(f"❌ Send failed: {e}")

# ── Test 5: Fetch and show unread emails ──────────────────────────────────────
print("\nFetching unread emails preview...")
try:
    import email as emaillib
    import imaplib

    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(EMAIL_USER, EMAIL_PASS)
    mail.select("inbox")
    _, data = mail.search(None, "UNSEEN")
    email_ids = data[0].split()

    if not email_ids:
        print("   No unread emails right now")
    else:
        for eid in email_ids[:3]:   # show max 3
            _, msg_data = mail.fetch(eid, "(RFC822)")
            msg     = emaillib.message_from_bytes(msg_data[0][1])
            sender  = msg.get("From", "Unknown")
            subject = msg.get("Subject", "No Subject")
            print(f"   📧 From: {sender}")
            print(f"      Subject: {subject}")
            print()

    mail.logout()
except Exception as e:
    print(f"❌ Fetch failed: {e}")

print("=" * 50)
print("Test complete")
print("=" * 50)