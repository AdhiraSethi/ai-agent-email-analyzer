import os
from dotenv import load_dotenv
load_dotenv()

import email_connector as ec
import agent
import json

print("Reading emails from Gmail...")
print("=" * 60)

# Step 1: Fetch unread emails from Gmail
emails = ec.fetch_unread_emails()

if not emails:
    print("No unread emails found.")
    print("Send a test email to your Gmail first, then run again.")
    exit()

print(f"Found {len(emails)} unread email(s)\n")

# Step 2: Process each email through VoxIntel
for i, email in enumerate(emails, 1):
    print(f"EMAIL {i}")
    print("-" * 60)
    print(f"From     : {email['sender']}")
    print(f"Name     : {email['customer_name']}")
    print(f"Subject  : {email['subject']}")
    print(f"Body     : {email['body'][:200]}")
    print()

    # Step 3: Run full AI pipeline
    print("Running AI pipeline...")
    result = agent.run(
        subject       = email["subject"],
        body          = email["body"],
        sender        = email["sender"],
        customer_name = email["customer_name"],
    )

    # Step 4: Show analysis
    analysis = result["analysis"]
    print("ANALYSIS")
    print(f"  Intent     : {analysis['intent']}")
    print(f"  Emotion    : {analysis['emotion']}")
    print(f"  Sentiment  : {analysis['sentiment']}")
    print(f"  Priority   : {analysis['priority']}")
    print(f"  Urgency    : {analysis['urgency']}")
    print(f"  Department : {analysis['department']}")
    print(f"  Confidence : {analysis['confidence_pct']}")
    print(f"  Summary    : {analysis['summary']}")
    print()

    # Step 5: Show decision
    print(f"DECISION   : {result['decision']}")
    print(f"REASON     : {result['decision_reason']}")
    print()

    # Step 6: Show drafted reply
    outgoing = result["outgoing_email"]
    print("DRAFTED REPLY EMAIL")
    print(f"  To      : {outgoing['to']}")
    print(f"  Subject : {outgoing['subject']}")
    print(f"  Body    :")
    print()
    print(outgoing["body"])

    # Step 7: Show manager alert if triggered
    if result["manager_alert"]:
        print()
        print("MANAGER ALERT TRIGGERED")
        print(f"  To      : {result['manager_alert']['to']}")
        print(f"  Subject : {result['manager_alert']['subject']}")

    print("=" * 60)

    # Step 8: Ask if you want to send the reply
    print()
    send = input(f"Send this reply to {email['sender']}? (yes/no): ")
    if send.lower() == "yes":
        sent = ec.send_email(
            to      = outgoing["to"],
            subject = outgoing["subject"],
            body    = outgoing["body"],
        )
        if sent:
            print(f"✅ Reply sent to {email['sender']}")
        else:
            print("❌ Failed to send — check email credentials")
    else:
        print("⏭  Skipped — reply not sent")

    print()

print("Done. All emails processed.")