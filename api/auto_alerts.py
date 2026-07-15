"""
api/auto_alerts.py
Week 7 — Automated Alert System (Email + Telegram)

Monitors the dashboard /alerts endpoint every N minutes.
When an alert fires, sends a notification via Telegram and/or Email.

Run with:
    python api/auto_alerts.py

Works without Raspberry Pi / Hailo hardware — alerts fire from your
laptop. When the Pi is ready, this same script runs on the Pi instead.
"""

import os
import time
import smtplib
import requests
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config — loaded from .env file
# ---------------------------------------------------------------------------
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
EMAIL_SENDER     = os.getenv("ALERT_EMAIL_SENDER", "")
EMAIL_PASSWORD   = os.getenv("ALERT_EMAIL_PASSWORD", "")
EMAIL_RECEIVER   = os.getenv("ALERT_EMAIL_RECEIVER", "")

DASHBOARD_URL    = "http://localhost:8001/alerts"
CHECK_INTERVAL   = 300   # check every 5 minutes (change to 60 for testing)


# ---------------------------------------------------------------------------
# Send Telegram message
# ---------------------------------------------------------------------------
def send_telegram(message: str) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[Telegram] Skipped — token or chat ID not set in .env")
        return False
    try:
        url  = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        resp = requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }, timeout=10)
        if resp.ok:
            print(f"[Telegram] Sent: {message[:60]}...")
            return True
        else:
            print(f"[Telegram] Failed: {resp.text}")
            return False
    except Exception as e:
        print(f"[Telegram] Error: {e}")
        return False


# ---------------------------------------------------------------------------
# Send Email via Gmail SMTP
# ---------------------------------------------------------------------------
def send_email(subject: str, body: str) -> bool:
    if not EMAIL_SENDER or not EMAIL_PASSWORD:
        print("[Email] Skipped — credentials not set in .env")
        return False
    try:
        msg           = MIMEText(body)
        msg["Subject"] = subject
        msg["From"]    = EMAIL_SENDER
        msg["To"]      = EMAIL_RECEIVER

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)

        print(f"[Email] Sent to {EMAIL_RECEIVER}: {subject}")
        return True
    except Exception as e:
        print(f"[Email] Error: {e}")
        return False


# ---------------------------------------------------------------------------
# Check the dashboard /alerts endpoint
# ---------------------------------------------------------------------------
def check_alerts() -> list:
    try:
        resp = requests.get(DASHBOARD_URL, timeout=5)
        data = resp.json()
        return data.get("alerts", [])
    except Exception as e:
        print(f"[Monitor] Could not reach dashboard API: {e}")
        return []


# ---------------------------------------------------------------------------
# Format alert message
# ---------------------------------------------------------------------------
def format_alert_message(alerts: list) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"🚨 <b>Farmspherica Alert</b> — {timestamp}"]
    for a in alerts:
        sensor  = a.get("sensor", "Unknown")
        message = a.get("message", "No details")
        lines.append(f"\n• <b>{sensor}</b>: {message}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main monitoring loop
# ---------------------------------------------------------------------------
def run_monitor():
    print(f"[Monitor] Starting alert monitor (checking every {CHECK_INTERVAL}s)")
    print(f"[Monitor] Dashboard: {DASHBOARD_URL}")
    print(f"[Monitor] Telegram: {'configured' if TELEGRAM_TOKEN else 'NOT configured'}")
    print(f"[Monitor] Email:    {'configured' if EMAIL_SENDER else 'NOT configured'}")
    print("[Monitor] Press Ctrl+C to stop\n")

    while True:
        alerts = check_alerts()
        if alerts:
            print(f"[Monitor] {len(alerts)} alert(s) detected at "
                  f"{datetime.now().strftime('%H:%M:%S')}")
            message = format_alert_message(alerts)
            send_telegram(message)
            send_email(
                subject=f"Farmspherica Alert — {len(alerts)} sensor issue(s)",
                body=message.replace("<b>", "").replace("</b>", "")
            )
        else:
            print(f"[Monitor] {datetime.now().strftime('%H:%M:%S')} — No alerts")

        time.sleep(CHECK_INTERVAL)


# ---------------------------------------------------------------------------
# One-shot test (used by the test file)
# ---------------------------------------------------------------------------
def test_telegram_connection() -> bool:
    msg = (f"✅ Farmspherica alert system connected — "
           f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return send_telegram(msg)


if __name__ == "__main__":
    run_monitor()