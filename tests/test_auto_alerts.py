"""
tests/test_auto_alerts.py
Tests for the Week 7 alert system.

Run with:
    python tests/test_auto_alerts.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / "api"))

from api.auto_alerts import (
    format_alert_message, send_telegram, send_email, check_alerts
)


def test_format_alert_message_structure():
    fake_alerts = [
        {"sensor": "pH", "value": 3.9,
         "message": "pH is 3.9 — below safe minimum of 4.0"},
        {"sensor": "ML_MODEL", "value": None,
         "message": "ML anomaly model flagged this reading"},
    ]
    msg = format_alert_message(fake_alerts)
    assert "Farmspherica Alert" in msg
    assert "pH" in msg
    assert "ML_MODEL" in msg
    print("Test 1 passed: alert message formats correctly")


def test_format_message_with_no_alerts():
    msg = format_alert_message([])
    assert "Farmspherica Alert" in msg
    print("Test 2 passed: empty alert list handled correctly")


def test_telegram_sends_without_crash():
    # Will print "Skipped" if .env not configured — that's fine
    result = send_telegram("Farmspherica test message from test_auto_alerts.py")
    assert isinstance(result, bool)
    print(f"Test 3 passed: Telegram send ran without crash (result={result})")


def test_email_sends_without_crash():
    result = send_email(
        subject="Farmspherica Test",
        body="This is a test message from test_auto_alerts.py"
    )
    assert isinstance(result, bool)
    print(f"Test 4 passed: Email send ran without crash (result={result})")


def test_check_alerts_returns_list():
    # Dashboard API may not be running — check_alerts must return a list either way
    alerts = check_alerts()
    assert isinstance(alerts, list)
    print(f"Test 5 passed: check_alerts returned a list ({len(alerts)} alerts)")


if __name__ == "__main__":
    test_format_alert_message_structure()
    test_format_message_with_no_alerts()
    test_telegram_sends_without_crash()
    test_email_sends_without_crash()
    test_check_alerts_returns_list()
    print("\nAll Week 7 alert system tests passed!")