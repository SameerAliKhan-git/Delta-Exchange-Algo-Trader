#!/usr/bin/env python3
"""
Alert Checker Script
=====================
Parses Prometheus API for any firing critical alerts and sends notifications.
Designed to run via cron for daily health checks.

Usage:
    python scripts/check_alerts.py
    python scripts/check_alerts.py --prometheus-url http://localhost:9090
    python scripts/check_alerts.py --email ops@company.com

Author: Quant Bot Ops
Version: 1.0.0
"""

import os
import sys
import json
import argparse
import smtplib
import requests
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class AlertState(Enum):
    FIRING = "firing"
    PENDING = "pending"
    INACTIVE = "inactive"


@dataclass
class Alert:
    """Represents a Prometheus alert."""
    name: str
    state: AlertState
    severity: str
    summary: str
    description: str
    labels: Dict[str, str]
    annotations: Dict[str, str]
    active_at: Optional[datetime]
    value: Optional[float]
    
    @property
    def is_critical(self) -> bool:
        return self.severity.lower() == "critical"
    
    @property
    def is_warning(self) -> bool:
        return self.severity.lower() == "warning"


class AlertChecker:
    """Check Prometheus for firing alerts."""
    
    def __init__(
        self,
        prometheus_url: Optional[str] = None,
        slack_webhook: Optional[str] = None,
        email_recipients: Optional[List[str]] = None,
        pagerduty_key: Optional[str] = None
    ):
        self.prometheus_url = prometheus_url or os.getenv(
            "PROMETHEUS_URL",
            "http://localhost:9090"
        )
        self.slack_webhook = slack_webhook or os.getenv("SLACK_WEBHOOK_URL")
        self.email_recipients = email_recipients or []
        self.pagerduty_key = pagerduty_key or os.getenv("PAGERDUTY_INTEGRATION_KEY")
        
        # Email settings
        self.smtp_host = os.getenv("SMTP_HOST", "localhost")
        self.smtp_port = int(os.getenv("SMTP_PORT", "25"))
        self.smtp_user = os.getenv("SMTP_USER")
        self.smtp_password = os.getenv("SMTP_PASSWORD")
        self.email_from = os.getenv("EMAIL_FROM", "quant-bot@localhost")
    
    def get_alerts(self) -> List[Alert]:
        """Fetch current alerts from Prometheus."""
        try:
            # Query Prometheus alerts API
            response = requests.get(
                f"{self.prometheus_url}/api/v1/alerts",
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            if data.get("status") != "success":
                print(f"Prometheus API error: {data}")
                return []
            
            alerts = []
            for alert_data in data.get("data", {}).get("alerts", []):
                alerts.append(Alert(
                    name=alert_data.get("labels", {}).get("alertname", "Unknown"),
                    state=AlertState(alert_data.get("state", "inactive")),
                    severity=alert_data.get("labels", {}).get("severity", "unknown"),
                    summary=alert_data.get("annotations", {}).get("summary", ""),
                    description=alert_data.get("annotations", {}).get("description", ""),
                    labels=alert_data.get("labels", {}),
                    annotations=alert_data.get("annotations", {}),
                    active_at=datetime.fromisoformat(
                        alert_data["activeAt"].replace("Z", "+00:00")
                    ) if alert_data.get("activeAt") else None,
                    value=float(alert_data.get("value", 0)) if alert_data.get("value") else None
                ))
            
            return alerts
            
        except requests.exceptions.ConnectionError:
            print(f"Cannot connect to Prometheus at {self.prometheus_url}")
            return []
        except Exception as e:
            print(f"Error fetching alerts: {e}")
            return []
    
    def check(self) -> Dict[str, Any]:
        """Run alert check and return summary."""
        alerts = self.get_alerts()
        
        firing = [a for a in alerts if a.state == AlertState.FIRING]
        pending = [a for a in alerts if a.state == AlertState.PENDING]
        
        critical = [a for a in firing if a.is_critical]
        warnings = [a for a in firing if a.is_warning]
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "prometheus_url": self.prometheus_url,
            "total_alerts": len(alerts),
            "firing_count": len(firing),
            "pending_count": len(pending),
            "critical_count": len(critical),
            "warning_count": len(warnings),
            "critical_alerts": [self._alert_to_dict(a) for a in critical],
            "warning_alerts": [self._alert_to_dict(a) for a in warnings],
            "status": "CRITICAL" if critical else ("WARNING" if warnings else "OK")
        }
        
        return result
    
    def _alert_to_dict(self, alert: Alert) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "name": alert.name,
            "severity": alert.severity,
            "summary": alert.summary,
            "description": alert.description,
            "active_since": alert.active_at.isoformat() if alert.active_at else None,
            "value": alert.value
        }
    
    def send_slack_notification(self, result: Dict[str, Any]) -> bool:
        """Send alert summary to Slack."""
        if not self.slack_webhook:
            return False
        
        # Build message
        status_emoji = {
            "CRITICAL": "🚨",
            "WARNING": "⚠️",
            "OK": "✅"
        }.get(result["status"], "ℹ️")
        
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{status_emoji} Alert Check: {result['status']}"
                }
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Total Firing:*\n{result['firing_count']}"},
                    {"type": "mrkdwn", "text": f"*Critical:*\n{result['critical_count']}"},
                    {"type": "mrkdwn", "text": f"*Warnings:*\n{result['warning_count']}"},
                    {"type": "mrkdwn", "text": f"*Pending:*\n{result['pending_count']}"}
                ]
            }
        ]
        
        # Add critical alerts
        for alert in result.get("critical_alerts", [])[:5]:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"🚨 *{alert['name']}*\n{alert['summary']}"
                }
            })
        
        # Add warning alerts
        for alert in result.get("warning_alerts", [])[:3]:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"⚠️ *{alert['name']}*\n{alert['summary']}"
                }
            })
        
        try:
            response = requests.post(
                self.slack_webhook,
                json={"blocks": blocks},
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Slack notification failed: {e}")
            return False
    
    def send_email_notification(self, result: Dict[str, Any]) -> bool:
        """Send alert summary via email."""
        if not self.email_recipients:
            return False
        
        subject = f"[{result['status']}] Quant Bot Alert Check - {result['critical_count']} Critical, {result['warning_count']} Warnings"
        
        # Build HTML body
        html = f"""
        <html>
        <body>
        <h2>Alert Check Summary</h2>
        <p><strong>Status:</strong> {result['status']}</p>
        <p><strong>Time:</strong> {result['timestamp']}</p>
        
        <h3>Summary</h3>
        <ul>
            <li>Total Firing: {result['firing_count']}</li>
            <li>Critical: {result['critical_count']}</li>
            <li>Warnings: {result['warning_count']}</li>
            <li>Pending: {result['pending_count']}</li>
        </ul>
        """
        
        if result.get("critical_alerts"):
            html += "<h3>🚨 Critical Alerts</h3><ul>"
            for alert in result["critical_alerts"]:
                html += f"<li><strong>{alert['name']}</strong>: {alert['summary']}</li>"
            html += "</ul>"
        
        if result.get("warning_alerts"):
            html += "<h3>⚠️ Warning Alerts</h3><ul>"
            for alert in result["warning_alerts"]:
                html += f"<li><strong>{alert['name']}</strong>: {alert['summary']}</li>"
            html += "</ul>"
        
        html += "</body></html>"
        
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.email_from
            msg["To"] = ", ".join(self.email_recipients)
            msg.attach(MIMEText(html, "html"))
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.smtp_user and self.smtp_password:
                    server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.email_from, self.email_recipients, msg.as_string())
            
            return True
        except Exception as e:
            print(f"Email notification failed: {e}")
            return False
    
    def trigger_pagerduty(self, result: Dict[str, Any]) -> bool:
        """Trigger PagerDuty incident for critical alerts."""
        if not self.pagerduty_key or result["status"] != "CRITICAL":
            return False
        
        try:
            payload = {
                "routing_key": self.pagerduty_key,
                "event_action": "trigger",
                "dedup_key": f"quant_bot_critical_{datetime.now().strftime('%Y%m%d')}",
                "payload": {
                    "summary": f"Quant Bot: {result['critical_count']} Critical Alerts Firing",
                    "source": "quant-bot-alert-checker",
                    "severity": "critical",
                    "custom_details": {
                        "critical_alerts": result["critical_alerts"],
                        "warning_count": result["warning_count"],
                        "prometheus_url": result["prometheus_url"]
                    }
                }
            }
            
            response = requests.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=payload,
                timeout=10
            )
            return response.status_code == 202
        except Exception as e:
            print(f"PagerDuty trigger failed: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Check Prometheus for firing alerts")
    parser.add_argument(
        "--prometheus-url",
        help="Prometheus URL (default: env PROMETHEUS_URL or localhost:9090)"
    )
    parser.add_argument(
        "--email",
        action="append",
        dest="emails",
        help="Email recipient (can be specified multiple times)"
    )
    parser.add_argument(
        "--slack",
        action="store_true",
        help="Send Slack notification"
    )
    parser.add_argument(
        "--pagerduty",
        action="store_true",
        help="Trigger PagerDuty for critical alerts"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON format"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only output on non-OK status"
    )
    
    args = parser.parse_args()
    
    checker = AlertChecker(
        prometheus_url=args.prometheus_url,
        email_recipients=args.emails
    )
    
    result = checker.check()
    
    # Output
    if args.json:
        print(json.dumps(result, indent=2))
    elif not args.quiet or result["status"] != "OK":
        print("="*60)
        print(f"ALERT CHECK: {result['status']}")
        print("="*60)
        print(f"Time: {result['timestamp']}")
        print(f"Prometheus: {result['prometheus_url']}")
        print(f"\nFiring: {result['firing_count']} | Critical: {result['critical_count']} | Warnings: {result['warning_count']}")
        
        if result["critical_alerts"]:
            print("\n🚨 CRITICAL ALERTS:")
            for alert in result["critical_alerts"]:
                print(f"  - {alert['name']}: {alert['summary']}")
        
        if result["warning_alerts"]:
            print("\n⚠️  WARNING ALERTS:")
            for alert in result["warning_alerts"]:
                print(f"  - {alert['name']}: {alert['summary']}")
        
        print("="*60)
    
    # Notifications
    if args.slack:
        if checker.send_slack_notification(result):
            print("✓ Slack notification sent")
        else:
            print("✗ Slack notification failed")
    
    if args.emails:
        if checker.send_email_notification(result):
            print(f"✓ Email sent to {', '.join(args.emails)}")
        else:
            print("✗ Email notification failed")
    
    if args.pagerduty and result["status"] == "CRITICAL":
        if checker.trigger_pagerduty(result):
            print("✓ PagerDuty incident triggered")
        else:
            print("✗ PagerDuty trigger failed")
    
    # Exit code based on status
    if result["status"] == "CRITICAL":
        sys.exit(2)
    elif result["status"] == "WARNING":
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
