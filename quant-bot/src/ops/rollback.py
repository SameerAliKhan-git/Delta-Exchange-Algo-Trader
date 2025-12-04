"""
Automated Rollback System
=========================
Production rollback with kill switch integration.

Triggers:
- Prometheus critical alert webhook
- KILL_SWITCH file creation
- Manual invocation

Actions:
- Stop all trading
- Rollback to last known good model
- Freeze online learning
- Notify ops team
- Create incident ticket
"""

import os
import sys
import json
import shutil
import asyncio
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any
import subprocess
import requests

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'rollback_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class RollbackConfig:
    """Rollback configuration."""
    
    # Paths
    KILL_SWITCH_FILE = Path("./KILL_SWITCH")
    MODEL_REGISTRY_DIR = Path("./models/registry")
    LAST_GOOD_MODEL_FILE = Path("./models/last_good_model.json")
    ROLLBACK_HISTORY_FILE = Path("./logs/rollback_history.json")
    
    # Notifications
    SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
    DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
    PAGERDUTY_KEY = os.getenv("PAGERDUTY_KEY", "")
    JIRA_API_URL = os.getenv("JIRA_API_URL", "")
    JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN", "")
    
    # Services
    TRADING_SERVICE_URL = os.getenv("TRADING_SERVICE_URL", "http://localhost:8000")
    
    # Safety
    MAX_ROLLBACK_ATTEMPTS = 3
    ROLLBACK_COOLDOWN_SECONDS = 300  # 5 minutes between rollbacks


# =============================================================================
# ROLLBACK ACTIONS
# =============================================================================

class RollbackManager:
    """Manage automated rollback operations."""
    
    def __init__(self, config: RollbackConfig = None):
        self.config = config or RollbackConfig()
        self.rollback_count = 0
        self.last_rollback_time = None
    
    def activate_kill_switch(self, reason: str) -> bool:
        """Activate the kill switch."""
        try:
            kill_data = {
                "activated_at": datetime.now().isoformat(),
                "reason": reason,
                "activated_by": "rollback_system"
            }
            
            with open(self.config.KILL_SWITCH_FILE, 'w') as f:
                json.dump(kill_data, f, indent=2)
            
            logger.critical(f"KILL SWITCH ACTIVATED: {reason}")
            return True
        except Exception as e:
            logger.error(f"Failed to activate kill switch: {e}")
            return False
    
    def stop_trading(self) -> bool:
        """Stop all trading activity."""
        try:
            # Method 1: API call to trading service
            try:
                response = requests.post(
                    f"{self.config.TRADING_SERVICE_URL}/api/v1/kill-switch",
                    json={"action": "activate", "reason": "automated_rollback"},
                    timeout=10
                )
                if response.status_code == 200:
                    logger.info("Trading stopped via API")
                    return True
            except requests.RequestException as e:
                logger.warning(f"API call failed: {e}")
            
            # Method 2: Kill switch file
            self.activate_kill_switch("automated_rollback_triggered")
            
            # Method 3: Send signal to process (if PID file exists)
            pid_file = Path("./trading.pid")
            if pid_file.exists():
                pid = int(pid_file.read_text().strip())
                try:
                    os.kill(pid, 15)  # SIGTERM
                    logger.info(f"Sent SIGTERM to trading process {pid}")
                except ProcessLookupError:
                    logger.warning(f"Process {pid} not found")
            
            return True
        except Exception as e:
            logger.error(f"Failed to stop trading: {e}")
            return False
    
    def get_last_good_model(self) -> Optional[Dict]:
        """Get the last known good model version."""
        try:
            if self.config.LAST_GOOD_MODEL_FILE.exists():
                with open(self.config.LAST_GOOD_MODEL_FILE, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            logger.error(f"Failed to get last good model: {e}")
            return None
    
    def rollback_model(self, target_version: str = None) -> bool:
        """Rollback to a specific model version or last good model."""
        try:
            if target_version:
                model_info = {"version": target_version}
            else:
                model_info = self.get_last_good_model()
                if not model_info:
                    logger.error("No last good model found")
                    return False
            
            version = model_info.get("version")
            model_path = self.config.MODEL_REGISTRY_DIR / f"model_{version}"
            
            if not model_path.exists():
                logger.error(f"Model version {version} not found at {model_path}")
                return False
            
            # Copy model to active location
            active_model_path = Path("./models/active")
            if active_model_path.exists():
                backup_path = Path(f"./models/backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                shutil.move(str(active_model_path), str(backup_path))
                logger.info(f"Backed up current model to {backup_path}")
            
            shutil.copytree(str(model_path), str(active_model_path))
            logger.info(f"Rolled back to model version {version}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to rollback model: {e}")
            return False
    
    def freeze_online_learning(self) -> bool:
        """Freeze online learning updates."""
        try:
            freeze_config = {
                "frozen_at": datetime.now().isoformat(),
                "accept_threshold": 0.99,  # Very high threshold = effectively frozen
                "reason": "automated_rollback"
            }
            
            freeze_file = Path("./config/online_learning_freeze.json")
            with open(freeze_file, 'w') as f:
                json.dump(freeze_config, f, indent=2)
            
            logger.info("Online learning frozen")
            return True
        except Exception as e:
            logger.error(f"Failed to freeze online learning: {e}")
            return False
    
    def reduce_position_sizes(self, reduction_factor: float = 0.1) -> bool:
        """Reduce all position sizes to minimum."""
        try:
            position_config = {
                "max_position_pct": reduction_factor,
                "reason": "automated_rollback",
                "applied_at": datetime.now().isoformat()
            }
            
            config_file = Path("./config/position_override.json")
            with open(config_file, 'w') as f:
                json.dump(position_config, f, indent=2)
            
            logger.info(f"Position sizes reduced to {reduction_factor * 100}%")
            return True
        except Exception as e:
            logger.error(f"Failed to reduce positions: {e}")
            return False
    
    def send_notifications(self, incident: Dict) -> None:
        """Send notifications to all channels."""
        message = f"""
🚨 **AUTOMATED ROLLBACK TRIGGERED**

**Time:** {incident['timestamp']}
**Reason:** {incident['reason']}
**Trigger:** {incident['trigger']}
**Actions Taken:**
{chr(10).join(f"  • {action}" for action in incident['actions'])}

**Status:** {incident['status']}

Immediate investigation required.
"""
        
        # Slack
        if self.config.SLACK_WEBHOOK_URL:
            try:
                requests.post(
                    self.config.SLACK_WEBHOOK_URL,
                    json={
                        "text": message,
                        "channel": "#trading-critical",
                        "username": "Rollback Bot",
                        "icon_emoji": ":rotating_light:"
                    },
                    timeout=10
                )
                logger.info("Slack notification sent")
            except Exception as e:
                logger.error(f"Slack notification failed: {e}")
        
        # Discord
        if self.config.DISCORD_WEBHOOK_URL:
            try:
                requests.post(
                    self.config.DISCORD_WEBHOOK_URL,
                    json={"content": message},
                    timeout=10
                )
                logger.info("Discord notification sent")
            except Exception as e:
                logger.error(f"Discord notification failed: {e}")
        
        # PagerDuty
        if self.config.PAGERDUTY_KEY:
            try:
                requests.post(
                    "https://events.pagerduty.com/v2/enqueue",
                    json={
                        "routing_key": self.config.PAGERDUTY_KEY,
                        "event_action": "trigger",
                        "payload": {
                            "summary": f"Trading Rollback: {incident['reason']}",
                            "severity": "critical",
                            "source": "rollback_system"
                        }
                    },
                    timeout=10
                )
                logger.info("PagerDuty alert sent")
            except Exception as e:
                logger.error(f"PagerDuty notification failed: {e}")
    
    def create_jira_ticket(self, incident: Dict) -> Optional[str]:
        """Create a JIRA ticket for the incident."""
        if not self.config.JIRA_API_URL or not self.config.JIRA_API_TOKEN:
            return None
        
        try:
            response = requests.post(
                f"{self.config.JIRA_API_URL}/rest/api/2/issue",
                headers={
                    "Authorization": f"Bearer {self.config.JIRA_API_TOKEN}",
                    "Content-Type": "application/json"
                },
                json={
                    "fields": {
                        "project": {"key": "TRADING"},
                        "summary": f"[AUTO] Rollback: {incident['reason'][:50]}",
                        "description": json.dumps(incident, indent=2),
                        "issuetype": {"name": "Incident"},
                        "priority": {"name": "Critical"}
                    }
                },
                timeout=30
            )
            
            if response.status_code == 201:
                ticket_key = response.json().get("key")
                logger.info(f"JIRA ticket created: {ticket_key}")
                return ticket_key
            else:
                logger.error(f"JIRA ticket creation failed: {response.text}")
                return None
        except Exception as e:
            logger.error(f"JIRA API error: {e}")
            return None
    
    def record_rollback(self, incident: Dict) -> None:
        """Record rollback in history."""
        try:
            history = []
            if self.config.ROLLBACK_HISTORY_FILE.exists():
                with open(self.config.ROLLBACK_HISTORY_FILE, 'r') as f:
                    history = json.load(f)
            
            history.append(incident)
            
            # Keep last 100 rollbacks
            history = history[-100:]
            
            self.config.ROLLBACK_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config.ROLLBACK_HISTORY_FILE, 'w') as f:
                json.dump(history, f, indent=2)
            
            logger.info("Rollback recorded in history")
        except Exception as e:
            logger.error(f"Failed to record rollback: {e}")
    
    async def execute_rollback(
        self,
        reason: str,
        trigger: str = "manual",
        target_version: str = None
    ) -> Dict:
        """Execute full rollback procedure."""
        
        # Check cooldown
        if self.last_rollback_time:
            elapsed = (datetime.now() - self.last_rollback_time).total_seconds()
            if elapsed < self.config.ROLLBACK_COOLDOWN_SECONDS:
                logger.warning(f"Rollback on cooldown. {self.config.ROLLBACK_COOLDOWN_SECONDS - elapsed:.0f}s remaining")
                return {"status": "cooldown", "reason": "Too soon since last rollback"}
        
        # Check max attempts
        if self.rollback_count >= self.config.MAX_ROLLBACK_ATTEMPTS:
            logger.error("Max rollback attempts reached. Manual intervention required.")
            return {"status": "max_attempts", "reason": "Max rollback attempts exceeded"}
        
        self.rollback_count += 1
        self.last_rollback_time = datetime.now()
        
        incident = {
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "trigger": trigger,
            "target_version": target_version,
            "actions": [],
            "status": "in_progress"
        }
        
        logger.critical(f"EXECUTING ROLLBACK: {reason}")
        
        # Step 1: Stop trading
        if self.stop_trading():
            incident["actions"].append("Trading stopped")
        else:
            incident["actions"].append("FAILED: Stop trading")
        
        # Step 2: Activate kill switch
        if self.activate_kill_switch(reason):
            incident["actions"].append("Kill switch activated")
        else:
            incident["actions"].append("FAILED: Kill switch")
        
        # Step 3: Rollback model
        if self.rollback_model(target_version):
            incident["actions"].append(f"Model rolled back to {target_version or 'last_good'}")
        else:
            incident["actions"].append("FAILED: Model rollback")
        
        # Step 4: Freeze online learning
        if self.freeze_online_learning():
            incident["actions"].append("Online learning frozen")
        else:
            incident["actions"].append("FAILED: Freeze online learning")
        
        # Step 5: Reduce position sizes
        if self.reduce_position_sizes(0.1):
            incident["actions"].append("Position sizes reduced to 10%")
        else:
            incident["actions"].append("FAILED: Reduce positions")
        
        # Determine final status
        failures = [a for a in incident["actions"] if a.startswith("FAILED")]
        incident["status"] = "completed" if not failures else "completed_with_errors"
        
        # Send notifications
        self.send_notifications(incident)
        
        # Create JIRA ticket
        ticket = self.create_jira_ticket(incident)
        if ticket:
            incident["jira_ticket"] = ticket
        
        # Record in history
        self.record_rollback(incident)
        
        logger.info(f"Rollback completed with status: {incident['status']}")
        return incident


# =============================================================================
# ALERT WEBHOOK HANDLER
# =============================================================================

async def handle_prometheus_alert(alert_data: Dict) -> Dict:
    """Handle incoming Prometheus alert webhook."""
    
    manager = RollbackManager()
    
    for alert in alert_data.get("alerts", []):
        if alert.get("status") == "firing":
            labels = alert.get("labels", {})
            annotations = alert.get("annotations", {})
            
            severity = labels.get("severity")
            action = annotations.get("action", "")
            
            if severity == "critical" and action == "AUTO_KILL_SWITCH":
                reason = annotations.get("summary", "Critical alert triggered")
                return await manager.execute_rollback(
                    reason=reason,
                    trigger=f"prometheus_alert:{labels.get('alertname')}"
                )
    
    return {"status": "no_action", "reason": "No critical alerts requiring rollback"}


# =============================================================================
# CLI
# =============================================================================

async def main():
    """CLI entry point."""
    
    parser = argparse.ArgumentParser(description="Automated Rollback System")
    parser.add_argument(
        "--target-version",
        type=str,
        help="Target model version to rollback to"
    )
    parser.add_argument(
        "--reason",
        type=str,
        default="Manual rollback initiated",
        help="Reason for rollback"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate rollback without taking action"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current rollback status"
    )
    
    args = parser.parse_args()
    
    manager = RollbackManager()
    
    if args.status:
        # Show current status
        kill_switch = manager.config.KILL_SWITCH_FILE.exists()
        last_good = manager.get_last_good_model()
        
        print("\n=== ROLLBACK SYSTEM STATUS ===")
        print(f"Kill Switch Active: {'YES' if kill_switch else 'NO'}")
        print(f"Last Good Model: {last_good.get('version') if last_good else 'None'}")
        
        if manager.config.ROLLBACK_HISTORY_FILE.exists():
            with open(manager.config.ROLLBACK_HISTORY_FILE, 'r') as f:
                history = json.load(f)
            print(f"Total Rollbacks: {len(history)}")
            if history:
                last = history[-1]
                print(f"Last Rollback: {last['timestamp']} - {last['reason']}")
        return
    
    if args.dry_run:
        print("\n=== DRY RUN - No actions will be taken ===")
        print(f"Reason: {args.reason}")
        print(f"Target Version: {args.target_version or 'last_good'}")
        print("\nActions that would be taken:")
        print("  1. Stop trading")
        print("  2. Activate kill switch")
        print("  3. Rollback model")
        print("  4. Freeze online learning")
        print("  5. Reduce position sizes")
        print("  6. Send notifications")
        print("  7. Create JIRA ticket")
        return
    
    # Execute rollback
    result = await manager.execute_rollback(
        reason=args.reason,
        trigger="cli",
        target_version=args.target_version
    )
    
    print(f"\nRollback Result: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())
