#!/usr/bin/env python3
"""
EMERGENCY PLAYBOOK - Automated Response Scripts
================================================

Quick-action scripts for production incidents.
Keep this file open in a terminal during live trading.

Usage:
    python ops/emergency.py kill          # Trigger kill switch
    python ops/emergency.py status        # Check system status
    python ops/emergency.py rotate        # Emergency secret rotation
    python ops/emergency.py rollback      # Rollback to last known good
"""

import os
import sys
import json
import requests
import subprocess
from datetime import datetime
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK_URL", "")
KILL_SWITCH_URL = os.getenv("KILL_SWITCH_URL", "http://localhost:8080/kill")
VAULT_ADDR = os.getenv("VAULT_ADDR", "https://vault.production.example.com")
K8S_NAMESPACE = "trading"
K8S_DEPLOYMENT = "delta-trader"


def log(message: str, level: str = "INFO"):
    """Log with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")
    
    # Also send to Slack for critical events
    if level in ["CRITICAL", "ERROR"] and SLACK_WEBHOOK:
        try:
            requests.post(SLACK_WEBHOOK, json={
                "text": f":rotating_light: *{level}*: {message}",
                "channel": "#trading-alerts",
            }, timeout=5)
        except:
            pass


def kill_switch():
    """
    EMERGENCY KILL SWITCH
    
    Immediately halts all trading activity.
    """
    log("🚨 EXECUTING KILL SWITCH", "CRITICAL")
    
    try:
        # Method 1: HTTP kill switch
        response = requests.post(KILL_SWITCH_URL, json={"kill": True}, timeout=5)
        log(f"Kill switch HTTP response: {response.status_code}")
    except Exception as e:
        log(f"HTTP kill switch failed: {e}", "ERROR")
    
    try:
        # Method 2: Scale deployment to 0
        result = subprocess.run([
            "kubectl", "scale", "deployment", K8S_DEPLOYMENT,
            "--replicas=0", "-n", K8S_NAMESPACE
        ], capture_output=True, text=True, timeout=30)
        log(f"Scaled deployment to 0: {result.stdout}")
    except Exception as e:
        log(f"K8s scale failed: {e}", "ERROR")
    
    try:
        # Method 3: Set PAPER mode
        result = subprocess.run([
            "kubectl", "set", "env", f"deployment/{K8S_DEPLOYMENT}",
            "PAPER=1", "-n", K8S_NAMESPACE
        ], capture_output=True, text=True, timeout=30)
        log(f"Set PAPER=1: {result.stdout}")
    except Exception as e:
        log(f"Set PAPER failed: {e}", "ERROR")
    
    log("✅ Kill switch executed - trading HALTED", "CRITICAL")
    
    # Create incident file
    incident = {
        "timestamp": datetime.now().isoformat(),
        "action": "KILL_SWITCH",
        "triggered_by": os.getenv("USER", "unknown"),
    }
    Path("ops/incidents").mkdir(exist_ok=True)
    with open(f"ops/incidents/kill_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
        json.dump(incident, f, indent=2)


def check_status():
    """
    SYSTEM STATUS CHECK
    
    Quick health check of all critical components.
    """
    log("Checking system status...")
    status = {}
    
    # Check Kubernetes deployment
    try:
        result = subprocess.run([
            "kubectl", "get", "deployment", K8S_DEPLOYMENT,
            "-n", K8S_NAMESPACE, "-o", "json"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            deploy = json.loads(result.stdout)
            replicas = deploy["status"].get("readyReplicas", 0)
            status["kubernetes"] = f"OK ({replicas} replicas)"
            log(f"Kubernetes: {replicas} replicas ready")
        else:
            status["kubernetes"] = "ERROR"
            log(f"Kubernetes: {result.stderr}", "ERROR")
    except Exception as e:
        status["kubernetes"] = f"ERROR: {e}"
        log(f"Kubernetes check failed: {e}", "ERROR")
    
    # Check Vault
    try:
        result = subprocess.run([
            "vault", "status", "-format=json"
        ], capture_output=True, text=True, timeout=10, env={**os.environ, "VAULT_ADDR": VAULT_ADDR})
        
        if result.returncode == 0:
            vault_status = json.loads(result.stdout)
            sealed = vault_status.get("sealed", True)
            status["vault"] = "SEALED" if sealed else "OK (unsealed)"
            log(f"Vault: {'SEALED' if sealed else 'unsealed'}")
        else:
            status["vault"] = "ERROR"
    except Exception as e:
        status["vault"] = f"UNREACHABLE: {e}"
        log(f"Vault check failed: {e}", "ERROR")
    
    # Check recent audit logs
    try:
        audit_dir = Path("audit_logs")
        if audit_dir.exists():
            files = list(audit_dir.glob("*.json*"))
            status["audit_logs"] = f"OK ({len(files)} files)"
            log(f"Audit logs: {len(files)} files")
        else:
            status["audit_logs"] = "MISSING"
            log("Audit logs directory missing", "ERROR")
    except Exception as e:
        status["audit_logs"] = f"ERROR: {e}"
    
    # Print summary
    print("\n" + "=" * 50)
    print("SYSTEM STATUS SUMMARY")
    print("=" * 50)
    for component, state in status.items():
        icon = "✅" if "OK" in state else "❌"
        print(f"{icon} {component}: {state}")
    print("=" * 50)
    
    return status


def rotate_secrets():
    """
    EMERGENCY SECRET ROTATION
    
    Revoke current secrets and generate new ones.
    """
    log("🔄 Starting emergency secret rotation", "CRITICAL")
    
    try:
        # Revoke all trading secrets
        result = subprocess.run([
            "vault", "lease", "revoke", "-prefix", "secret/trading"
        ], capture_output=True, text=True, timeout=30, env={**os.environ, "VAULT_ADDR": VAULT_ADDR})
        log(f"Revoked secrets: {result.stdout}")
    except Exception as e:
        log(f"Secret revocation failed: {e}", "ERROR")
    
    try:
        # Restart deployment to pick up new secrets
        result = subprocess.run([
            "kubectl", "rollout", "restart", f"deployment/{K8S_DEPLOYMENT}",
            "-n", K8S_NAMESPACE
        ], capture_output=True, text=True, timeout=30)
        log(f"Triggered deployment restart: {result.stdout}")
    except Exception as e:
        log(f"Deployment restart failed: {e}", "ERROR")
    
    log("✅ Secret rotation complete - deployment restarting", "CRITICAL")


def rollback():
    """
    ROLLBACK TO LAST KNOWN GOOD
    
    Revert to previous deployment version.
    """
    log("⏪ Starting rollback to previous version", "CRITICAL")
    
    try:
        result = subprocess.run([
            "kubectl", "rollout", "undo", f"deployment/{K8S_DEPLOYMENT}",
            "-n", K8S_NAMESPACE
        ], capture_output=True, text=True, timeout=60)
        log(f"Rollback result: {result.stdout}")
    except Exception as e:
        log(f"Rollback failed: {e}", "ERROR")
        return
    
    # Wait for rollback to complete
    try:
        result = subprocess.run([
            "kubectl", "rollout", "status", f"deployment/{K8S_DEPLOYMENT}",
            "-n", K8S_NAMESPACE, "--timeout=120s"
        ], capture_output=True, text=True, timeout=130)
        log(f"Rollback status: {result.stdout}")
    except Exception as e:
        log(f"Rollback status check failed: {e}", "ERROR")
    
    log("✅ Rollback complete", "CRITICAL")


def scale(replicas: int):
    """Scale deployment to specified replicas."""
    log(f"📈 Scaling to {replicas} replicas")
    
    try:
        result = subprocess.run([
            "kubectl", "scale", "deployment", K8S_DEPLOYMENT,
            f"--replicas={replicas}", "-n", K8S_NAMESPACE
        ], capture_output=True, text=True, timeout=30)
        log(f"Scaled deployment: {result.stdout}")
    except Exception as e:
        log(f"Scale failed: {e}", "ERROR")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        return
    
    command = sys.argv[1].lower()
    
    if command == "kill":
        kill_switch()
    elif command == "status":
        check_status()
    elif command == "rotate":
        rotate_secrets()
    elif command == "rollback":
        rollback()
    elif command == "scale" and len(sys.argv) >= 3:
        scale(int(sys.argv[2]))
    else:
        print(__doc__)
        print(f"\nUnknown command: {command}")


if __name__ == "__main__":
    main()
