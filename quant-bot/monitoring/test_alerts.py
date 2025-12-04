"""
Prometheus Alert Testing Utility
=================================
Push synthetic metrics to Prometheus Pushgateway to trigger specific alerts
for verification and testing purposes.

Usage:
    python monitoring/test_alerts.py --alert pnl_critical
    python monitoring/test_alerts.py --alert slippage_spike
    python monitoring/test_alerts.py --all
    python monitoring/test_alerts.py --list
    python monitoring/test_alerts.py --clear

Author: Quant Bot Ops
Version: 1.0.0
"""

import os
import sys
import argparse
import requests
import time
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class TestAlert:
    """Definition of a test alert."""
    name: str
    description: str
    severity: AlertSeverity
    metrics: Dict[str, float]
    expected_behavior: str
    reset_metrics: Dict[str, float]


# Registry of test alerts
TEST_ALERTS: Dict[str, TestAlert] = {
    # P&L Alerts
    "pnl_critical": TestAlert(
        name="PnL Critical Loss",
        description="Simulates critical P&L loss triggering kill switch",
        severity=AlertSeverity.CRITICAL,
        metrics={
            "strategy_net_pnl_total": -1000000,
            "strategy_realized_pnl": -500000,
            "strategy_unrealized_pnl": -500000,
        },
        expected_behavior="KILL_SWITCH activation, PagerDuty alert, Slack critical notification",
        reset_metrics={
            "strategy_net_pnl_total": 50000,
            "strategy_realized_pnl": 30000,
            "strategy_unrealized_pnl": 20000,
        }
    ),
    
    "pnl_warning": TestAlert(
        name="PnL Warning",
        description="Simulates P&L warning threshold breach",
        severity=AlertSeverity.WARNING,
        metrics={
            "strategy_net_pnl_total": -50000,
            "strategy_drawdown_percent": 0.08,
        },
        expected_behavior="Slack warning notification, position size reduction",
        reset_metrics={
            "strategy_net_pnl_total": 10000,
            "strategy_drawdown_percent": 0.02,
        }
    ),
    
    # Execution Alerts
    "slippage_spike": TestAlert(
        name="Slippage Spike",
        description="Simulates excessive slippage on executions",
        severity=AlertSeverity.WARNING,
        metrics={
            "execution_slippage_bps": 150,
            "execution_slippage_realized": 0.015,
            "execution_fill_rate": 0.6,
        },
        expected_behavior="Slack warning, execution throttling, smart router adjustment",
        reset_metrics={
            "execution_slippage_bps": 10,
            "execution_slippage_realized": 0.001,
            "execution_fill_rate": 0.95,
        }
    ),
    
    "fill_rate_low": TestAlert(
        name="Low Fill Rate",
        description="Simulates poor order fill rates",
        severity=AlertSeverity.WARNING,
        metrics={
            "execution_fill_rate": 0.4,
            "execution_partial_fills": 50,
            "execution_rejected_orders": 20,
        },
        expected_behavior="Slack warning, order sizing review",
        reset_metrics={
            "execution_fill_rate": 0.95,
            "execution_partial_fills": 2,
            "execution_rejected_orders": 0,
        }
    ),
    
    # Model Alerts
    "model_degradation": TestAlert(
        name="Model Degradation",
        description="Simulates ML model performance degradation",
        severity=AlertSeverity.WARNING,
        metrics={
            "ml_meta_model_precision": 0.45,
            "ml_meta_model_recall": 0.50,
            "ml_meta_model_f1": 0.47,
            "ml_prediction_accuracy": 0.48,
        },
        expected_behavior="Slack warning, model retraining trigger, fallback to baseline",
        reset_metrics={
            "ml_meta_model_precision": 0.85,
            "ml_meta_model_recall": 0.82,
            "ml_meta_model_f1": 0.83,
            "ml_prediction_accuracy": 0.80,
        }
    ),
    
    "feature_drift": TestAlert(
        name="Feature Drift",
        description="Simulates significant feature distribution drift",
        severity=AlertSeverity.WARNING,
        metrics={
            "ml_feature_psi_max": 0.35,
            "ml_feature_drift_count": 5,
            "ml_data_quality_score": 0.6,
        },
        expected_behavior="Slack warning, feature investigation, potential model pause",
        reset_metrics={
            "ml_feature_psi_max": 0.05,
            "ml_feature_drift_count": 0,
            "ml_data_quality_score": 0.95,
        }
    ),
    
    # Risk Alerts
    "circuit_breaker": TestAlert(
        name="Circuit Breaker Triggered",
        description="Simulates circuit breaker activation",
        severity=AlertSeverity.CRITICAL,
        metrics={
            "risk_circuit_breaker_status": 1,
            "risk_consecutive_losses": 10,
            "risk_loss_rate_1h": 0.15,
        },
        expected_behavior="KILL_SWITCH activation, all trading halted, PagerDuty critical",
        reset_metrics={
            "risk_circuit_breaker_status": 0,
            "risk_consecutive_losses": 0,
            "risk_loss_rate_1h": 0.01,
        }
    ),
    
    "position_limit": TestAlert(
        name="Position Limit Breach",
        description="Simulates position limit breach",
        severity=AlertSeverity.WARNING,
        metrics={
            "risk_position_utilization": 0.98,
            "risk_gross_exposure": 1500000,
            "risk_concentration_max": 0.45,
        },
        expected_behavior="Slack warning, position reduction, new trades blocked",
        reset_metrics={
            "risk_position_utilization": 0.5,
            "risk_gross_exposure": 500000,
            "risk_concentration_max": 0.15,
        }
    ),
    
    # Order Flow Alerts
    "orderflow_gating": TestAlert(
        name="Order Flow Gating High Block Rate",
        description="Simulates high order flow gating block rate",
        severity=AlertSeverity.WARNING,
        metrics={
            "orderflow_gate_block_rate": 0.75,
            "orderflow_toxic_flow_ratio": 0.4,
            "orderflow_adverse_selection": 0.3,
        },
        expected_behavior="Slack warning, order flow analysis, potential strategy pause",
        reset_metrics={
            "orderflow_gate_block_rate": 0.1,
            "orderflow_toxic_flow_ratio": 0.05,
            "orderflow_adverse_selection": 0.02,
        }
    ),
    
    # Infrastructure Alerts
    "latency_spike": TestAlert(
        name="Latency Spike",
        description="Simulates high latency to exchange",
        severity=AlertSeverity.WARNING,
        metrics={
            "infra_exchange_latency_ms": 500,
            "infra_api_error_rate": 0.15,
            "infra_websocket_reconnects": 10,
        },
        expected_behavior="Slack warning, latency investigation, potential failover",
        reset_metrics={
            "infra_exchange_latency_ms": 50,
            "infra_api_error_rate": 0.001,
            "infra_websocket_reconnects": 0,
        }
    ),
    
    "memory_pressure": TestAlert(
        name="Memory Pressure",
        description="Simulates high memory usage",
        severity=AlertSeverity.WARNING,
        metrics={
            "infra_memory_usage_percent": 0.92,
            "infra_gc_pause_ms": 500,
        },
        expected_behavior="Slack warning, memory investigation, potential restart",
        reset_metrics={
            "infra_memory_usage_percent": 0.5,
            "infra_gc_pause_ms": 10,
        }
    ),
    
    # Options/Greeks Alerts
    "gamma_exposure": TestAlert(
        name="High Gamma Exposure",
        description="Simulates excessive gamma exposure in options portfolio",
        severity=AlertSeverity.WARNING,
        metrics={
            "options_portfolio_gamma": 0.08,
            "options_gamma_pnl_1pct": 80000,
            "options_delta_exposure": 0.15,
        },
        expected_behavior="Slack warning, gamma hedging trigger, position review",
        reset_metrics={
            "options_portfolio_gamma": 0.02,
            "options_gamma_pnl_1pct": 10000,
            "options_delta_exposure": 0.05,
        }
    ),
    
    "vega_exposure": TestAlert(
        name="High Vega Exposure",
        description="Simulates excessive vega exposure before event",
        severity=AlertSeverity.WARNING,
        metrics={
            "options_portfolio_vega": 25000,
            "options_iv_rank": 0.95,
            "options_vol_crush_risk": 0.25,
        },
        expected_behavior="Slack warning, vega reduction, vol crush protection",
        reset_metrics={
            "options_portfolio_vega": 5000,
            "options_iv_rank": 0.5,
            "options_vol_crush_risk": 0.05,
        }
    ),
}


class AlertTester:
    """Test alert triggering via Prometheus Pushgateway."""
    
    def __init__(self, pushgateway_url: Optional[str] = None):
        self.pushgateway_url = pushgateway_url or os.getenv(
            "PROMETHEUS_PUSHGATEWAY_URL",
            "http://localhost:9091"
        )
        self.job_name = "alert_test"
        
    def push_metrics(self, metrics: Dict[str, float], labels: Optional[Dict[str, str]] = None) -> bool:
        """Push metrics to Pushgateway."""
        labels = labels or {}
        labels["test"] = "true"
        labels["timestamp"] = datetime.now().isoformat()
        
        # Build Prometheus text format
        lines = []
        for metric_name, value in metrics.items():
            # Add labels
            label_str = ",".join([f'{k}="{v}"' for k, v in labels.items()])
            if label_str:
                lines.append(f"{metric_name}{{{label_str}}} {value}")
            else:
                lines.append(f"{metric_name} {value}")
        
        payload = "\n".join(lines) + "\n"
        
        try:
            url = f"{self.pushgateway_url}/metrics/job/{self.job_name}"
            response = requests.post(
                url,
                data=payload,
                headers={"Content-Type": "text/plain"},
                timeout=10
            )
            return response.status_code in (200, 202, 204)
        except Exception as e:
            print(f"Failed to push metrics: {e}")
            return False
    
    def clear_metrics(self) -> bool:
        """Clear all test metrics from Pushgateway."""
        try:
            url = f"{self.pushgateway_url}/metrics/job/{self.job_name}"
            response = requests.delete(url, timeout=10)
            return response.status_code in (200, 202, 204)
        except Exception as e:
            print(f"Failed to clear metrics: {e}")
            return False
    
    def trigger_alert(self, alert_name: str, hold_seconds: int = 30) -> bool:
        """Trigger a specific alert and optionally hold it."""
        if alert_name not in TEST_ALERTS:
            print(f"Unknown alert: {alert_name}")
            return False
        
        alert = TEST_ALERTS[alert_name]
        
        print(f"\n{'='*60}")
        print(f"TRIGGERING ALERT: {alert.name}")
        print(f"{'='*60}")
        print(f"Description: {alert.description}")
        print(f"Severity: {alert.severity.value}")
        print(f"Expected: {alert.expected_behavior}")
        print(f"\nMetrics being pushed:")
        for metric, value in alert.metrics.items():
            print(f"  {metric}: {value}")
        
        # Push alert-triggering metrics
        success = self.push_metrics(
            alert.metrics,
            labels={"alert_name": alert_name, "severity": alert.severity.value}
        )
        
        if success:
            print(f"\n✅ Metrics pushed successfully")
            print(f"⏱️  Holding for {hold_seconds} seconds to allow alert to fire...")
            
            # Wait for alert to fire
            for i in range(hold_seconds, 0, -5):
                print(f"   {i} seconds remaining...")
                time.sleep(5)
            
            print(f"\n🔄 Resetting metrics to normal values...")
            self.push_metrics(
                alert.reset_metrics,
                labels={"alert_name": alert_name, "severity": "normal"}
            )
            print(f"✅ Alert test complete")
            return True
        else:
            print(f"\n❌ Failed to push metrics")
            return False
    
    def trigger_all(self, hold_seconds: int = 15) -> Dict[str, bool]:
        """Trigger all alerts sequentially."""
        results = {}
        
        print("\n" + "="*60)
        print("TRIGGERING ALL ALERTS")
        print("="*60)
        print(f"Total alerts: {len(TEST_ALERTS)}")
        print(f"Hold time per alert: {hold_seconds}s")
        print(f"Estimated total time: {len(TEST_ALERTS) * hold_seconds}s")
        print("="*60)
        
        for alert_name in TEST_ALERTS:
            results[alert_name] = self.trigger_alert(alert_name, hold_seconds)
            time.sleep(2)  # Brief pause between alerts
        
        # Summary
        print("\n" + "="*60)
        print("ALERT TEST SUMMARY")
        print("="*60)
        passed = sum(1 for v in results.values() if v)
        failed = len(results) - passed
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        
        for alert_name, success in results.items():
            status = "✅" if success else "❌"
            print(f"  {status} {alert_name}")
        
        return results
    
    def list_alerts(self):
        """List all available test alerts."""
        print("\n" + "="*60)
        print("AVAILABLE TEST ALERTS")
        print("="*60)
        
        for severity in AlertSeverity:
            alerts = [a for a in TEST_ALERTS.values() if a.severity == severity]
            if alerts:
                print(f"\n{severity.value.upper()} ({len(alerts)}):")
                print("-" * 40)
                for alert in alerts:
                    name = [k for k, v in TEST_ALERTS.items() if v == alert][0]
                    print(f"  {name}")
                    print(f"    {alert.description}")
        
        print("\n" + "="*60)
        print(f"Total: {len(TEST_ALERTS)} alerts")


def main():
    parser = argparse.ArgumentParser(
        description="Prometheus Alert Testing Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python monitoring/test_alerts.py --list
  python monitoring/test_alerts.py --alert pnl_critical
  python monitoring/test_alerts.py --alert slippage_spike --hold 60
  python monitoring/test_alerts.py --all --hold 10
  python monitoring/test_alerts.py --clear
        """
    )
    
    parser.add_argument(
        "--alert",
        help="Trigger a specific alert by name"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Trigger all alerts sequentially"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available test alerts"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all test metrics from Pushgateway"
    )
    parser.add_argument(
        "--hold",
        type=int,
        default=30,
        help="Seconds to hold alert-triggering metrics (default: 30)"
    )
    parser.add_argument(
        "--pushgateway-url",
        help="Prometheus Pushgateway URL (default: env PROMETHEUS_PUSHGATEWAY_URL)"
    )
    
    args = parser.parse_args()
    
    tester = AlertTester(args.pushgateway_url)
    
    if args.list:
        tester.list_alerts()
    elif args.clear:
        print("Clearing test metrics from Pushgateway...")
        if tester.clear_metrics():
            print("✅ Metrics cleared")
        else:
            print("❌ Failed to clear metrics")
            sys.exit(1)
    elif args.all:
        results = tester.trigger_all(args.hold)
        if not all(results.values()):
            sys.exit(1)
    elif args.alert:
        if not tester.trigger_alert(args.alert, args.hold):
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
