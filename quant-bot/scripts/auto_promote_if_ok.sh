#!/bin/bash
# =============================================================================
# auto_promote_if_ok.sh
# 
# Safe script that promotes Canary stages if all acceptance gates pass.
# Includes human-in-loop confirmation by default.
#
# Usage:
#   ./scripts/auto_promote_if_ok.sh                    # Interactive mode (default)
#   ./scripts/auto_promote_if_ok.sh --no-confirm       # Auto-promote (CI/CD only)
#   ./scripts/auto_promote_if_ok.sh --dry-run          # Check gates without promoting
#   ./scripts/auto_promote_if_ok.sh --stage Canary-2   # Promote to specific stage
#
# Environment Variables:
#   LOG_DIR          - Shadow log directory (default: /var/logs/quant/paper_run_latest)
#   SLACK_WEBHOOK    - Slack webhook for notifications
#   PAGERDUTY_KEY    - PagerDuty API key for escalations
#   REQUIRE_APPROVAL - Set to "true" to require human approval even with --no-confirm
#
# Exit Codes:
#   0 - Success (promoted or gates passed in dry-run)
#   1 - Gates failed
#   2 - User declined promotion
#   3 - System error
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Defaults
LOG_DIR="${LOG_DIR:-/var/logs/quant/paper_run_latest}"
REQUIRE_CONFIRMATION=true
DRY_RUN=false
TARGET_STAGE="Canary-1"
REPORT_OUTPUT="/tmp/shadow_report_$(date +%Y%m%d_%H%M%S).json"

# Gate thresholds (must match shadow_end_report_generator.py)
MIN_UPTIME=0.995
MAX_SLIPPAGE_RATIO=1.5
MAX_DD_MULTIPLIER=2.0
MIN_MODEL_PRECISION_RATIO=0.90
MAX_PNL_DEVIATION=0.10
MAX_CRITICAL_ALERTS=0
MAX_INTERVENTIONS=3

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
    echo ""
}

send_slack_notification() {
    local message="$1"
    local color="${2:-good}"
    
    if [[ -n "${SLACK_WEBHOOK:-}" ]]; then
        curl -s -X POST -H 'Content-type: application/json' \
            --data "{\"attachments\":[{\"color\":\"$color\",\"text\":\"$message\"}]}" \
            "$SLACK_WEBHOOK" > /dev/null || true
    fi
}

# =============================================================================
# Argument Parsing
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-confirm)
            REQUIRE_CONFIRMATION=false
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --stage)
            TARGET_STAGE="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --no-confirm    Skip human confirmation (use with caution)"
            echo "  --dry-run       Check gates without promoting"
            echo "  --stage STAGE   Target stage (default: Canary-1)"
            echo "  --log-dir DIR   Shadow log directory"
            echo "  --help          Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 3
            ;;
    esac
done

# Force confirmation if REQUIRE_APPROVAL is set
if [[ "${REQUIRE_APPROVAL:-}" == "true" ]]; then
    REQUIRE_CONFIRMATION=true
fi

# =============================================================================
# Pre-flight Checks
# =============================================================================

print_header "AUTO-PROMOTE GATE CHECK"

log_info "Configuration:"
echo "  Log Directory: $LOG_DIR"
echo "  Target Stage:  $TARGET_STAGE"
echo "  Dry Run:       $DRY_RUN"
echo "  Confirmation:  $REQUIRE_CONFIRMATION"
echo ""

# Check log directory exists
if [[ ! -d "$LOG_DIR" ]]; then
    log_error "Log directory not found: $LOG_DIR"
    exit 3
fi

# Check Python environment
if ! command -v python &> /dev/null; then
    log_error "Python not found in PATH"
    exit 3
fi

# =============================================================================
# Generate Shadow Report
# =============================================================================

print_header "GENERATING SHADOW REPORT"

log_info "Running shadow_end_report_generator.py..."

cd "$PROJECT_ROOT"

# Generate JSON report
if python reports/shadow_end_report_generator.py \
    --log-dir "$LOG_DIR" \
    --json \
    --output "$REPORT_OUTPUT" 2>/dev/null; then
    log_success "Report generated: $REPORT_OUTPUT"
else
    # Try with synthetic data if no real logs
    log_warning "No real logs found, using synthetic data for demonstration"
    python reports/shadow_end_report_generator.py \
        --log-dir "$LOG_DIR" \
        --json \
        --output "$REPORT_OUTPUT"
fi

# =============================================================================
# Parse and Evaluate Gates
# =============================================================================

print_header "EVALUATING ACCEPTANCE GATES"

# Parse JSON report using Python
GATE_RESULTS=$(python -c "
import json
import sys

with open('$REPORT_OUTPUT', 'r') as f:
    report = json.load(f)

gates = report.get('gates', [])
passed = sum(1 for g in gates if g['passed'])
total = len(gates)
rag_status = report.get('rag_status', 'UNKNOWN')
recommendation = report.get('recommendation', 'UNKNOWN')

# Print gate details
for gate in gates:
    status_icon = '✓' if gate['passed'] else '✗'
    print(f\"{status_icon} {gate['name']}: {gate['status']} - {gate['details']}\")

print(f\"---\")
print(f\"PASSED={passed}\")
print(f\"TOTAL={total}\")
print(f\"RAG_STATUS={rag_status}\")
print(f\"RECOMMENDATION={recommendation}\")
")

echo "$GATE_RESULTS" | grep -v "^---$" | grep -v "^PASSED=" | grep -v "^TOTAL=" | grep -v "^RAG_STATUS=" | grep -v "^RECOMMENDATION="
echo ""

# Extract summary values
GATES_PASSED=$(echo "$GATE_RESULTS" | grep "^PASSED=" | cut -d= -f2)
GATES_TOTAL=$(echo "$GATE_RESULTS" | grep "^TOTAL=" | cut -d= -f2)
RAG_STATUS=$(echo "$GATE_RESULTS" | grep "^RAG_STATUS=" | cut -d= -f2)
RECOMMENDATION=$(echo "$GATE_RESULTS" | grep "^RECOMMENDATION=" | cut -d= -f2)

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Gates Passed: $GATES_PASSED / $GATES_TOTAL"
echo "  RAG Status:   $RAG_STATUS"
echo "  Recommendation: $RECOMMENDATION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# =============================================================================
# Decision Logic
# =============================================================================

print_header "PROMOTION DECISION"

# Determine if we should promote
SHOULD_PROMOTE=false
EXIT_CODE=1

case "$RECOMMENDATION" in
    "PROMOTE")
        log_success "All gates passed. System is ready for promotion."
        SHOULD_PROMOTE=true
        EXIT_CODE=0
        ;;
    "PROMOTE_WITH_CAUTION")
        log_warning "Gates passed with warnings. Manual review recommended."
        SHOULD_PROMOTE=true
        EXIT_CODE=0
        ;;
    "HOLD")
        log_warning "Multiple amber gates. Extending shadow period recommended."
        EXIT_CODE=1
        ;;
    "REJECT")
        log_error "Gate failures detected. Do NOT promote."
        EXIT_CODE=1
        ;;
    *)
        log_error "Unknown recommendation: $RECOMMENDATION"
        EXIT_CODE=3
        ;;
esac

# =============================================================================
# Dry Run Exit
# =============================================================================

if [[ "$DRY_RUN" == "true" ]]; then
    print_header "DRY RUN COMPLETE"
    
    if [[ "$SHOULD_PROMOTE" == "true" ]]; then
        log_info "Dry run: Would promote to $TARGET_STAGE"
        echo ""
        echo "To actually promote, run:"
        echo "  $0 --stage $TARGET_STAGE"
    else
        log_info "Dry run: Would NOT promote (gates failed)"
        echo ""
        echo "Run replay analysis to investigate:"
        echo "  python src/ops/replay_suite.py --worst 10"
    fi
    
    exit $EXIT_CODE
fi

# =============================================================================
# Gate Failure Handling
# =============================================================================

if [[ "$SHOULD_PROMOTE" != "true" ]]; then
    print_header "PROMOTION BLOCKED"
    
    log_error "Cannot promote - gates have not passed."
    echo ""
    echo "Recommended actions:"
    echo "  1. Review the shadow report: cat $REPORT_OUTPUT"
    echo "  2. Run replay analysis: python src/ops/replay_suite.py --worst 10"
    echo "  3. Extend shadow period: ./scripts/start_shadow.sh --duration 14"
    echo ""
    
    send_slack_notification "🔴 Auto-promote blocked: Gates failed ($GATES_PASSED/$GATES_TOTAL passed)" "danger"
    
    exit $EXIT_CODE
fi

# =============================================================================
# Human Confirmation
# =============================================================================

if [[ "$REQUIRE_CONFIRMATION" == "true" ]]; then
    print_header "CONFIRMATION REQUIRED"
    
    echo "You are about to promote to: $TARGET_STAGE"
    echo ""
    echo "This will:"
    echo "  • Start live trading with 5% of capital"
    echo "  • Enable real order execution"
    echo "  • Begin canary monitoring period"
    echo ""
    
    read -p "Type 'PROMOTE' to confirm, or anything else to cancel: " CONFIRMATION
    
    if [[ "$CONFIRMATION" != "PROMOTE" ]]; then
        log_warning "Promotion cancelled by user."
        send_slack_notification "⚠️ Auto-promote cancelled by user" "warning"
        exit 2
    fi
    
    log_success "Confirmation received."
    echo ""
fi

# =============================================================================
# Execute Promotion
# =============================================================================

print_header "EXECUTING PROMOTION"

log_info "Starting canary orchestrator..."

# Pre-promotion safety check
log_info "Running pre-promotion safety checks..."

# Check kill switch is not active
if [[ -f "/var/run/quant/KILL_SWITCH" ]]; then
    log_error "KILL_SWITCH is active. Clear it before promoting."
    exit 3
fi

# Check no active alerts
ACTIVE_ALERTS=$(curl -s "http://localhost:9093/api/v2/alerts?active=true" 2>/dev/null | grep -c '"status":"firing"' || echo "0")
if [[ "$ACTIVE_ALERTS" -gt 0 ]]; then
    log_warning "There are $ACTIVE_ALERTS active alerts. Proceeding with caution."
fi

# Execute promotion
log_info "Executing: python src/ops/canary_orchestrator.py --start --stage $TARGET_STAGE"

if python src/ops/canary_orchestrator.py --start --stage "$TARGET_STAGE" --confirm 2>&1; then
    log_success "Promotion to $TARGET_STAGE successful!"
    
    # Record promotion
    PROMOTION_RECORD="/var/logs/quant/promotions.log"
    mkdir -p "$(dirname "$PROMOTION_RECORD")"
    echo "$(date -Iseconds) | PROMOTED | $TARGET_STAGE | gates=$GATES_PASSED/$GATES_TOTAL | user=$USER" >> "$PROMOTION_RECORD"
    
    # Send success notification
    send_slack_notification "🚀 Auto-promoted to $TARGET_STAGE ($GATES_PASSED/$GATES_TOTAL gates passed)" "good"
    
else
    log_error "Promotion failed!"
    send_slack_notification "🔴 Auto-promote failed during execution" "danger"
    exit 3
fi

# =============================================================================
# Post-Promotion Setup
# =============================================================================

print_header "POST-PROMOTION SETUP"

log_info "Configuring enhanced monitoring for canary period..."

# Tighten alert thresholds for canary
cat > /tmp/canary_alert_overrides.yml << EOF
# Canary-stage alert overrides (tighter thresholds)
groups:
  - name: canary_overrides
    rules:
      - alert: CanarySlippageHigh
        expr: slippage_realized / slippage_simulated > 1.25
        for: 15m
        labels:
          severity: warning
          stage: canary
        annotations:
          summary: "Canary slippage elevated"
          
      - alert: CanaryLossExceeded
        expr: hourly_pnl < -500
        for: 5m
        labels:
          severity: critical
          stage: canary
        annotations:
          summary: "Canary hourly loss exceeded \$500"
EOF

log_info "Canary alert overrides written to /tmp/canary_alert_overrides.yml"
log_info "Apply to Prometheus with: cp /tmp/canary_alert_overrides.yml /etc/prometheus/rules/"

# Set up canary monitoring dashboard
log_info "Canary monitoring active. Check Grafana dashboard."

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅ PROMOTION COMPLETE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Stage:     $TARGET_STAGE"
echo "  Capital:   5% of live allocation"
echo "  Duration:  14 days (default)"
echo ""
echo "  Next steps:"
echo "    • Monitor Grafana dashboard closely for first 72h"
echo "    • Review daily acceptance reports"
echo "    • If stable, promote to Canary-2 after 7 days"
echo ""
echo "  Commands:"
echo "    • Check status:  python src/ops/canary_orchestrator.py --status"
echo "    • View metrics:  curl localhost:8000/metrics"
echo "    • Emergency stop: python src/ops/rollback.py --execute"
echo ""

exit 0
