#!/bin/bash
###############################################################################
# SMOKE TEST SUITE
# Validates all critical wiring before shadow/production deployment
# Run: ./scripts/smoke_test.sh
###############################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${PROJECT_ROOT}/logs/smoke_tests"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/smoke_test_${TIMESTAMP}.log"

# Create log directory
mkdir -p "$LOG_DIR"

# Initialize counters
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

###############################################################################
# Utility Functions
###############################################################################

log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

header() {
    log "\n${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    log "${BLUE}  $1${NC}"
    log "${BLUE}═══════════════════════════════════════════════════════════════${NC}\n"
}

pass() {
    log "${GREEN}✅ PASS:${NC} $1"
    ((TESTS_PASSED++))
}

fail() {
    log "${RED}❌ FAIL:${NC} $1"
    ((TESTS_FAILED++))
}

skip() {
    log "${YELLOW}⏭️  SKIP:${NC} $1"
    ((TESTS_SKIPPED++))
}

warn() {
    log "${YELLOW}⚠️  WARN:${NC} $1"
}

###############################################################################
# Environment Check
###############################################################################

check_environment() {
    header "1. ENVIRONMENT VALIDATION"
    
    # Check Python
    if command -v python &> /dev/null; then
        PYTHON_VERSION=$(python --version 2>&1)
        pass "Python installed: $PYTHON_VERSION"
    else
        fail "Python not found"
        return 1
    fi
    
    # Check required env vars
    REQUIRED_VARS=(
        "TRADING_SERVICE_URL"
        "TRADING_API_KEY"
        "TRADING_API_SECRET"
    )
    
    OPTIONAL_VARS=(
        "SLACK_WEBHOOK_URL"
        "DISCORD_WEBHOOK_URL"
        "PAGERDUTY_INTEGRATION_KEY"
        "PROMETHEUS_PUSHGATEWAY_URL"
        "GRAFANA_API_KEY"
        "MODEL_REGISTRY_URL"
        "S3_ARTIFACT_BUCKET"
    )
    
    for var in "${REQUIRED_VARS[@]}"; do
        if [[ -n "${!var}" ]]; then
            pass "Required env var set: $var"
        else
            fail "Required env var missing: $var"
        fi
    done
    
    for var in "${OPTIONAL_VARS[@]}"; do
        if [[ -n "${!var}" ]]; then
            pass "Optional env var set: $var"
        else
            warn "Optional env var not set: $var (some features disabled)"
        fi
    done
}

###############################################################################
# Module Import Tests
###############################################################################

test_module_imports() {
    header "2. MODULE IMPORT TESTS"
    
    cd "$PROJECT_ROOT"
    
    # Test core modules
    MODULES=(
        "src.options.pricing_engine"
        "src.options.volatility_surface"
        "src.options.strategies"
        "src.options.risk_engine"
        "src.arbitrage.funding_arbitrage"
        "src.arbitrage.statistical_arbitrage"
        "src.arbitrage.cross_exchange"
        "src.risk.circuit_breaker"
        "src.risk.dynamic_sizing"
        "src.execution.smart_router"
        "src.ml.online_learning"
        "src.ml.feature_monitor"
    )
    
    for module in "${MODULES[@]}"; do
        if python -c "import $module" 2>/dev/null; then
            pass "Import: $module"
        else
            fail "Import failed: $module"
        fi
    done
}

###############################################################################
# Ops Scripts Tests
###############################################################################

test_ops_scripts() {
    header "3. OPS SCRIPTS VALIDATION"
    
    cd "$PROJECT_ROOT"
    
    # Test rollback script
    log "\n${YELLOW}Testing rollback.py...${NC}"
    if python src/ops/rollback.py --status 2>&1 | tee -a "$LOG_FILE"; then
        pass "Rollback status check"
    else
        fail "Rollback status check"
    fi
    
    # Test rollback dry-run
    if python src/ops/rollback.py --dry-run --target-version v1.0.0 2>&1 | tee -a "$LOG_FILE"; then
        pass "Rollback dry-run"
    else
        fail "Rollback dry-run"
    fi
    
    # Test canary orchestrator dry-run
    log "\n${YELLOW}Testing canary_orchestrator.py...${NC}"
    if python src/ops/canary_orchestrator.py --dry-run --stage Canary-1 2>&1 | tee -a "$LOG_FILE"; then
        pass "Canary orchestrator dry-run"
    else
        fail "Canary orchestrator dry-run"
    fi
    
    # Test daily report generator
    log "\n${YELLOW}Testing daily_report_generator.py...${NC}"
    REPORT_OUTPUT="/tmp/smoke_test_daily_report_${TIMESTAMP}.md"
    if python src/ops/daily_report_generator.py --sample --output "$REPORT_OUTPUT" 2>&1 | tee -a "$LOG_FILE"; then
        if [[ -f "$REPORT_OUTPUT" ]]; then
            pass "Daily report generator (output: $REPORT_OUTPUT)"
            # Show first 20 lines
            log "\n${BLUE}Report preview:${NC}"
            head -20 "$REPORT_OUTPUT" | tee -a "$LOG_FILE"
        else
            fail "Daily report generator (no output file)"
        fi
    else
        fail "Daily report generator"
    fi
    
    # Test replay suite dry-run
    log "\n${YELLOW}Testing replay_suite.py...${NC}"
    if python src/ops/replay_suite.py --by-id "SMOKE-TEST-001" --dry-run 2>&1 | tee -a "$LOG_FILE"; then
        pass "Replay suite dry-run"
    else
        fail "Replay suite dry-run"
    fi
}

###############################################################################
# Prometheus Alert Tests
###############################################################################

test_prometheus_alerts() {
    header "4. PROMETHEUS ALERT TESTS"
    
    if [[ -z "$PROMETHEUS_PUSHGATEWAY_URL" ]]; then
        skip "Prometheus tests (PROMETHEUS_PUSHGATEWAY_URL not set)"
        return
    fi
    
    # Test pushgateway connectivity
    log "\n${YELLOW}Testing Prometheus Pushgateway connectivity...${NC}"
    if curl -s -o /dev/null -w "%{http_code}" "$PROMETHEUS_PUSHGATEWAY_URL" | grep -q "200\|204"; then
        pass "Prometheus Pushgateway reachable"
    else
        fail "Prometheus Pushgateway not reachable"
        return
    fi
    
    # Push test metric
    log "\n${YELLOW}Pushing test metric...${NC}"
    TEST_METRIC="smoke_test_timestamp $(date +%s)"
    if curl -s -X POST "${PROMETHEUS_PUSHGATEWAY_URL}/metrics/job/smoke_test" -d "$TEST_METRIC"; then
        pass "Test metric pushed to Pushgateway"
    else
        fail "Failed to push test metric"
    fi
}

###############################################################################
# Slack/Discord Notification Tests
###############################################################################

test_notifications() {
    header "5. NOTIFICATION CHANNEL TESTS"
    
    # Test Slack
    if [[ -n "$SLACK_WEBHOOK_URL" ]]; then
        log "\n${YELLOW}Testing Slack webhook...${NC}"
        SLACK_PAYLOAD='{"text":"🧪 Smoke Test: Notification channel verified at '"$(date -Iseconds)"'"}'
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST -H 'Content-type: application/json' \
            --data "$SLACK_PAYLOAD" "$SLACK_WEBHOOK_URL")
        if [[ "$HTTP_CODE" == "200" ]]; then
            pass "Slack webhook"
        else
            fail "Slack webhook (HTTP $HTTP_CODE)"
        fi
    else
        skip "Slack test (SLACK_WEBHOOK_URL not set)"
    fi
    
    # Test Discord
    if [[ -n "$DISCORD_WEBHOOK_URL" ]]; then
        log "\n${YELLOW}Testing Discord webhook...${NC}"
        DISCORD_PAYLOAD='{"content":"🧪 Smoke Test: Notification channel verified at '"$(date -Iseconds)"'"}'
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST -H 'Content-type: application/json' \
            --data "$DISCORD_PAYLOAD" "$DISCORD_WEBHOOK_URL")
        if [[ "$HTTP_CODE" == "204" ]]; then
            pass "Discord webhook"
        else
            fail "Discord webhook (HTTP $HTTP_CODE)"
        fi
    else
        skip "Discord test (DISCORD_WEBHOOK_URL not set)"
    fi
}

###############################################################################
# Grafana Dashboard Tests
###############################################################################

test_grafana() {
    header "6. GRAFANA DASHBOARD TESTS"
    
    if [[ -z "$GRAFANA_API_KEY" ]] || [[ -z "$GRAFANA_URL" ]]; then
        skip "Grafana tests (GRAFANA_API_KEY or GRAFANA_URL not set)"
        return
    fi
    
    # Test Grafana API
    log "\n${YELLOW}Testing Grafana API...${NC}"
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $GRAFANA_API_KEY" \
        "$GRAFANA_URL/api/health")
    if [[ "$HTTP_CODE" == "200" ]]; then
        pass "Grafana API reachable"
    else
        fail "Grafana API (HTTP $HTTP_CODE)"
    fi
    
    # Check if dashboard exists
    DASHBOARD_FILE="${PROJECT_ROOT}/monitoring/grafana_production_dashboard.json"
    if [[ -f "$DASHBOARD_FILE" ]]; then
        pass "Grafana dashboard JSON exists"
    else
        fail "Grafana dashboard JSON not found"
    fi
}

###############################################################################
# Unit Tests
###############################################################################

run_unit_tests() {
    header "7. UNIT TESTS"
    
    cd "$PROJECT_ROOT"
    
    # Run pytest if available
    if command -v pytest &> /dev/null; then
        log "\n${YELLOW}Running pytest...${NC}"
        if pytest tests/ -v --tb=short 2>&1 | tee -a "$LOG_FILE"; then
            pass "Unit tests"
        else
            fail "Unit tests (see log for details)"
        fi
    else
        # Fall back to test script
        log "\n${YELLOW}Running test_options_arbitrage.py...${NC}"
        if python test_options_arbitrage.py 2>&1 | tee -a "$LOG_FILE"; then
            pass "Options & Arbitrage tests"
        else
            fail "Options & Arbitrage tests"
        fi
    fi
}

###############################################################################
# API Connectivity Tests
###############################################################################

test_api_connectivity() {
    header "8. TRADING API CONNECTIVITY"
    
    if [[ -z "$TRADING_SERVICE_URL" ]]; then
        skip "API connectivity (TRADING_SERVICE_URL not set)"
        return
    fi
    
    # Test API health endpoint
    log "\n${YELLOW}Testing Trading API...${NC}"
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 "$TRADING_SERVICE_URL/health" 2>/dev/null || echo "000")
    
    if [[ "$HTTP_CODE" == "200" ]]; then
        pass "Trading API health check"
    elif [[ "$HTTP_CODE" == "000" ]]; then
        warn "Trading API not reachable (may be expected for paper trading)"
        skip "Trading API connectivity"
    else
        fail "Trading API health check (HTTP $HTTP_CODE)"
    fi
}

###############################################################################
# Summary
###############################################################################

print_summary() {
    header "SMOKE TEST SUMMARY"
    
    TOTAL=$((TESTS_PASSED + TESTS_FAILED + TESTS_SKIPPED))
    
    log "Total Tests: $TOTAL"
    log "${GREEN}Passed: $TESTS_PASSED${NC}"
    log "${RED}Failed: $TESTS_FAILED${NC}"
    log "${YELLOW}Skipped: $TESTS_SKIPPED${NC}"
    log "\nLog file: $LOG_FILE"
    
    if [[ $TESTS_FAILED -eq 0 ]]; then
        log "\n${GREEN}════════════════════════════════════════════════════════════════${NC}"
        log "${GREEN}  🎉 ALL SMOKE TESTS PASSED - READY FOR SHADOW DEPLOYMENT${NC}"
        log "${GREEN}════════════════════════════════════════════════════════════════${NC}\n"
        return 0
    else
        log "\n${RED}════════════════════════════════════════════════════════════════${NC}"
        log "${RED}  ❌ SMOKE TESTS FAILED - FIX ISSUES BEFORE DEPLOYMENT${NC}"
        log "${RED}════════════════════════════════════════════════════════════════${NC}\n"
        return 1
    fi
}

###############################################################################
# Main Execution
###############################################################################

main() {
    log "Smoke Test Started: $(date -Iseconds)"
    log "Project Root: $PROJECT_ROOT"
    log "Log File: $LOG_FILE\n"
    
    check_environment
    test_module_imports
    test_ops_scripts
    test_prometheus_alerts
    test_notifications
    test_grafana
    run_unit_tests
    test_api_connectivity
    
    print_summary
}

# Run main
main "$@"
