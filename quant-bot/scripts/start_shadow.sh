#!/bin/bash
###############################################################################
# START SHADOW TRADING
# Wrapper script to start 30-day shadow run with proper logging and validation
# Run: ./scripts/start_shadow.sh [--duration DAYS] [--dry-run]
###############################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Default configuration
SHADOW_DURATION=30
DRY_RUN=false
LOG_DIR="${PROJECT_ROOT}/logs/shadow_runs"
RUN_ID="shadow_${TIMESTAMP}"

###############################################################################
# Parse Arguments
###############################################################################

while [[ $# -gt 0 ]]; do
    case $1 in
        --duration)
            SHADOW_DURATION="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --duration DAYS    Shadow run duration (default: 30)"
            echo "  --dry-run          Validate without starting"
            echo "  --log-dir DIR      Log directory"
            echo "  --help             Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

###############################################################################
# Banner
###############################################################################

echo -e "${CYAN}"
cat << 'EOF'
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║   ███████╗██╗  ██╗ █████╗ ██████╗  ██████╗ ██╗    ██╗                    ║
║   ██╔════╝██║  ██║██╔══██╗██╔══██╗██╔═══██╗██║    ██║                    ║
║   ███████╗███████║███████║██║  ██║██║   ██║██║ █╗ ██║                    ║
║   ╚════██║██╔══██║██╔══██║██║  ██║██║   ██║██║███╗██║                    ║
║   ███████║██║  ██║██║  ██║██████╔╝╚██████╔╝╚███╔███╔╝                    ║
║   ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝  ╚═════╝  ╚══╝╚══╝                     ║
║                                                                           ║
║              PAPER TRADING ORCHESTRATOR - SHADOW MODE                     ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo -e "${BLUE}Run ID:${NC} $RUN_ID"
echo -e "${BLUE}Duration:${NC} $SHADOW_DURATION days"
echo -e "${BLUE}Dry Run:${NC} $DRY_RUN"
echo -e "${BLUE}Log Dir:${NC} $LOG_DIR"
echo ""

###############################################################################
# Pre-flight Checks
###############################################################################

preflight_checks() {
    echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}  PRE-FLIGHT CHECKS${NC}"
    echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
    
    CHECKS_PASSED=true
    
    # Check required environment variables
    echo -e "\n${BLUE}Checking environment variables...${NC}"
    
    REQUIRED_VARS=(
        "TRADING_API_KEY"
        "TRADING_API_SECRET"
    )
    
    for var in "${REQUIRED_VARS[@]}"; do
        if [[ -n "${!var}" ]]; then
            echo -e "  ${GREEN}✓${NC} $var is set"
        else
            echo -e "  ${RED}✗${NC} $var is NOT set"
            CHECKS_PASSED=false
        fi
    done
    
    # Check optional but recommended
    OPTIONAL_VARS=(
        "SLACK_WEBHOOK_URL"
        "PROMETHEUS_PUSHGATEWAY_URL"
    )
    
    for var in "${OPTIONAL_VARS[@]}"; do
        if [[ -n "${!var}" ]]; then
            echo -e "  ${GREEN}✓${NC} $var is set"
        else
            echo -e "  ${YELLOW}○${NC} $var not set (recommended)"
        fi
    done
    
    # Check smoke tests passed
    echo -e "\n${BLUE}Checking smoke test status...${NC}"
    SMOKE_LOG=$(ls -t "${PROJECT_ROOT}/logs/smoke_tests/"smoke_test_*.log 2>/dev/null | head -1)
    if [[ -n "$SMOKE_LOG" ]] && grep -q "ALL SMOKE TESTS PASSED" "$SMOKE_LOG"; then
        echo -e "  ${GREEN}✓${NC} Recent smoke tests passed"
    else
        echo -e "  ${YELLOW}○${NC} No recent smoke test found - recommend running smoke_test.sh first"
    fi
    
    # Check disk space
    echo -e "\n${BLUE}Checking disk space...${NC}"
    AVAILABLE_GB=$(df -BG "$PROJECT_ROOT" | tail -1 | awk '{print $4}' | tr -d 'G')
    if [[ "$AVAILABLE_GB" -gt 10 ]]; then
        echo -e "  ${GREEN}✓${NC} Sufficient disk space: ${AVAILABLE_GB}GB available"
    else
        echo -e "  ${RED}✗${NC} Low disk space: ${AVAILABLE_GB}GB available (need 10GB+)"
        CHECKS_PASSED=false
    fi
    
    # Check Python modules
    echo -e "\n${BLUE}Checking Python modules...${NC}"
    cd "$PROJECT_ROOT"
    if python -c "from src.trading.paper_trading import PaperTradingEngine" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} Paper trading module OK"
    else
        echo -e "  ${RED}✗${NC} Paper trading module import failed"
        CHECKS_PASSED=false
    fi
    
    # Check monitoring stack
    echo -e "\n${BLUE}Checking monitoring stack...${NC}"
    if [[ -n "$PROMETHEUS_PUSHGATEWAY_URL" ]]; then
        if curl -s -o /dev/null -w "" --connect-timeout 3 "$PROMETHEUS_PUSHGATEWAY_URL" 2>/dev/null; then
            echo -e "  ${GREEN}✓${NC} Prometheus Pushgateway reachable"
        else
            echo -e "  ${YELLOW}○${NC} Prometheus Pushgateway not reachable (metrics disabled)"
        fi
    fi
    
    echo ""
    if [[ "$CHECKS_PASSED" == "true" ]]; then
        echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
        echo -e "${GREEN}  PRE-FLIGHT CHECKS PASSED${NC}"
        echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
        return 0
    else
        echo -e "${RED}═══════════════════════════════════════════════════════════════${NC}"
        echo -e "${RED}  PRE-FLIGHT CHECKS FAILED - FIX ISSUES BEFORE PROCEEDING${NC}"
        echo -e "${RED}═══════════════════════════════════════════════════════════════${NC}"
        return 1
    fi
}

###############################################################################
# Setup Logging
###############################################################################

setup_logging() {
    echo -e "\n${BLUE}Setting up logging...${NC}"
    
    RUN_LOG_DIR="${LOG_DIR}/${RUN_ID}"
    mkdir -p "$RUN_LOG_DIR"
    
    # Create symlink to latest
    ln -sfn "$RUN_LOG_DIR" "${LOG_DIR}/latest"
    
    echo -e "  ${GREEN}✓${NC} Log directory: $RUN_LOG_DIR"
    echo -e "  ${GREEN}✓${NC} Symlink: ${LOG_DIR}/latest"
    
    # Create run manifest
    cat > "${RUN_LOG_DIR}/manifest.json" << EOF
{
    "run_id": "${RUN_ID}",
    "start_time": "$(date -Iseconds)",
    "duration_days": ${SHADOW_DURATION},
    "mode": "shadow",
    "dry_run": ${DRY_RUN},
    "project_root": "${PROJECT_ROOT}",
    "git_commit": "$(git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "git_branch": "$(git -C "$PROJECT_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')",
    "hostname": "$(hostname)",
    "user": "$(whoami)"
}
EOF
    echo -e "  ${GREEN}✓${NC} Manifest created"
}

###############################################################################
# Start Shadow Trading
###############################################################################

start_shadow() {
    echo -e "\n${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}  STARTING SHADOW TRADING${NC}"
    echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo -e "\n${CYAN}[DRY RUN] Would execute:${NC}"
        echo -e "python -m src.trading.paper_trading_orchestrator \\"
        echo -e "  --mode shadow \\"
        echo -e "  --start-now \\"
        echo -e "  --duration ${SHADOW_DURATION} \\"
        echo -e "  --log-dir ${RUN_LOG_DIR} \\"
        echo -e "  --run-id ${RUN_ID}"
        if [[ -n "$PROMETHEUS_PUSHGATEWAY_URL" ]]; then
            echo -e "  --prometheus-push-url ${PROMETHEUS_PUSHGATEWAY_URL}"
        fi
        echo ""
        echo -e "${GREEN}Dry run complete - no trading started${NC}"
        return 0
    fi
    
    # Build command
    CMD="python -m src.trading.paper_trading_orchestrator"
    CMD+=" --mode shadow"
    CMD+=" --start-now"
    CMD+=" --duration ${SHADOW_DURATION}"
    CMD+=" --log-dir ${RUN_LOG_DIR}"
    CMD+=" --run-id ${RUN_ID}"
    
    if [[ -n "$PROMETHEUS_PUSHGATEWAY_URL" ]]; then
        CMD+=" --prometheus-push-url ${PROMETHEUS_PUSHGATEWAY_URL}"
    fi
    
    # Send startup notification
    if [[ -n "$SLACK_WEBHOOK_URL" ]]; then
        curl -s -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"🚀 *Shadow Trading Started*\n• Run ID: ${RUN_ID}\n• Duration: ${SHADOW_DURATION} days\n• Host: $(hostname)\"}" \
            "$SLACK_WEBHOOK_URL" > /dev/null
    fi
    
    echo -e "\n${GREEN}Starting shadow trading...${NC}"
    echo -e "Command: $CMD"
    echo -e "Log: ${RUN_LOG_DIR}/trading.log"
    echo ""
    
    # Run in background with nohup
    cd "$PROJECT_ROOT"
    nohup $CMD > "${RUN_LOG_DIR}/trading.log" 2>&1 &
    PID=$!
    
    echo "$PID" > "${RUN_LOG_DIR}/trading.pid"
    
    sleep 2
    
    if ps -p $PID > /dev/null 2>&1; then
        echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
        echo -e "${GREEN}  SHADOW TRADING STARTED SUCCESSFULLY${NC}"
        echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
        echo ""
        echo -e "  ${BLUE}PID:${NC} $PID"
        echo -e "  ${BLUE}Run ID:${NC} $RUN_ID"
        echo -e "  ${BLUE}Duration:${NC} $SHADOW_DURATION days"
        echo -e "  ${BLUE}End Date:${NC} $(date -d "+${SHADOW_DURATION} days" "+%Y-%m-%d %H:%M")"
        echo -e "  ${BLUE}Logs:${NC} ${RUN_LOG_DIR}/"
        echo ""
        echo -e "  ${CYAN}To monitor:${NC}"
        echo -e "    tail -f ${RUN_LOG_DIR}/trading.log"
        echo ""
        echo -e "  ${CYAN}To stop:${NC}"
        echo -e "    kill \$(cat ${RUN_LOG_DIR}/trading.pid)"
        echo ""
    else
        echo -e "${RED}═══════════════════════════════════════════════════════════════${NC}"
        echo -e "${RED}  FAILED TO START SHADOW TRADING${NC}"
        echo -e "${RED}═══════════════════════════════════════════════════════════════${NC}"
        echo ""
        echo -e "Check logs: ${RUN_LOG_DIR}/trading.log"
        tail -20 "${RUN_LOG_DIR}/trading.log" 2>/dev/null || echo "No log output"
        return 1
    fi
}

###############################################################################
# Setup Daily Cron
###############################################################################

setup_daily_cron() {
    echo -e "\n${BLUE}Setting up daily report cron...${NC}"
    
    # Create cron script
    CRON_SCRIPT="${PROJECT_ROOT}/scripts/daily_shadow_report.sh"
    cat > "$CRON_SCRIPT" << EOF
#!/bin/bash
# Auto-generated daily report script for shadow run ${RUN_ID}
cd ${PROJECT_ROOT}
python src/ops/daily_report_generator.py --run-id ${RUN_ID} --send
python scripts/check_alerts.py
EOF
    chmod +x "$CRON_SCRIPT"
    
    echo -e "  ${GREEN}✓${NC} Daily report script created: $CRON_SCRIPT"
    echo -e "  ${CYAN}To add to cron (runs 08:05 UTC daily):${NC}"
    echo -e "    echo '5 8 * * * ${CRON_SCRIPT}' | crontab -"
}

###############################################################################
# Main
###############################################################################

main() {
    # Pre-flight checks
    if ! preflight_checks; then
        echo -e "\n${RED}Aborting due to pre-flight check failures${NC}"
        exit 1
    fi
    
    # Setup logging
    setup_logging
    
    # Start shadow trading
    start_shadow
    
    # Setup daily cron
    if [[ "$DRY_RUN" != "true" ]]; then
        setup_daily_cron
    fi
}

main "$@"
