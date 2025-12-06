# =============================================================================
#                        LIVE TRADING RUNBOOK
# =============================================================================
# Delta Exchange Algo Trader - Institutional Production Guide
# Version: 1.0.0
# Last Updated: 2025-12-06
# =============================================================================

## TABLE OF CONTENTS

1. [Pre-Launch Checklist](#1-pre-launch-checklist)
2. [48-Hour Launch Sequence](#2-48-hour-launch-sequence)
3. [Monitoring & Alerting](#3-monitoring--alerting)
4. [Emergency Procedures](#4-emergency-procedures)
5. [Scaling Schedule](#5-scaling-schedule)
6. [Operational Hygiene](#6-operational-hygiene)

---

## 1. PRE-LAUNCH CHECKLIST

### 1.1 Environment Variables Required

```bash
# Vault Configuration
export VAULT_ADDR="https://vault.production.example.com"
export VAULT_AUTH_METHOD="kubernetes"  # or "approle"
export VAULT_K8S_ROLE="delta-trader"

# Trading Configuration
export PAPER=0
export CAPITAL_ALLOCATION_PCT=1

# Monitoring
export SLACK_WEBHOOK_URL="https://hooks.slack.com/..."
export PROMETHEUS_PORT=9090
```

### 1.2 Run Preflight Check

```bash
python ops/launch.py preflight
```

Expected output: **6/6 PASS**

### 1.3 Run Go/No-Go Gate Check

```bash
python -m institutional.go_nogo
```

Expected output: **5/5 PASS, DECISION: GO**

---

## 2. 48-HOUR LAUNCH SEQUENCE

### Hour 0: Deploy Live Pod

```bash
# Deploy with 1% allocation
kubectl apply -f k8s/live.yaml
kubectl annotate deployment/delta-trader deployment.kubernetes.io/revision="v1.0.0-live"

# Verify deployment
kubectl get pods -n trading -w
```

### Hour 0-24: First Trading Session

**WATCH:**
- Grafana: "Live vs Paper Slippage" panel
- Slack: #trading-alerts channel
- AMRC: Zero kill-switch triggers

**ACCEPTABLE:**
- Slippage ≤ 1.1 × paper estimate
- Drawdown < 0.5%
- Zero Vault alerts

### Hour 24: First Review

```bash
# Export audit chunk for auditor
python -c "
from institutional.audit import AuditBus
audit = AuditBus()
entries = audit.query(start_date='TODAY', end_date='TODAY')
print(f'Entries: {len(entries)}')
"

# Check metrics
python -m institutional.go_nogo
```

**IF PASSING:** Bump to 3% allocation
**IF FAILING:** Investigate before proceeding

### Hour 24-168 (Days 2-7): Monitoring Period

- Daily audit chunk export
- Daily go/no-go check
- Watch for any anomalies

### Day 7+: Scale to Target

```bash
# After 7 consecutive green days
python ops/launch.py scale 5
```

---

## 3. MONITORING & ALERTING

### 3.1 Key Metrics Dashboard

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Slippage Ratio | ≤ 1.1x | > 1.2x |
| Daily Drawdown | < 0.5% | > 2% |
| AMRC Halt Time | < 60ms | > 100ms |
| Vault Token TTL | > 30min | < 5min |
| Order Fill Rate | > 95% | < 80% |

### 3.2 Alert Channels

- **Slack**: #trading-alerts (all alerts)
- **PagerDuty**: trading-oncall (critical only)
- **Email**: trading-team@yourcompany.com (daily digest)

### 3.3 Grafana Dashboards

1. **Trading Overview**: P&L, positions, volume
2. **Risk Metrics**: Drawdown, exposure, VaR
3. **Execution Quality**: Slippage, fill rates, latency
4. **Infrastructure**: CPU, memory, Vault status

---

## 4. EMERGENCY PROCEDURES

### 4.1 Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│                    EMERGENCY PLAYBOOK                       │
├─────────────────────────────────────────────────────────────┤
│ Scenario              │ Command                             │
├───────────────────────┼─────────────────────────────────────┤
│ Kill all trading      │ python ops/emergency.py kill        │
│ Check system status   │ python ops/emergency.py status      │
│ Rotate leaked secrets │ python ops/emergency.py rotate      │
│ Rollback deployment   │ python ops/emergency.py rollback    │
│ Scale to 0 replicas   │ python ops/emergency.py scale 0     │
└───────────────────────┴─────────────────────────────────────┘
```

### 4.2 Incident Response Matrix

| Severity | Response Time | Actions |
|----------|---------------|---------|
| P1 (Critical) | < 5 min | Kill switch, page on-call, incident bridge |
| P2 (High) | < 15 min | Investigate, reduce allocation if needed |
| P3 (Medium) | < 1 hour | Create ticket, monitor closely |
| P4 (Low) | Next business day | Add to backlog |

### 4.3 Common Scenarios

**Sudden 5% Drawdown:**
```bash
python ops/emergency.py kill
# Investigate before resuming
```

**Vault Unreachable:**
```bash
kubectl set env deploy/delta-trader VAULT_ADDR=$FALLBACK_VAULT -n trading
```

**API Key Compromised:**
```bash
vault lease revoke -prefix secret/trading
python ops/emergency.py rotate
```

**Container CVE Discovered:**
```bash
# Build new image with fix
docker build -t ghcr.io/your-org/delta-trader:v1.0.1 .

# Deploy new version
kubectl set image deploy/delta-trader trader=ghcr.io/your-org/delta-trader:v1.0.1 -n trading

# Verify signature
cosign verify ghcr.io/your-org/delta-trader:v1.0.1
```

---

## 5. SCALING SCHEDULE

### 5.1 Allocation Ladder

| Phase | Allocation | Criteria | Duration |
|-------|------------|----------|----------|
| 1 | 1% | Initial deploy | Days 1-7 |
| 2 | 3% | 7 green days, slippage ≤ 1.1x | Days 8-14 |
| 3 | 5% | 14 green days, drawdown < 1% | Days 15+ |
| 4 | 10%* | 30 green days, LP approval | Month 2+ |

*Requires manual sign-off from risk committee

### 5.2 Scale-Up Procedure

```bash
# 1. Verify criteria
python -m institutional.go_nogo

# 2. Review last 7 days
python ops/launch.py scale 3  # or 5

# 3. Update and deploy
kubectl set env deploy/delta-trader CAPITAL_ALLOCATION_PCT=3 -n trading

# 4. Document
echo "Scaled to 3% at $(date)" >> ops/launch_logs/scaling.log
```

---

## 6. OPERATIONAL HYGIENE

### 6.1 Weekly Tasks

- [ ] Rotate Vault secrets
- [ ] Rerun stress test suite
- [ ] Archive audit chunks to cold storage
- [ ] Review slippage trends
- [ ] Update model performance trackers

### 6.2 Monthly Tasks

- [ ] Auditor spot-check
- [ ] Capacity re-estimation
- [ ] Model card refresh
- [ ] Dependency security scan
- [ ] Backup verification

### 6.3 Quarterly Tasks

- [ ] Full penetration test
- [ ] DR failover drill
- [ ] LP letter renewal
- [ ] Strategy performance review
- [ ] Infrastructure capacity planning

---

## APPENDIX A: COMMANDS CHEAT SHEET

```bash
# Daily operations
python -m institutional.go_nogo                    # Gate check
python ops/launch.py monitor                       # Show dashboards
kubectl logs -f deploy/delta-trader -n trading    # Live logs

# Deployment
kubectl apply -f k8s/live.yaml                    # Deploy
kubectl rollout status deploy/delta-trader        # Watch deploy
kubectl rollout undo deploy/delta-trader          # Rollback

# Emergency
python ops/emergency.py kill                      # KILL SWITCH
python ops/emergency.py status                    # Health check
python ops/emergency.py rotate                    # Rotate secrets

# Scaling
python ops/emergency.py scale 0                   # Stop trading
python ops/emergency.py scale 1                   # Resume trading
```

---

## APPENDIX B: CONTACT LIST

| Role | Name | Contact |
|------|------|---------|
| Primary On-Call | TBD | pager |
| Secondary On-Call | TBD | pager |
| Risk Officer | TBD | phone |
| External Auditor | Smith & Associates | email |

---

**Document Control:**
- Version: 1.0.0
- Author: Delta Trading Team
- Approved By: Risk Committee
- Review Date: Quarterly
