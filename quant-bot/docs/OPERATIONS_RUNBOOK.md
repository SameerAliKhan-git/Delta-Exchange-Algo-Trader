# Production Operations Runbook

## Quick Reference

| Action | Command |
|--------|---------|
| **Emergency Kill** | `python src/ops/rollback.py --reason "emergency"` |
| **Check Status** | `python src/ops/canary_orchestrator.py --status` |
| **View Daily Report** | `python src/ops/daily_report_generator.py --no-send` |
| **Replay Adverse Trades** | `python src/ops/replay_suite.py --last-n 10` |
| **Canary Progress** | `python src/ops/canary_orchestrator.py --progress` |

---

## 1. Daily Operations

### 1.1 Morning Checklist (08:00 UTC)

```bash
# 1. Check system status
python src/ops/canary_orchestrator.py --status

# 2. Review daily report (auto-generated at 08:00)
cat reports/daily/daily_report_$(date +%Y-%m-%d).md

# 3. Check for overnight alerts
# Review Slack #trading-alerts channel

# 4. Verify monitoring
curl http://localhost:9090/-/healthy  # Prometheus
curl http://localhost:3000/api/health # Grafana
```

### 1.2 Key Metrics to Monitor

| Metric | Warning | Critical |
|--------|---------|----------|
| Hourly P&L | -0.3% AUM | -0.5% AUM |
| Daily P&L | -1% AUM | -2% AUM |
| Max Drawdown | >7% | >10% |
| Slippage Ratio | >2x | >3x |
| Fill Rate | <95% | <90% |
| Model Precision | <55% | <50% |

### 1.3 Evening Checklist (20:00 UTC)

```bash
# 1. Generate end-of-day summary
python src/ops/daily_report_generator.py

# 2. Check canary progress (if applicable)
python src/ops/canary_orchestrator.py --report

# 3. Review any flagged trades
python src/ops/replay_suite.py --threshold -50 --days 1
```

---

## 2. Alert Response Procedures

### 2.1 Critical Alert: Loss Limit Breached

**Trigger:** `HourlyLossExceeded` or `DailyLossExceeded` alert

**Immediate Actions (within 5 minutes):**

1. Acknowledge alert in PagerDuty/Slack
2. Verify kill switch activation:
   ```bash
   cat KILL_SWITCH  # Should exist if auto-triggered
   ```
3. If not auto-activated, manually activate:
   ```bash
   python src/ops/rollback.py --reason "Loss limit breach"
   ```
4. Check current positions:
   ```bash
   curl http://localhost:8000/api/v1/positions
   ```

**Investigation (within 1 hour):**

1. Pull trade logs:
   ```bash
   python src/ops/replay_suite.py --date $(date +%Y-%m-%d) --threshold -100
   ```
2. Check for execution issues:
   - Review slippage vs expected
   - Check fill rates
   - Look for unusual market conditions

3. Document findings in incident ticket

**Resolution:**

1. Address root cause
2. Test fix in shadow mode (minimum 2 hours)
3. Resume with reduced position sizes (10% normal)
4. Monitor closely for 24 hours
5. Gradually restore position sizes

### 2.2 Critical Alert: Slippage Spike

**Trigger:** `RealisedSlippageCritical` alert (>3x simulated)

**Immediate Actions:**

1. Reduce position sizes:
   ```bash
   echo '{"max_position_pct": 0.005}' > config/position_override.json
   ```
2. Check exchange status
3. Check market liquidity conditions

**Investigation:**

1. Compare orderbook depth vs historical
2. Review timing of fills
3. Check for large market orders

**Resolution:**

1. Recalibrate Almgren-Chriss model
2. Consider execution venue changes
3. Adjust participation rate limits

### 2.3 Warning Alert: Model Drift

**Trigger:** `ModelDriftDetected_PSI` or `ModelDriftDetected_KS`

**Actions:**

1. Freeze online learning (if not automatic):
   ```bash
   echo '{"accept_threshold": 0.99}' > config/online_learning_freeze.json
   ```
2. Review feature distributions
3. Schedule batch retrain if persistent

---

## 3. Rollback Procedures

### 3.1 Automated Rollback

Rollback is automatically triggered by:
- Critical Prometheus alerts with `action: AUTO_KILL_SWITCH`
- KILL_SWITCH file creation
- Canary acceptance criteria failure

**What happens automatically:**
1. Trading stopped
2. Kill switch file created
3. Model rolled back to last known good
4. Online learning frozen
5. Position sizes reduced to 10%
6. Notifications sent (Slack, PagerDuty)
7. JIRA ticket created

### 3.2 Manual Rollback

```bash
# Full rollback
python src/ops/rollback.py --reason "Manual: describe reason"

# Rollback to specific version
python src/ops/rollback.py --target-version v2.3.1 --reason "Manual rollback to v2.3.1"

# Dry run (see what would happen)
python src/ops/rollback.py --dry-run
```

### 3.3 Recovery After Rollback

1. **Investigate root cause** (mandatory)
   ```bash
   python src/ops/replay_suite.py --last-n 20
   ```

2. **Test fix in shadow mode** (minimum 24 hours)
   ```bash
   python src/ops/canary_orchestrator.py --start shadow
   ```

3. **Verify metrics are healthy**
   - P&L positive or neutral
   - Slippage within normal range
   - No new alerts

4. **Deactivate kill switch** (requires authorization)
   ```bash
   # Requires manual confirmation
   rm KILL_SWITCH
   ```

5. **Resume canary progression**
   ```bash
   python src/ops/canary_orchestrator.py --start canary_1
   ```

---

## 4. Canary Deployment

### 4.1 Progression Rules

| Stage | AUM % | Duration | Max DD | P&L Dev |
|-------|-------|----------|--------|---------|
| Shadow | 0% | 30 days | N/A | N/A |
| Canary-1 | 1% | 7 days | 2% | ±10% |
| Canary-2 | 5% | 14 days | 3% | ±10% |
| Production | 100% | Ongoing | 10% | ±15% |

### 4.2 Commands

```bash
# Check current stage and report
python src/ops/canary_orchestrator.py --report

# Manually start a stage
python src/ops/canary_orchestrator.py --start canary_1

# Check if ready to progress
python src/ops/canary_orchestrator.py --progress

# Run continuous monitoring
python src/ops/canary_orchestrator.py --run

# Force rollback
python src/ops/canary_orchestrator.py --rollback
```

### 4.3 Acceptance Criteria

Before progressing, ALL of these must be true:

- [ ] Minimum duration elapsed
- [ ] P&L within ±10% of simulated
- [ ] Slippage ≤ 1.5x simulated
- [ ] No critical alerts
- [ ] Max drawdown under threshold
- [ ] Model precision ≥ 90% of baseline

---

## 5. Model Updates

### 5.1 Online Learning Policy

- Updates evaluated in shadow mode first
- Must improve net P&L by configurable threshold
- Automatic rollback if degradation detected
- Maximum 10 updates per hour (throttled)

### 5.2 Batch Retrain Procedure

1. **Schedule** (weekly, during low volume)
2. **Prepare** validation data (purged, embargor)
3. **Train** with cost-aware objective
4. **Validate**:
   - Walk-forward Sharpe > 1.0
   - Deflated Sharpe positive
   - PBO < 0.3
5. **Deploy to shadow** for 48 hours
6. **Register** in model registry with sign-off
7. **Canary** deploy new model

### 5.3 Model Registry

```bash
# List registered models
ls models/registry/

# Check current active model
cat models/active/model_info.json

# Check last known good model
cat models/last_good_model.json
```

---

## 6. Post-Mortem Template

### Incident Details

| Field | Value |
|-------|-------|
| Date/Time | |
| Duration | |
| Severity | |
| Impact | |
| Root Cause | |

### Timeline

| Time (UTC) | Event |
|------------|-------|
| HH:MM | Alert triggered |
| HH:MM | Response began |
| HH:MM | Root cause identified |
| HH:MM | Fix implemented |
| HH:MM | Service restored |

### Analysis

**What happened?**

**Why did it happen?**

**Why wasn't it caught earlier?**

### Actions

| Action | Owner | Deadline | Status |
|--------|-------|----------|--------|
| | | | |

### Lessons Learned

1. 
2. 
3. 

---

## 7. Emergency Contacts

| Role | Contact |
|------|---------|
| Primary On-Call | |
| Secondary On-Call | |
| Risk Manager | |
| Engineering Lead | |
| Exchange Support | |

---

## 8. Useful Commands Reference

```bash
# === Status & Monitoring ===
python src/ops/canary_orchestrator.py --status
python src/ops/rollback.py --status
curl localhost:8080/metrics
curl localhost:8080/health

# === Reports ===
python src/ops/daily_report_generator.py --no-send
python src/ops/daily_report_generator.py --date 2024-01-15
python src/ops/replay_suite.py --last-n 20 --threshold -100

# === Emergency Actions ===
python src/ops/rollback.py --reason "Emergency"
echo '{}' > KILL_SWITCH  # Manual kill switch
rm KILL_SWITCH           # Deactivate (careful!)

# === Canary Management ===
python src/ops/canary_orchestrator.py --start shadow
python src/ops/canary_orchestrator.py --progress
python src/ops/canary_orchestrator.py --rollback

# === Configuration ===
cat config/trading_config.json
cat config/position_override.json
cat config/online_learning_freeze.json
```

---

*Last Updated: December 2024*
*Version: 1.0*
