<#
.SYNOPSIS
    Smoke Test Suite for Quant Bot
.DESCRIPTION
    Validates all critical wiring before shadow/production deployment.
    Run: .\scripts\smoke_test.ps1
.NOTES
    Author: Quant Bot Ops
    Version: 1.0.0
#>

param(
    [switch]$SkipAPI,
    [switch]$Verbose
)

$ErrorActionPreference = "Continue"

# Colors
function Write-ColorOutput {
    param([string]$Message, [string]$Color = "White")
    Write-Host $Message -ForegroundColor $Color
}

function Write-Pass { param([string]$Message) Write-ColorOutput "✅ PASS: $Message" "Green" }
function Write-Fail { param([string]$Message) Write-ColorOutput "❌ FAIL: $Message" "Red" }
function Write-Skip { param([string]$Message) Write-ColorOutput "⏭️  SKIP: $Message" "Yellow" }
function Write-Warn { param([string]$Message) Write-ColorOutput "⚠️  WARN: $Message" "Yellow" }
function Write-Header { 
    param([string]$Message)
    Write-ColorOutput "`n═══════════════════════════════════════════════════════════════" "Cyan"
    Write-ColorOutput "  $Message" "Cyan"
    Write-ColorOutput "═══════════════════════════════════════════════════════════════`n" "Cyan"
}

# Initialize counters
$script:TestsPassed = 0
$script:TestsFailed = 0
$script:TestsSkipped = 0

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$LogDir = Join-Path $ProjectRoot "logs\smoke_tests"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogDir "smoke_test_$Timestamp.log"

# Create log directory
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

Write-ColorOutput "Smoke Test Started: $(Get-Date -Format 'o')" "White"
Write-ColorOutput "Project Root: $ProjectRoot" "White"
Write-ColorOutput "Log File: $LogFile`n" "White"

# =============================================================================
# 1. Environment Check
# =============================================================================
Write-Header "1. ENVIRONMENT VALIDATION"

# Check Python
try {
    $pythonVersion = & python --version 2>&1
    Write-Pass "Python installed: $pythonVersion"
    $script:TestsPassed++
} catch {
    Write-Fail "Python not found"
    $script:TestsFailed++
}

# Check required env vars
$RequiredVars = @("TRADING_API_KEY", "TRADING_API_SECRET")
$OptionalVars = @("SLACK_WEBHOOK_URL", "PROMETHEUS_PUSHGATEWAY_URL", "GRAFANA_API_KEY")

foreach ($var in $RequiredVars) {
    if ([Environment]::GetEnvironmentVariable($var)) {
        Write-Pass "Required env var set: $var"
        $script:TestsPassed++
    } else {
        Write-Warn "Required env var missing: $var (set in .env or environment)"
        # Don't fail - might be using .env file
    }
}

foreach ($var in $OptionalVars) {
    if ([Environment]::GetEnvironmentVariable($var)) {
        Write-Pass "Optional env var set: $var"
    } else {
        Write-Warn "Optional env var not set: $var (some features disabled)"
    }
}

# =============================================================================
# 2. Module Import Tests
# =============================================================================
Write-Header "2. MODULE IMPORT TESTS"

Set-Location $ProjectRoot

$Modules = @(
    "src.options.pricing_engine",
    "src.options.volatility_surface",
    "src.options.strategies",
    "src.arbitrage.funding_arbitrage",
    "src.arbitrage.statistical_arbitrage",
    "src.arbitrage.cross_exchange"
)

foreach ($module in $Modules) {
    try {
        $result = & python -c "import $module" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Pass "Import: $module"
            $script:TestsPassed++
        } else {
            Write-Fail "Import failed: $module"
            $script:TestsFailed++
        }
    } catch {
        Write-Fail "Import failed: $module - $_"
        $script:TestsFailed++
    }
}

# =============================================================================
# 3. Ops Scripts Tests
# =============================================================================
Write-Header "3. OPS SCRIPTS VALIDATION"

# Test rollback status
Write-ColorOutput "`nTesting rollback.py..." "Yellow"
try {
    & python src/ops/rollback.py --status 2>&1 | Tee-Object -FilePath $LogFile -Append
    if ($LASTEXITCODE -eq 0) {
        Write-Pass "Rollback status check"
        $script:TestsPassed++
    } else {
        Write-Fail "Rollback status check"
        $script:TestsFailed++
    }
} catch {
    Write-Fail "Rollback status check - $_"
    $script:TestsFailed++
}

# Test rollback dry-run
try {
    & python src/ops/rollback.py --dry-run --target-version v1.0.0 2>&1 | Tee-Object -FilePath $LogFile -Append
    if ($LASTEXITCODE -eq 0) {
        Write-Pass "Rollback dry-run"
        $script:TestsPassed++
    } else {
        Write-Fail "Rollback dry-run"
        $script:TestsFailed++
    }
} catch {
    Write-Fail "Rollback dry-run - $_"
    $script:TestsFailed++
}

# Test canary orchestrator
Write-ColorOutput "`nTesting canary_orchestrator.py..." "Yellow"
try {
    & python src/ops/canary_orchestrator.py --dry-run --stage Canary-1 2>&1 | Tee-Object -FilePath $LogFile -Append
    if ($LASTEXITCODE -eq 0) {
        Write-Pass "Canary orchestrator dry-run"
        $script:TestsPassed++
    } else {
        Write-Fail "Canary orchestrator dry-run"
        $script:TestsFailed++
    }
} catch {
    Write-Fail "Canary orchestrator dry-run - $_"
    $script:TestsFailed++
}

# Test daily report generator
Write-ColorOutput "`nTesting daily_report_generator.py..." "Yellow"
$ReportOutput = Join-Path $env:TEMP "smoke_test_daily_report_$Timestamp.md"
try {
    & python src/ops/daily_report_generator.py --sample --output $ReportOutput 2>&1 | Tee-Object -FilePath $LogFile -Append
    if ((Test-Path $ReportOutput) -and $LASTEXITCODE -eq 0) {
        Write-Pass "Daily report generator (output: $ReportOutput)"
        $script:TestsPassed++
        Write-ColorOutput "`nReport preview:" "Cyan"
        Get-Content $ReportOutput -Head 15
    } else {
        Write-Fail "Daily report generator (no output file)"
        $script:TestsFailed++
    }
} catch {
    Write-Fail "Daily report generator - $_"
    $script:TestsFailed++
}

# Test replay suite
Write-ColorOutput "`nTesting replay_suite.py..." "Yellow"
try {
    & python src/ops/replay_suite.py --by-id "SMOKE-TEST-001" --dry-run 2>&1 | Tee-Object -FilePath $LogFile -Append
    if ($LASTEXITCODE -eq 0) {
        Write-Pass "Replay suite dry-run"
        $script:TestsPassed++
    } else {
        Write-Fail "Replay suite dry-run"
        $script:TestsFailed++
    }
} catch {
    Write-Fail "Replay suite dry-run - $_"
    $script:TestsFailed++
}

# =============================================================================
# 4. Unit Tests
# =============================================================================
Write-Header "4. UNIT TESTS"

Write-ColorOutput "Running test_options_arbitrage.py..." "Yellow"
try {
    & python test_options_arbitrage.py 2>&1 | Tee-Object -FilePath $LogFile -Append
    if ($LASTEXITCODE -eq 0) {
        Write-Pass "Options & Arbitrage tests"
        $script:TestsPassed++
    } else {
        Write-Fail "Options & Arbitrage tests"
        $script:TestsFailed++
    }
} catch {
    Write-Fail "Options & Arbitrage tests - $_"
    $script:TestsFailed++
}

# =============================================================================
# 5. Notification Tests
# =============================================================================
Write-Header "5. NOTIFICATION TESTS"

$SlackWebhook = [Environment]::GetEnvironmentVariable("SLACK_WEBHOOK_URL")
if ($SlackWebhook) {
    Write-ColorOutput "Testing Slack webhook..." "Yellow"
    try {
        $body = @{ text = "🧪 Smoke Test: Notification channel verified at $(Get-Date -Format 'o')" } | ConvertTo-Json
        $response = Invoke-RestMethod -Uri $SlackWebhook -Method Post -Body $body -ContentType "application/json" -ErrorAction Stop
        Write-Pass "Slack webhook"
        $script:TestsPassed++
    } catch {
        Write-Fail "Slack webhook - $_"
        $script:TestsFailed++
    }
} else {
    Write-Skip "Slack test (SLACK_WEBHOOK_URL not set)"
    $script:TestsSkipped++
}

# =============================================================================
# 6. Prometheus Tests
# =============================================================================
Write-Header "6. PROMETHEUS TESTS"

$PushgatewayUrl = [Environment]::GetEnvironmentVariable("PROMETHEUS_PUSHGATEWAY_URL")
if ($PushgatewayUrl) {
    Write-ColorOutput "Testing Prometheus Pushgateway..." "Yellow"
    try {
        $response = Invoke-WebRequest -Uri $PushgatewayUrl -Method Get -TimeoutSec 5 -ErrorAction Stop
        Write-Pass "Prometheus Pushgateway reachable"
        $script:TestsPassed++
    } catch {
        Write-Fail "Prometheus Pushgateway not reachable"
        $script:TestsFailed++
    }
} else {
    Write-Skip "Prometheus tests (PROMETHEUS_PUSHGATEWAY_URL not set)"
    $script:TestsSkipped++
}

# =============================================================================
# Summary
# =============================================================================
Write-Header "SMOKE TEST SUMMARY"

$Total = $script:TestsPassed + $script:TestsFailed + $script:TestsSkipped

Write-ColorOutput "Total Tests: $Total" "White"
Write-ColorOutput "Passed: $($script:TestsPassed)" "Green"
Write-ColorOutput "Failed: $($script:TestsFailed)" "Red"
Write-ColorOutput "Skipped: $($script:TestsSkipped)" "Yellow"
Write-ColorOutput "`nLog file: $LogFile" "White"

if ($script:TestsFailed -eq 0) {
    Write-ColorOutput "`n════════════════════════════════════════════════════════════════" "Green"
    Write-ColorOutput "  🎉 ALL SMOKE TESTS PASSED - READY FOR SHADOW DEPLOYMENT" "Green"
    Write-ColorOutput "════════════════════════════════════════════════════════════════`n" "Green"
    exit 0
} else {
    Write-ColorOutput "`n════════════════════════════════════════════════════════════════" "Red"
    Write-ColorOutput "  ❌ SMOKE TESTS FAILED - FIX ISSUES BEFORE DEPLOYMENT" "Red"
    Write-ColorOutput "════════════════════════════════════════════════════════════════`n" "Red"
    exit 1
}
