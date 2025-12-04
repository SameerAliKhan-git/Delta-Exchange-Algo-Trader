"""
Automated Daily Report Generator
================================
Generates daily trading reports with Slack/Discord webhook delivery.

Report Contents:
- Net P&L (gross & net after costs)
- Realized vs simulated slippage
- Trade statistics (count, avg P&L, win rate, profit factor)
- Turnover (% AUM/day)
- Max drawdown (rolling 30d, 7d)
- Circuit breaker/kill triggers
- Model update events
- Per-strategy breakdown
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import statistics
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StrategyMetrics:
    """Metrics for a single strategy."""
    name: str
    net_pnl: float
    gross_pnl: float
    trades: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float
    avg_trade_pnl: float
    max_win: float
    max_loss: float
    slippage_realized_bps: float
    slippage_simulated_bps: float


@dataclass  
class DailyMetrics:
    """Aggregated daily metrics."""
    date: str
    
    # P&L
    net_pnl: float
    gross_pnl: float
    transaction_costs: float
    
    # Slippage
    slippage_realized_bps: float
    slippage_simulated_bps: float
    slippage_ratio: float
    
    # Trades
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_trade_pnl: float
    
    # Turnover
    turnover_usd: float
    turnover_pct_aum: float
    
    # Risk
    max_drawdown_30d: float
    max_drawdown_7d: float
    current_drawdown: float
    sharpe_30d: float
    
    # Events
    circuit_breaker_triggers: int
    kill_switch_activations: int
    model_updates: int
    
    # Model
    model_precision: float
    model_precision_baseline: float
    
    # By strategy
    strategies: List[StrategyMetrics]
    
    # Fill quality
    fill_rate: float
    avg_fill_latency_ms: float


class DailyReportGenerator:
    """
    Generate and distribute daily trading reports.
    
    Usage:
        generator = DailyReportGenerator()
        await generator.generate_and_send()
    """
    
    def __init__(
        self,
        metrics_dir: str = "./data/metrics",
        reports_dir: str = "./reports/daily",
        slack_webhook: str = None,
        discord_webhook: str = None,
        email_config: Dict = None
    ):
        self.metrics_dir = Path(metrics_dir)
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        self.slack_webhook = slack_webhook or os.getenv("SLACK_WEBHOOK_URL")
        self.discord_webhook = discord_webhook or os.getenv("DISCORD_WEBHOOK_URL")
        self.email_config = email_config
    
    def collect_metrics(self, date: datetime = None) -> DailyMetrics:
        """Collect metrics for a specific date."""
        date = date or datetime.now() - timedelta(days=1)
        date_str = date.strftime("%Y-%m-%d")
        
        # Load from metrics files (would connect to actual data sources)
        metrics_file = self.metrics_dir / f"daily_{date_str}.json"
        
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                data = json.load(f)
            return self._parse_metrics(data, date_str)
        
        # Generate sample metrics if no data (for demo)
        return self._generate_sample_metrics(date_str)
    
    def _parse_metrics(self, data: Dict, date_str: str) -> DailyMetrics:
        """Parse metrics from data dict."""
        strategies = []
        for s in data.get('strategies', []):
            strategies.append(StrategyMetrics(**s))
        
        return DailyMetrics(
            date=date_str,
            net_pnl=data.get('net_pnl', 0),
            gross_pnl=data.get('gross_pnl', 0),
            transaction_costs=data.get('transaction_costs', 0),
            slippage_realized_bps=data.get('slippage_realized_bps', 0),
            slippage_simulated_bps=data.get('slippage_simulated_bps', 0),
            slippage_ratio=data.get('slippage_ratio', 1.0),
            total_trades=data.get('total_trades', 0),
            winning_trades=data.get('winning_trades', 0),
            losing_trades=data.get('losing_trades', 0),
            win_rate=data.get('win_rate', 0),
            profit_factor=data.get('profit_factor', 0),
            avg_trade_pnl=data.get('avg_trade_pnl', 0),
            turnover_usd=data.get('turnover_usd', 0),
            turnover_pct_aum=data.get('turnover_pct_aum', 0),
            max_drawdown_30d=data.get('max_drawdown_30d', 0),
            max_drawdown_7d=data.get('max_drawdown_7d', 0),
            current_drawdown=data.get('current_drawdown', 0),
            sharpe_30d=data.get('sharpe_30d', 0),
            circuit_breaker_triggers=data.get('circuit_breaker_triggers', 0),
            kill_switch_activations=data.get('kill_switch_activations', 0),
            model_updates=data.get('model_updates', 0),
            model_precision=data.get('model_precision', 0),
            model_precision_baseline=data.get('model_precision_baseline', 0),
            strategies=strategies,
            fill_rate=data.get('fill_rate', 1.0),
            avg_fill_latency_ms=data.get('avg_fill_latency_ms', 0)
        )
    
    def _generate_sample_metrics(self, date_str: str) -> DailyMetrics:
        """Generate sample metrics for demo."""
        import random
        
        net_pnl = random.uniform(-500, 2000)
        transaction_costs = random.uniform(50, 200)
        
        return DailyMetrics(
            date=date_str,
            net_pnl=net_pnl,
            gross_pnl=net_pnl + transaction_costs,
            transaction_costs=transaction_costs,
            slippage_realized_bps=random.uniform(5, 15),
            slippage_simulated_bps=random.uniform(8, 12),
            slippage_ratio=random.uniform(0.8, 1.3),
            total_trades=random.randint(20, 100),
            winning_trades=random.randint(10, 60),
            losing_trades=random.randint(10, 40),
            win_rate=random.uniform(0.45, 0.60),
            profit_factor=random.uniform(1.0, 2.0),
            avg_trade_pnl=net_pnl / 50,
            turnover_usd=random.uniform(50000, 200000),
            turnover_pct_aum=random.uniform(0.5, 2.0),
            max_drawdown_30d=random.uniform(0.02, 0.08),
            max_drawdown_7d=random.uniform(0.01, 0.04),
            current_drawdown=random.uniform(0, 0.03),
            sharpe_30d=random.uniform(0.8, 2.0),
            circuit_breaker_triggers=random.randint(0, 2),
            kill_switch_activations=0,
            model_updates=random.randint(0, 5),
            model_precision=random.uniform(0.55, 0.70),
            model_precision_baseline=0.60,
            strategies=[
                StrategyMetrics(
                    name="momentum",
                    net_pnl=net_pnl * 0.6,
                    gross_pnl=(net_pnl + transaction_costs) * 0.6,
                    trades=30,
                    wins=18,
                    losses=12,
                    win_rate=0.60,
                    profit_factor=1.8,
                    avg_trade_pnl=net_pnl * 0.6 / 30,
                    max_win=200,
                    max_loss=-100,
                    slippage_realized_bps=8,
                    slippage_simulated_bps=10
                ),
                StrategyMetrics(
                    name="mean_reversion",
                    net_pnl=net_pnl * 0.4,
                    gross_pnl=(net_pnl + transaction_costs) * 0.4,
                    trades=20,
                    wins=11,
                    losses=9,
                    win_rate=0.55,
                    profit_factor=1.4,
                    avg_trade_pnl=net_pnl * 0.4 / 20,
                    max_win=150,
                    max_loss=-80,
                    slippage_realized_bps=6,
                    slippage_simulated_bps=8
                )
            ],
            fill_rate=0.97,
            avg_fill_latency_ms=45
        )
    
    def generate_markdown_report(self, metrics: DailyMetrics) -> str:
        """Generate markdown report."""
        
        # P&L emoji
        pnl_emoji = "🟢" if metrics.net_pnl >= 0 else "🔴"
        
        # Slippage status
        slip_emoji = "✅" if metrics.slippage_ratio <= 1.5 else "⚠️"
        
        # Win rate color
        wr_emoji = "🎯" if metrics.win_rate >= 0.5 else "📉"
        
        report = f"""
# Daily Trading Report - {metrics.date}

## 📊 Summary

| Metric | Value |
|--------|-------|
| {pnl_emoji} Net P&L | **${metrics.net_pnl:,.2f}** |
| Gross P&L | ${metrics.gross_pnl:,.2f} |
| Transaction Costs | ${metrics.transaction_costs:,.2f} |
| {wr_emoji} Win Rate | {metrics.win_rate * 100:.1f}% |
| Profit Factor | {metrics.profit_factor:.2f} |
| Sharpe (30d) | {metrics.sharpe_30d:.2f} |

## ⚡ Execution Quality

| Metric | Value |
|--------|-------|
| {slip_emoji} Slippage (Realized) | {metrics.slippage_realized_bps:.1f} bps |
| Slippage (Simulated) | {metrics.slippage_simulated_bps:.1f} bps |
| Slippage Ratio | {metrics.slippage_ratio:.2f}x |
| Fill Rate | {metrics.fill_rate * 100:.1f}% |
| Avg Fill Latency | {metrics.avg_fill_latency_ms:.0f} ms |

## 📈 Trade Activity

| Metric | Value |
|--------|-------|
| Total Trades | {metrics.total_trades} |
| Winners | {metrics.winning_trades} |
| Losers | {metrics.losing_trades} |
| Avg Trade P&L | ${metrics.avg_trade_pnl:.2f} |
| Turnover | ${metrics.turnover_usd:,.0f} ({metrics.turnover_pct_aum:.1f}% AUM) |

## 🛡️ Risk Metrics

| Metric | Value |
|--------|-------|
| Max DD (30d) | {metrics.max_drawdown_30d * 100:.2f}% |
| Max DD (7d) | {metrics.max_drawdown_7d * 100:.2f}% |
| Current DD | {metrics.current_drawdown * 100:.2f}% |

## 🚨 Events

| Metric | Value |
|--------|-------|
| Circuit Breakers | {metrics.circuit_breaker_triggers} |
| Kill Switch | {metrics.kill_switch_activations} |
| Model Updates | {metrics.model_updates} |

## 🤖 Model Performance

| Metric | Value |
|--------|-------|
| Precision | {metrics.model_precision * 100:.1f}% |
| Baseline | {metrics.model_precision_baseline * 100:.1f}% |
| vs Baseline | {(metrics.model_precision / metrics.model_precision_baseline - 1) * 100:+.1f}% |

## 📋 Strategy Breakdown

| Strategy | Net P&L | Trades | Win Rate | PF | Slippage |
|----------|---------|--------|----------|-----|----------|
"""
        
        for s in metrics.strategies:
            report += f"| {s.name} | ${s.net_pnl:,.2f} | {s.trades} | {s.win_rate * 100:.0f}% | {s.profit_factor:.2f} | {s.slippage_realized_bps:.1f}bps |\n"
        
        # Alerts section
        alerts = []
        if metrics.slippage_ratio > 1.5:
            alerts.append("⚠️ Slippage exceeds 1.5x simulated")
        if metrics.circuit_breaker_triggers > 0:
            alerts.append(f"⚠️ {metrics.circuit_breaker_triggers} circuit breaker triggers")
        if metrics.kill_switch_activations > 0:
            alerts.append(f"🚨 {metrics.kill_switch_activations} kill switch activations")
        if metrics.model_precision < metrics.model_precision_baseline * 0.9:
            alerts.append("⚠️ Model precision below 90% of baseline")
        if metrics.fill_rate < 0.95:
            alerts.append("⚠️ Fill rate below 95%")
        
        if alerts:
            report += "\n## ⚠️ Alerts\n\n"
            for alert in alerts:
                report += f"- {alert}\n"
        
        report += f"""
---
*Report generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC*
"""
        
        return report
    
    def generate_slack_message(self, metrics: DailyMetrics) -> Dict:
        """Generate Slack-formatted message."""
        pnl_color = "#36a64f" if metrics.net_pnl >= 0 else "#ff0000"
        pnl_emoji = ":chart_with_upwards_trend:" if metrics.net_pnl >= 0 else ":chart_with_downwards_trend:"
        
        return {
            "attachments": [
                {
                    "color": pnl_color,
                    "blocks": [
                        {
                            "type": "header",
                            "text": {
                                "type": "plain_text",
                                "text": f"📊 Daily Report - {metrics.date}"
                            }
                        },
                        {
                            "type": "section",
                            "fields": [
                                {"type": "mrkdwn", "text": f"*{pnl_emoji} Net P&L*\n${metrics.net_pnl:,.2f}"},
                                {"type": "mrkdwn", "text": f"*Win Rate*\n{metrics.win_rate * 100:.1f}%"},
                                {"type": "mrkdwn", "text": f"*Sharpe (30d)*\n{metrics.sharpe_30d:.2f}"},
                                {"type": "mrkdwn", "text": f"*Trades*\n{metrics.total_trades}"}
                            ]
                        },
                        {
                            "type": "section",
                            "fields": [
                                {"type": "mrkdwn", "text": f"*Slippage*\n{metrics.slippage_realized_bps:.1f} bps ({metrics.slippage_ratio:.2f}x)"},
                                {"type": "mrkdwn", "text": f"*Max DD (7d)*\n{metrics.max_drawdown_7d * 100:.2f}%"},
                                {"type": "mrkdwn", "text": f"*Fill Rate*\n{metrics.fill_rate * 100:.1f}%"},
                                {"type": "mrkdwn", "text": f"*Model Precision*\n{metrics.model_precision * 100:.1f}%"}
                            ]
                        },
                        {
                            "type": "divider"
                        },
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"*Events:* CB: {metrics.circuit_breaker_triggers} | Kill: {metrics.kill_switch_activations} | Updates: {metrics.model_updates}"
                            }
                        }
                    ]
                }
            ]
        }
    
    def generate_discord_message(self, metrics: DailyMetrics) -> Dict:
        """Generate Discord-formatted message."""
        pnl_color = 0x36a64f if metrics.net_pnl >= 0 else 0xff0000
        
        return {
            "embeds": [
                {
                    "title": f"📊 Daily Trading Report - {metrics.date}",
                    "color": pnl_color,
                    "fields": [
                        {"name": "💰 Net P&L", "value": f"${metrics.net_pnl:,.2f}", "inline": True},
                        {"name": "📈 Win Rate", "value": f"{metrics.win_rate * 100:.1f}%", "inline": True},
                        {"name": "📊 Sharpe (30d)", "value": f"{metrics.sharpe_30d:.2f}", "inline": True},
                        {"name": "🔢 Trades", "value": str(metrics.total_trades), "inline": True},
                        {"name": "⚡ Slippage", "value": f"{metrics.slippage_realized_bps:.1f} bps", "inline": True},
                        {"name": "📉 Max DD (7d)", "value": f"{metrics.max_drawdown_7d * 100:.2f}%", "inline": True},
                        {"name": "🎯 Fill Rate", "value": f"{metrics.fill_rate * 100:.1f}%", "inline": True},
                        {"name": "🤖 Model", "value": f"{metrics.model_precision * 100:.1f}%", "inline": True},
                        {"name": "🚨 Events", "value": f"CB: {metrics.circuit_breaker_triggers} | Kill: {metrics.kill_switch_activations}", "inline": True}
                    ],
                    "footer": {"text": f"Generated at {datetime.now().strftime('%H:%M:%S')} UTC"}
                }
            ]
        }
    
    async def send_slack(self, metrics: DailyMetrics) -> bool:
        """Send report to Slack."""
        if not self.slack_webhook:
            logger.warning("No Slack webhook configured")
            return False
        
        try:
            message = self.generate_slack_message(metrics)
            response = requests.post(
                self.slack_webhook,
                json=message,
                timeout=30
            )
            if response.status_code == 200:
                logger.info("Slack report sent")
                return True
            else:
                logger.error(f"Slack send failed: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Slack error: {e}")
            return False
    
    async def send_discord(self, metrics: DailyMetrics) -> bool:
        """Send report to Discord."""
        if not self.discord_webhook:
            logger.warning("No Discord webhook configured")
            return False
        
        try:
            message = self.generate_discord_message(metrics)
            response = requests.post(
                self.discord_webhook,
                json=message,
                timeout=30
            )
            if response.status_code in [200, 204]:
                logger.info("Discord report sent")
                return True
            else:
                logger.error(f"Discord send failed: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Discord error: {e}")
            return False
    
    def save_report(self, metrics: DailyMetrics, markdown: str) -> Path:
        """Save report to file."""
        report_file = self.reports_dir / f"daily_report_{metrics.date}.md"
        with open(report_file, 'w') as f:
            f.write(markdown)
        
        # Also save JSON
        json_file = self.reports_dir / f"daily_report_{metrics.date}.json"
        with open(json_file, 'w') as f:
            json.dump(asdict(metrics), f, indent=2, default=str)
        
        logger.info(f"Report saved to {report_file}")
        return report_file
    
    async def generate_and_send(self, date: datetime = None) -> Dict:
        """Generate report and send to all channels."""
        metrics = self.collect_metrics(date)
        markdown = self.generate_markdown_report(metrics)
        
        # Save locally
        report_file = self.save_report(metrics, markdown)
        
        # Send to channels
        results = {
            'date': metrics.date,
            'report_file': str(report_file),
            'slack_sent': await self.send_slack(metrics),
            'discord_sent': await self.send_discord(metrics)
        }
        
        print(markdown)
        
        return results


# =============================================================================
# CLI
# =============================================================================

async def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Daily Report Generator")
    parser.add_argument("--date", type=str, help="Report date (YYYY-MM-DD)")
    parser.add_argument("--no-send", action="store_true", help="Generate without sending")
    parser.add_argument("--output", type=str, help="Output directory")
    
    args = parser.parse_args()
    
    date = None
    if args.date:
        date = datetime.strptime(args.date, "%Y-%m-%d")
    
    generator = DailyReportGenerator(
        reports_dir=args.output or "./reports/daily"
    )
    
    if args.no_send:
        metrics = generator.collect_metrics(date)
        markdown = generator.generate_markdown_report(metrics)
        generator.save_report(metrics, markdown)
        print(markdown)
    else:
        results = await generator.generate_and_send(date)
        print(f"\nResults: {json.dumps(results, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())
