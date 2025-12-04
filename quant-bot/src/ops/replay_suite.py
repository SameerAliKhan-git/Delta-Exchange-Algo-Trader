"""
Adverse Event Replay Suite
===========================
Re-run adverse trading events against the simulator for root cause analysis.

Purpose:
- Replay specific trades with full context
- Compare actual vs expected behavior
- Identify execution failures
- Test fixes before redeployment

Usage:
    python replay_suite.py --trade-id TRADE-123
    python replay_suite.py --date 2024-01-15 --symbol BTC-PERP
    python replay_suite.py --last-n 10 --filter "pnl < -100"
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TradeSnapshot:
    """Complete snapshot of a trade for replay."""
    trade_id: str
    timestamp: datetime
    symbol: str
    direction: str
    
    # Entry
    entry_price: float
    entry_size: float
    entry_signal_strength: float
    entry_regime: str
    
    # Exit
    exit_price: float
    exit_timestamp: datetime
    
    # P&L
    gross_pnl: float
    net_pnl: float
    slippage_bps: float
    expected_slippage_bps: float
    
    # Context
    orderflow_imbalance: float
    orderflow_approved: bool
    regime_approved: bool
    
    # Market state at entry
    market_state: Dict = field(default_factory=dict)
    
    # Model inputs
    features: Dict = field(default_factory=dict)
    model_version: str = ""
    model_prediction: float = 0.0
    
    # Orderbook snapshot
    orderbook_snapshot: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        d['exit_timestamp'] = self.exit_timestamp.isoformat()
        return d


@dataclass
class ReplayResult:
    """Result of replaying a trade."""
    trade_id: str
    original_pnl: float
    simulated_pnl: float
    pnl_deviation: float
    
    original_slippage: float
    simulated_slippage: float
    slippage_deviation: float
    
    # Gate decisions
    original_orderflow_approved: bool
    simulated_orderflow_approved: bool
    original_regime_approved: bool
    simulated_regime_approved: bool
    
    # Analysis
    root_cause_hypothesis: str
    recommendations: List[str]
    
    # Detailed comparison
    comparison_details: Dict = field(default_factory=dict)


class TradeReplaySuite:
    """
    Replay adverse trades for analysis.
    
    Loads trade snapshots and re-executes through the simulation
    environment to identify discrepancies.
    """
    
    def __init__(
        self,
        trade_log_dir: str = "./data/trades",
        market_data_dir: str = "./data/market",
        replay_output_dir: str = "./reports/replay"
    ):
        self.trade_log_dir = Path(trade_log_dir)
        self.market_data_dir = Path(market_data_dir)
        self.replay_output_dir = Path(replay_output_dir)
        self.replay_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize simulator components
        self._orderflow_gate = None
        self._regime_gate = None
        self._impact_model = None
        self._cost_analyzer = None
    
    async def initialize_simulators(self):
        """Initialize simulation components."""
        try:
            from ..execution.orderflow_gate import OrderFlowGate
            self._orderflow_gate = OrderFlowGate()
        except ImportError:
            logger.warning("OrderFlowGate not available")
        
        try:
            from ..strategies.regime_gate import RegimeGate
            self._regime_gate = RegimeGate()
        except ImportError:
            logger.warning("RegimeGate not available")
        
        try:
            from ..execution.almgren_chriss import AlmgrenChrissModel
            self._impact_model = AlmgrenChrissModel()
        except ImportError:
            logger.warning("AlmgrenChrissModel not available")
        
        try:
            from ..utils.cost_sensitivity import CostSensitivityAnalyzer
            self._cost_analyzer = CostSensitivityAnalyzer()
        except ImportError:
            logger.warning("CostSensitivityAnalyzer not available")
    
    def load_trade(self, trade_id: str) -> Optional[TradeSnapshot]:
        """Load a specific trade by ID."""
        # Search for trade in log files
        for log_file in self.trade_log_dir.glob("*.json"):
            try:
                with open(log_file, 'r') as f:
                    trades = json.load(f)
                
                if isinstance(trades, list):
                    for t in trades:
                        if t.get('trade_id') == trade_id:
                            return self._parse_trade(t)
                elif isinstance(trades, dict):
                    if trades.get('trade_id') == trade_id:
                        return self._parse_trade(trades)
            except Exception as e:
                logger.warning(f"Error reading {log_file}: {e}")
        
        return None
    
    def load_trades_by_date(
        self,
        date: datetime,
        symbol: str = None
    ) -> List[TradeSnapshot]:
        """Load all trades for a specific date."""
        date_str = date.strftime("%Y-%m-%d")
        trades = []
        
        log_file = self.trade_log_dir / f"trades_{date_str}.json"
        if log_file.exists():
            with open(log_file, 'r') as f:
                data = json.load(f)
            
            for t in data if isinstance(data, list) else [data]:
                trade = self._parse_trade(t)
                if trade and (not symbol or trade.symbol == symbol):
                    trades.append(trade)
        
        return trades
    
    def load_adverse_trades(
        self,
        days: int = 7,
        pnl_threshold: float = -100
    ) -> List[TradeSnapshot]:
        """Load trades with P&L below threshold."""
        adverse = []
        
        for i in range(days):
            date = datetime.now() - timedelta(days=i)
            trades = self.load_trades_by_date(date)
            
            for trade in trades:
                if trade.net_pnl < pnl_threshold:
                    adverse.append(trade)
        
        # Sort by P&L (worst first)
        adverse.sort(key=lambda t: t.net_pnl)
        
        return adverse
    
    def _parse_trade(self, data: Dict) -> Optional[TradeSnapshot]:
        """Parse trade data into TradeSnapshot."""
        try:
            return TradeSnapshot(
                trade_id=data.get('trade_id', ''),
                timestamp=datetime.fromisoformat(data.get('timestamp', '')),
                symbol=data.get('symbol', ''),
                direction=data.get('direction', ''),
                entry_price=data.get('entry_price', 0),
                entry_size=data.get('entry_size', 0),
                entry_signal_strength=data.get('signal_strength', 0),
                entry_regime=data.get('regime', ''),
                exit_price=data.get('exit_price', 0),
                exit_timestamp=datetime.fromisoformat(data.get('exit_timestamp', data.get('timestamp', ''))),
                gross_pnl=data.get('gross_pnl', 0),
                net_pnl=data.get('net_pnl', 0),
                slippage_bps=data.get('slippage_bps', 0),
                expected_slippage_bps=data.get('expected_slippage_bps', 0),
                orderflow_imbalance=data.get('orderflow_imbalance', 0),
                orderflow_approved=data.get('orderflow_approved', True),
                regime_approved=data.get('regime_approved', True),
                market_state=data.get('market_state', {}),
                features=data.get('features', {}),
                model_version=data.get('model_version', ''),
                model_prediction=data.get('model_prediction', 0),
                orderbook_snapshot=data.get('orderbook', {})
            )
        except Exception as e:
            logger.warning(f"Failed to parse trade: {e}")
            return None
    
    async def replay_trade(self, trade: TradeSnapshot) -> ReplayResult:
        """Replay a single trade through the simulator."""
        
        logger.info(f"Replaying trade {trade.trade_id}...")
        
        # 1. Re-evaluate order flow gate
        simulated_orderflow_approved = trade.orderflow_approved
        if self._orderflow_gate and trade.orderbook_snapshot:
            try:
                approved, reason, confidence = await self._orderflow_gate.should_trade(
                    symbol=trade.symbol,
                    direction=trade.direction,
                    orderbook=trade.orderbook_snapshot,
                    market_data=trade.market_state
                )
                simulated_orderflow_approved = approved
            except Exception as e:
                logger.warning(f"Order flow gate replay failed: {e}")
        
        # 2. Re-evaluate regime gate
        simulated_regime_approved = trade.regime_approved
        if self._regime_gate and trade.market_state:
            try:
                approved, reason, regime = self._regime_gate.is_strategy_allowed(
                    strategy_name=trade.market_state.get('strategy', 'momentum'),
                    market_data=trade.market_state
                )
                simulated_regime_approved = approved
            except Exception as e:
                logger.warning(f"Regime gate replay failed: {e}")
        
        # 3. Re-calculate expected slippage
        simulated_slippage = trade.expected_slippage_bps
        if self._impact_model:
            try:
                impact = self._impact_model.calculate_total_impact(
                    order_size=trade.entry_size * trade.entry_price,
                    adv=trade.market_state.get('daily_volume', 10000000),
                    volatility=trade.market_state.get('volatility', 0.02),
                    execution_time=5
                )
                simulated_slippage = impact.get('total_impact_bps', trade.expected_slippage_bps)
            except Exception as e:
                logger.warning(f"Impact model replay failed: {e}")
        
        # 4. Calculate simulated P&L
        price_move = trade.exit_price - trade.entry_price
        if trade.direction == 'short':
            price_move = -price_move
        
        gross_pnl = price_move * trade.entry_size
        simulated_cost = trade.entry_size * trade.entry_price * simulated_slippage / 10000
        simulated_pnl = gross_pnl - simulated_cost
        
        # 5. Analyze discrepancies
        pnl_deviation = (trade.net_pnl - simulated_pnl) / abs(trade.net_pnl) if trade.net_pnl != 0 else 0
        slippage_deviation = (trade.slippage_bps - simulated_slippage) / simulated_slippage if simulated_slippage > 0 else 0
        
        # 6. Generate root cause hypothesis
        root_cause = self._analyze_root_cause(
            trade=trade,
            simulated_pnl=simulated_pnl,
            simulated_slippage=simulated_slippage,
            simulated_orderflow_approved=simulated_orderflow_approved,
            simulated_regime_approved=simulated_regime_approved
        )
        
        # 7. Generate recommendations
        recommendations = self._generate_recommendations(
            trade=trade,
            root_cause=root_cause,
            pnl_deviation=pnl_deviation,
            slippage_deviation=slippage_deviation
        )
        
        return ReplayResult(
            trade_id=trade.trade_id,
            original_pnl=trade.net_pnl,
            simulated_pnl=simulated_pnl,
            pnl_deviation=pnl_deviation,
            original_slippage=trade.slippage_bps,
            simulated_slippage=simulated_slippage,
            slippage_deviation=slippage_deviation,
            original_orderflow_approved=trade.orderflow_approved,
            simulated_orderflow_approved=simulated_orderflow_approved,
            original_regime_approved=trade.regime_approved,
            simulated_regime_approved=simulated_regime_approved,
            root_cause_hypothesis=root_cause,
            recommendations=recommendations,
            comparison_details={
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'direction': trade.direction,
                'size': trade.entry_size,
                'regime': trade.entry_regime,
                'signal_strength': trade.entry_signal_strength,
                'orderflow_imbalance': trade.orderflow_imbalance
            }
        )
    
    def _analyze_root_cause(
        self,
        trade: TradeSnapshot,
        simulated_pnl: float,
        simulated_slippage: float,
        simulated_orderflow_approved: bool,
        simulated_regime_approved: bool
    ) -> str:
        """Analyze potential root cause of adverse trade."""
        
        causes = []
        
        # Order flow gate disagreement
        if trade.orderflow_approved and not simulated_orderflow_approved:
            causes.append("Order flow gate should have blocked this trade")
        
        # Regime gate disagreement
        if trade.regime_approved and not simulated_regime_approved:
            causes.append("Regime gate should have blocked this trade")
        
        # Slippage significantly higher than expected
        if trade.slippage_bps > simulated_slippage * 1.5:
            causes.append(f"Execution slippage {trade.slippage_bps:.1f}bps >> expected {simulated_slippage:.1f}bps")
        
        # Adverse signal
        if trade.entry_signal_strength < 0.5:
            causes.append(f"Weak signal strength ({trade.entry_signal_strength:.2f})")
        
        # Adverse order flow
        if trade.direction == 'long' and trade.orderflow_imbalance < 0:
            causes.append(f"Long trade against negative order flow ({trade.orderflow_imbalance:.2f})")
        elif trade.direction == 'short' and trade.orderflow_imbalance > 0:
            causes.append(f"Short trade against positive order flow ({trade.orderflow_imbalance:.2f})")
        
        # Large position relative to ADV
        adv = trade.market_state.get('daily_volume', 10000000)
        trade_value = trade.entry_size * trade.entry_price
        if adv > 0 and trade_value / adv > 0.01:
            causes.append(f"Large position ({trade_value/adv*100:.1f}% of ADV)")
        
        if not causes:
            causes.append("No clear root cause identified - may be market noise")
        
        return "; ".join(causes)
    
    def _generate_recommendations(
        self,
        trade: TradeSnapshot,
        root_cause: str,
        pnl_deviation: float,
        slippage_deviation: float
    ) -> List[str]:
        """Generate recommendations based on analysis."""
        
        recommendations = []
        
        if "order flow gate should have blocked" in root_cause.lower():
            recommendations.append("Review order flow gate threshold settings")
            recommendations.append("Consider tightening imbalance requirements")
        
        if "regime gate should have blocked" in root_cause.lower():
            recommendations.append("Review strategy-regime compatibility matrix")
            recommendations.append("Consider adding more regime-specific filters")
        
        if "slippage" in root_cause.lower():
            recommendations.append("Review Almgren-Chriss model calibration")
            recommendations.append("Consider reducing position sizes in low liquidity")
            recommendations.append("Evaluate alternative execution venues")
        
        if "weak signal" in root_cause.lower():
            recommendations.append("Increase minimum signal threshold")
            recommendations.append("Add signal strength to position sizing")
        
        if "against" in root_cause.lower() and "flow" in root_cause.lower():
            recommendations.append("CRITICAL: Order flow gate is not blocking adverse trades")
            recommendations.append("Verify order flow gate is in execution path")
        
        if abs(slippage_deviation) > 0.5:
            recommendations.append("Large slippage model error - recalibrate")
        
        if not recommendations:
            recommendations.append("Monitor for similar patterns")
            recommendations.append("Consider adding to pattern recognition")
        
        return recommendations
    
    async def replay_multiple(
        self,
        trades: List[TradeSnapshot]
    ) -> List[ReplayResult]:
        """Replay multiple trades."""
        results = []
        
        for trade in trades:
            result = await self.replay_trade(trade)
            results.append(result)
        
        return results
    
    def generate_replay_report(
        self,
        results: List[ReplayResult]
    ) -> str:
        """Generate replay analysis report."""
        
        if not results:
            return "No trades to analyze."
        
        lines = [
            "=" * 70,
            "ADVERSE EVENT REPLAY REPORT",
            "=" * 70,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Trades Analyzed: {len(results)}",
            "",
        ]
        
        # Summary statistics
        total_original_pnl = sum(r.original_pnl for r in results)
        total_simulated_pnl = sum(r.simulated_pnl for r in results)
        avg_slippage_deviation = np.mean([abs(r.slippage_deviation) for r in results])
        
        # Gate disagreements
        orderflow_disagreements = sum(
            1 for r in results 
            if r.original_orderflow_approved != r.simulated_orderflow_approved
        )
        regime_disagreements = sum(
            1 for r in results 
            if r.original_regime_approved != r.simulated_regime_approved
        )
        
        lines.extend([
            "-" * 70,
            "SUMMARY",
            "-" * 70,
            f"Total Original P&L: ${total_original_pnl:,.2f}",
            f"Total Simulated P&L: ${total_simulated_pnl:,.2f}",
            f"Unexplained Loss: ${total_original_pnl - total_simulated_pnl:,.2f}",
            f"Avg Slippage Deviation: {avg_slippage_deviation * 100:.1f}%",
            f"Order Flow Gate Disagreements: {orderflow_disagreements}",
            f"Regime Gate Disagreements: {regime_disagreements}",
            "",
        ])
        
        # Root cause summary
        all_causes = []
        for r in results:
            all_causes.extend(r.root_cause_hypothesis.split("; "))
        
        cause_counts = {}
        for cause in all_causes:
            cause_counts[cause] = cause_counts.get(cause, 0) + 1
        
        lines.extend([
            "-" * 70,
            "ROOT CAUSE FREQUENCY",
            "-" * 70,
        ])
        for cause, count in sorted(cause_counts.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"  [{count}] {cause}")
        lines.append("")
        
        # Individual trade details
        lines.extend([
            "-" * 70,
            "INDIVIDUAL TRADE ANALYSIS",
            "-" * 70,
        ])
        
        for r in results[:20]:  # Top 20
            lines.extend([
                f"\n📍 Trade: {r.trade_id}",
                f"   Original P&L: ${r.original_pnl:,.2f}",
                f"   Simulated P&L: ${r.simulated_pnl:,.2f}",
                f"   Slippage: {r.original_slippage:.1f}bps (expected {r.simulated_slippage:.1f}bps)",
                f"   Order Flow: {'✅' if r.original_orderflow_approved else '❌'} → {'✅' if r.simulated_orderflow_approved else '❌'}",
                f"   Regime: {'✅' if r.original_regime_approved else '❌'} → {'✅' if r.simulated_regime_approved else '❌'}",
                f"   Root Cause: {r.root_cause_hypothesis}",
                f"   Recommendations:",
            ])
            for rec in r.recommendations[:3]:
                lines.append(f"      • {rec}")
        
        # Aggregate recommendations
        all_recs = []
        for r in results:
            all_recs.extend(r.recommendations)
        
        rec_counts = {}
        for rec in all_recs:
            rec_counts[rec] = rec_counts.get(rec, 0) + 1
        
        lines.extend([
            "",
            "-" * 70,
            "TOP RECOMMENDATIONS",
            "-" * 70,
        ])
        for rec, count in sorted(rec_counts.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"  [{count}] {rec}")
        
        lines.extend([
            "",
            "=" * 70,
            "END OF REPORT",
            "=" * 70,
        ])
        
        return "\n".join(lines)
    
    def save_replay_report(
        self,
        results: List[ReplayResult],
        output_name: str = None
    ) -> Path:
        """Save replay report to file."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = output_name or f"replay_report_{timestamp}"
        
        # Save markdown report
        report = self.generate_replay_report(results)
        md_file = self.replay_output_dir / f"{output_name}.md"
        with open(md_file, 'w') as f:
            f.write(report)
        
        # Save JSON data
        json_file = self.replay_output_dir / f"{output_name}.json"
        with open(json_file, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2, default=str)
        
        logger.info(f"Replay report saved to {md_file}")
        return md_file


# =============================================================================
# CLI
# =============================================================================

async def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Adverse Event Replay Suite")
    parser.add_argument("--trade-id", type=str, help="Replay specific trade")
    parser.add_argument("--date", type=str, help="Replay trades from date (YYYY-MM-DD)")
    parser.add_argument("--symbol", type=str, help="Filter by symbol")
    parser.add_argument("--last-n", type=int, default=10, help="Replay last N adverse trades")
    parser.add_argument("--threshold", type=float, default=-100, help="P&L threshold for adverse")
    parser.add_argument("--days", type=int, default=7, help="Days to look back")
    parser.add_argument("--output", type=str, help="Output report name")
    
    args = parser.parse_args()
    
    suite = TradeReplaySuite()
    await suite.initialize_simulators()
    
    trades = []
    
    if args.trade_id:
        trade = suite.load_trade(args.trade_id)
        if trade:
            trades = [trade]
        else:
            print(f"Trade {args.trade_id} not found")
            return
    elif args.date:
        date = datetime.strptime(args.date, "%Y-%m-%d")
        trades = suite.load_trades_by_date(date, args.symbol)
    else:
        trades = suite.load_adverse_trades(
            days=args.days,
            pnl_threshold=args.threshold
        )[:args.last_n]
    
    if not trades:
        print("No trades found matching criteria")
        
        # Generate sample trade for demo
        print("\nGenerating sample trade for demo...")
        sample_trade = TradeSnapshot(
            trade_id="DEMO-001",
            timestamp=datetime.now() - timedelta(hours=2),
            symbol="BTC-PERP",
            direction="long",
            entry_price=50000,
            entry_size=0.1,
            entry_signal_strength=0.4,
            entry_regime="VOLATILE",
            exit_price=49800,
            exit_timestamp=datetime.now() - timedelta(hours=1),
            gross_pnl=-20,
            net_pnl=-35,
            slippage_bps=30,
            expected_slippage_bps=10,
            orderflow_imbalance=-0.3,
            orderflow_approved=True,
            regime_approved=True,
            market_state={'daily_volume': 100000000, 'volatility': 0.03, 'strategy': 'momentum'}
        )
        trades = [sample_trade]
    
    print(f"\nReplaying {len(trades)} trades...")
    results = await suite.replay_multiple(trades)
    
    report = suite.generate_replay_report(results)
    print(report)
    
    suite.save_replay_report(results, args.output)


if __name__ == "__main__":
    asyncio.run(main())
