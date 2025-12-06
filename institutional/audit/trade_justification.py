"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║         TRADE JUSTIFICATION - Explainable Trade Decisions                     ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Generates human-readable justifications for every trade decision.
Used for regulatory compliance, internal review, and model debugging.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

logger = logging.getLogger("Audit.Justification")


@dataclass
class SignalContribution:
    """Contribution of a signal to the trade decision."""
    signal_name: str
    signal_value: float
    weight: float
    contribution: float  # value * weight
    direction: str  # 'bullish', 'bearish', 'neutral'
    description: str


@dataclass
class RiskCheck:
    """Result of a risk check."""
    check_name: str
    passed: bool
    value: float
    threshold: float
    description: str


@dataclass
class TradeJustification:
    """
    Complete justification for a trade decision.
    
    This document explains WHY a trade was made, which is essential for:
    1. Regulatory compliance
    2. Internal audit
    3. Model debugging
    4. Performance attribution
    """
    # Identification
    justification_id: str
    order_id: str
    timestamp: datetime
    
    # Decision summary
    decision: str  # 'enter_long', 'enter_short', 'exit', 'hold'
    symbol: str
    quantity: float
    price: float
    
    # Model information
    model_id: str
    model_version: str
    strategy_name: str
    
    # Market context
    regime: str
    regime_confidence: float
    volatility_regime: str
    trend_direction: str
    
    # Signal breakdown
    signals: List[SignalContribution] = field(default_factory=list)
    aggregate_signal: float = 0.0
    signal_threshold: float = 0.0
    
    # Risk checks
    risk_checks: List[RiskCheck] = field(default_factory=list)
    all_risk_checks_passed: bool = True
    
    # Order flow confirmation
    tvs_score: float = 0.0
    order_flow_confirmation: str = ""
    
    # Meta-learner input
    strategy_weight: float = 0.0
    weight_reason: str = ""
    
    # Execution rationale
    execution_algo: str = ""
    expected_slippage: float = 0.0
    urgency: str = "normal"  # 'low', 'normal', 'high', 'critical'
    
    # Human-readable summary
    summary: str = ""
    detailed_narrative: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'justification_id': self.justification_id,
            'order_id': self.order_id,
            'timestamp': self.timestamp.isoformat(),
            'decision': self.decision,
            'symbol': self.symbol,
            'quantity': self.quantity,
            'price': self.price,
            'model_id': self.model_id,
            'model_version': self.model_version,
            'strategy_name': self.strategy_name,
            'regime': self.regime,
            'regime_confidence': self.regime_confidence,
            'volatility_regime': self.volatility_regime,
            'trend_direction': self.trend_direction,
            'signals': [
                {
                    'name': s.signal_name,
                    'value': s.signal_value,
                    'weight': s.weight,
                    'contribution': s.contribution,
                    'direction': s.direction,
                    'description': s.description,
                }
                for s in self.signals
            ],
            'aggregate_signal': self.aggregate_signal,
            'signal_threshold': self.signal_threshold,
            'risk_checks': [
                {
                    'name': r.check_name,
                    'passed': r.passed,
                    'value': r.value,
                    'threshold': r.threshold,
                    'description': r.description,
                }
                for r in self.risk_checks
            ],
            'all_risk_checks_passed': self.all_risk_checks_passed,
            'tvs_score': self.tvs_score,
            'order_flow_confirmation': self.order_flow_confirmation,
            'strategy_weight': self.strategy_weight,
            'weight_reason': self.weight_reason,
            'execution_algo': self.execution_algo,
            'expected_slippage': self.expected_slippage,
            'urgency': self.urgency,
            'summary': self.summary,
            'detailed_narrative': self.detailed_narrative,
        }
    
    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            f"# Trade Justification: {self.order_id}",
            f"",
            f"**Generated:** {self.timestamp.isoformat()}",
            f"**Model:** {self.model_id} v{self.model_version}",
            f"**Strategy:** {self.strategy_name}",
            f"",
            f"## Decision Summary",
            f"",
            f"- **Action:** {self.decision.upper()}",
            f"- **Symbol:** {self.symbol}",
            f"- **Quantity:** {self.quantity}",
            f"- **Price:** {self.price}",
            f"",
            f"## Market Context",
            f"",
            f"| Factor | Value |",
            f"|--------|-------|",
            f"| Regime | {self.regime} ({self.regime_confidence:.1%}) |",
            f"| Volatility | {self.volatility_regime} |",
            f"| Trend | {self.trend_direction} |",
            f"",
            f"## Signal Breakdown",
            f"",
            f"| Signal | Value | Weight | Contribution | Direction |",
            f"|--------|-------|--------|--------------|-----------|",
        ]
        
        for s in self.signals:
            lines.append(
                f"| {s.signal_name} | {s.signal_value:.3f} | {s.weight:.2f} | "
                f"{s.contribution:.3f} | {s.direction} |"
            )
        
        lines.extend([
            f"",
            f"**Aggregate Signal:** {self.aggregate_signal:.3f} (threshold: {self.signal_threshold:.3f})",
            f"",
            f"## Risk Checks",
            f"",
            f"| Check | Status | Value | Threshold |",
            f"|-------|--------|-------|-----------|",
        ])
        
        for r in self.risk_checks:
            status = "✅ PASS" if r.passed else "❌ FAIL"
            lines.append(f"| {r.check_name} | {status} | {r.value:.3f} | {r.threshold:.3f} |")
        
        lines.extend([
            f"",
            f"## Order Flow Confirmation",
            f"",
            f"- **TVS Score:** {self.tvs_score:.2f}",
            f"- **Confirmation:** {self.order_flow_confirmation}",
            f"",
            f"## Execution Details",
            f"",
            f"- **Algorithm:** {self.execution_algo}",
            f"- **Expected Slippage:** {self.expected_slippage:.2f} bps",
            f"- **Urgency:** {self.urgency}",
            f"",
            f"## Summary",
            f"",
            f"{self.summary}",
            f"",
            f"---",
            f"*Justification ID: {self.justification_id}*",
        ])
        
        return "\n".join(lines)


class JustificationBuilder:
    """
    Builder for creating trade justifications.
    
    Usage:
        builder = JustificationBuilder()
        justification = (
            builder
            .order("ORDER123", "BTCUSD", 0.1, 50000)
            .model("momentum_v2", "2.1.0", "Momentum DNN")
            .market_context("trending", 0.85, "high", "bullish")
            .add_signal("RSI", 0.7, 0.3, "bullish", "RSI indicates oversold")
            .add_signal("MACD", 0.5, 0.4, "bullish", "MACD bullish crossover")
            .add_risk_check("position_limit", True, 0.5, 1.0, "Within limits")
            .order_flow(0.75, "Order flow confirms bullish bias")
            .execution("TWAP", 2.5, "normal")
            .build()
        )
    """
    
    def __init__(self):
        self._reset()
    
    def _reset(self):
        self._data = {
            'signals': [],
            'risk_checks': [],
        }
    
    def order(
        self,
        order_id: str,
        symbol: str,
        quantity: float,
        price: float,
        decision: str = "enter_long",
    ) -> 'JustificationBuilder':
        """Set order details."""
        self._data['order_id'] = order_id
        self._data['symbol'] = symbol
        self._data['quantity'] = quantity
        self._data['price'] = price
        self._data['decision'] = decision
        return self
    
    def model(
        self,
        model_id: str,
        model_version: str,
        strategy_name: str,
    ) -> 'JustificationBuilder':
        """Set model information."""
        self._data['model_id'] = model_id
        self._data['model_version'] = model_version
        self._data['strategy_name'] = strategy_name
        return self
    
    def market_context(
        self,
        regime: str,
        regime_confidence: float,
        volatility_regime: str,
        trend_direction: str,
    ) -> 'JustificationBuilder':
        """Set market context."""
        self._data['regime'] = regime
        self._data['regime_confidence'] = regime_confidence
        self._data['volatility_regime'] = volatility_regime
        self._data['trend_direction'] = trend_direction
        return self
    
    def add_signal(
        self,
        name: str,
        value: float,
        weight: float,
        direction: str,
        description: str,
    ) -> 'JustificationBuilder':
        """Add a signal contribution."""
        self._data['signals'].append(SignalContribution(
            signal_name=name,
            signal_value=value,
            weight=weight,
            contribution=value * weight,
            direction=direction,
            description=description,
        ))
        return self
    
    def add_risk_check(
        self,
        name: str,
        passed: bool,
        value: float,
        threshold: float,
        description: str,
    ) -> 'JustificationBuilder':
        """Add a risk check result."""
        self._data['risk_checks'].append(RiskCheck(
            check_name=name,
            passed=passed,
            value=value,
            threshold=threshold,
            description=description,
        ))
        return self
    
    def order_flow(
        self,
        tvs_score: float,
        confirmation: str,
    ) -> 'JustificationBuilder':
        """Set order flow confirmation."""
        self._data['tvs_score'] = tvs_score
        self._data['order_flow_confirmation'] = confirmation
        return self
    
    def strategy_allocation(
        self,
        weight: float,
        reason: str,
    ) -> 'JustificationBuilder':
        """Set strategy allocation details."""
        self._data['strategy_weight'] = weight
        self._data['weight_reason'] = reason
        return self
    
    def execution(
        self,
        algo: str,
        expected_slippage: float,
        urgency: str = "normal",
    ) -> 'JustificationBuilder':
        """Set execution details."""
        self._data['execution_algo'] = algo
        self._data['expected_slippage'] = expected_slippage
        self._data['urgency'] = urgency
        return self
    
    def build(self) -> TradeJustification:
        """Build the justification."""
        import uuid
        
        # Calculate aggregate signal
        signals = self._data.get('signals', [])
        aggregate = sum(s.contribution for s in signals)
        
        # Check all risk checks passed
        risk_checks = self._data.get('risk_checks', [])
        all_passed = all(r.passed for r in risk_checks)
        
        # Generate summary
        summary = self._generate_summary()
        
        justification = TradeJustification(
            justification_id=str(uuid.uuid4())[:8],
            order_id=self._data.get('order_id', 'UNKNOWN'),
            timestamp=datetime.now(),
            decision=self._data.get('decision', 'unknown'),
            symbol=self._data.get('symbol', 'UNKNOWN'),
            quantity=self._data.get('quantity', 0),
            price=self._data.get('price', 0),
            model_id=self._data.get('model_id', 'unknown'),
            model_version=self._data.get('model_version', '0.0.0'),
            strategy_name=self._data.get('strategy_name', 'unknown'),
            regime=self._data.get('regime', 'unknown'),
            regime_confidence=self._data.get('regime_confidence', 0),
            volatility_regime=self._data.get('volatility_regime', 'unknown'),
            trend_direction=self._data.get('trend_direction', 'unknown'),
            signals=signals,
            aggregate_signal=aggregate,
            signal_threshold=0.5,  # Default
            risk_checks=risk_checks,
            all_risk_checks_passed=all_passed,
            tvs_score=self._data.get('tvs_score', 0),
            order_flow_confirmation=self._data.get('order_flow_confirmation', ''),
            strategy_weight=self._data.get('strategy_weight', 0),
            weight_reason=self._data.get('weight_reason', ''),
            execution_algo=self._data.get('execution_algo', ''),
            expected_slippage=self._data.get('expected_slippage', 0),
            urgency=self._data.get('urgency', 'normal'),
            summary=summary,
        )
        
        self._reset()
        return justification
    
    def _generate_summary(self) -> str:
        """Generate human-readable summary."""
        parts = []
        
        decision = self._data.get('decision', 'unknown')
        symbol = self._data.get('symbol', 'UNKNOWN')
        quantity = self._data.get('quantity', 0)
        
        parts.append(f"Decision to {decision.replace('_', ' ')} {quantity} {symbol}.")
        
        regime = self._data.get('regime', 'unknown')
        trend = self._data.get('trend_direction', 'unknown')
        parts.append(f"Market is in {regime} regime with {trend} trend.")
        
        signals = self._data.get('signals', [])
        if signals:
            bullish = sum(1 for s in signals if s.direction == 'bullish')
            bearish = sum(1 for s in signals if s.direction == 'bearish')
            parts.append(f"Signals: {bullish} bullish, {bearish} bearish.")
        
        tvs = self._data.get('tvs_score', 0)
        parts.append(f"Order flow TVS: {tvs:.2f}.")
        
        risk_checks = self._data.get('risk_checks', [])
        passed = sum(1 for r in risk_checks if r.passed)
        total = len(risk_checks)
        parts.append(f"Risk checks: {passed}/{total} passed.")
        
        return " ".join(parts)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Build justification
    builder = JustificationBuilder()
    
    justification = (
        builder
        .order("ORDER001", "BTCUSD", 0.5, 50000, "enter_long")
        .model("momentum_v2", "2.1.0", "Momentum DNN")
        .market_context("trending", 0.85, "high", "bullish")
        .add_signal("RSI", 0.7, 0.3, "bullish", "RSI indicates oversold bounce")
        .add_signal("MACD", 0.6, 0.4, "bullish", "MACD bullish crossover")
        .add_signal("OBI", 0.55, 0.15, "bullish", "Order book imbalance toward bids")
        .add_signal("CVD", 0.4, 0.15, "neutral", "CVD slightly positive")
        .add_risk_check("position_limit", True, 0.3, 1.0, "Position within limits")
        .add_risk_check("drawdown", True, 0.02, 0.05, "Drawdown acceptable")
        .add_risk_check("volatility", True, 0.8, 1.5, "Volatility normal")
        .order_flow(0.78, "Strong bid absorption, whale accumulation detected")
        .strategy_allocation(0.25, "Trending regime favors momentum strategy")
        .execution("TWAP", 2.5, "normal")
        .build()
    )
    
    # Print JSON
    print("=== JSON Format ===")
    print(json.dumps(justification.to_dict(), indent=2))
    
    # Print Markdown
    print("\n=== Markdown Format ===")
    print(justification.to_markdown())
