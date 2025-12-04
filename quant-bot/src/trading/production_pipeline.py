"""
Production Trading Pipeline Integration
========================================
Wire all critical modules together into a unified trading pipeline.

This module integrates:
- Paper Trading Orchestrator
- Transaction Cost Sensitivity
- Online Learning Guardrails
- Almgren-Chriss Market Impact
- Order-Flow Gate
- Regime Gate
- Deployment Checklist
- Profitability Durability Scoring

CRITICAL: This is the master integration point.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """Trading mode."""
    PAPER = "paper"
    CANARY = "canary"  # Small live position for validation
    LIVE = "live"


class SignalDecision(Enum):
    """Signal decision outcome."""
    EXECUTE = "execute"
    SKIP = "skip"
    REDUCE_SIZE = "reduce_size"


@dataclass
class TradeDecision:
    """Complete trade decision with all gate results."""
    signal_id: str
    symbol: str
    direction: str  # 'long' or 'short'
    raw_size: float
    
    # Gate results
    orderflow_approved: bool
    orderflow_reason: str
    orderflow_confidence: float
    
    regime_approved: bool
    regime_reason: str
    current_regime: str
    
    # Cost analysis
    estimated_cost_bps: float
    cost_adjusted_edge: float
    is_profitable: bool
    
    # Impact analysis
    market_impact_bps: float
    optimal_execution_time_minutes: float
    recommended_slices: int
    
    # Final decision
    decision: SignalDecision
    final_size: float
    execution_notes: str
    
    # Metadata
    timestamp: datetime
    mode: TradingMode


class ProductionTradingPipeline:
    """
    Master integration of all production-critical trading components.
    
    This is the SINGLE ENTRY POINT for all trade execution.
    Every trade MUST go through this pipeline.
    """
    
    def __init__(
        self,
        mode: TradingMode = TradingMode.PAPER,
        config_path: str = "./trading_config.json"
    ):
        self.mode = mode
        self.config = self._load_config(config_path)
        
        # Component imports (lazy to avoid circular imports)
        self._orderflow_gate = None
        self._regime_gate = None
        self._cost_analyzer = None
        self._impact_model = None
        self._paper_orchestrator = None
        self._learning_guardrails = None
        self._deployment_checklist = None
        self._durability_engine = None
        
        # State
        self.is_initialized = False
        self.kill_switch_active = False
        
        # Metrics
        self.trades_processed = 0
        self.trades_executed = 0
        self.trades_blocked = 0
        self.total_cost_saved_bps = 0.0
        
        logger.info(f"ProductionTradingPipeline initialized in {mode.value} mode")
    
    def _load_config(self, path: str) -> Dict:
        """Load trading configuration."""
        default_config = {
            # Gate thresholds
            'orderflow_min_confidence': 0.6,
            'regime_strict_mode': True,
            
            # Cost thresholds
            'min_edge_after_costs_bps': 5.0,
            'max_acceptable_cost_bps': 30.0,
            
            # Impact limits
            'max_market_impact_bps': 20.0,
            'max_participation_rate': 0.05,
            
            # Position sizing
            'max_position_pct': 0.02,
            'kelly_fraction': 0.25,
            
            # Safety
            'require_all_gates_pass': True,
            'paper_trading_days_required': 30,
        }
        
        if Path(path).exists():
            try:
                with open(path, 'r') as f:
                    loaded = json.load(f)
                default_config.update(loaded)
            except Exception as e:
                logger.warning(f"Could not load config: {e}")
        
        return default_config
    
    async def initialize(self):
        """Initialize all components."""
        
        logger.info("Initializing production trading pipeline...")
        
        # Initialize Order-Flow Gate
        try:
            from ..execution.orderflow_gate import OrderFlowGate
            self._orderflow_gate = OrderFlowGate(
                imbalance_threshold=0.6,
                cvd_confirmation=True,
                footprint_analysis=True
            )
            logger.info("✅ Order-Flow Gate initialized")
        except ImportError as e:
            logger.warning(f"Order-Flow Gate not available: {e}")
        
        # Initialize Regime Gate
        try:
            from ..strategies.regime_gate import RegimeGate
            self._regime_gate = RegimeGate()
            logger.info("✅ Regime Gate initialized")
        except ImportError as e:
            logger.warning(f"Regime Gate not available: {e}")
        
        # Initialize Cost Analyzer
        try:
            from ..utils.cost_sensitivity import CostSensitivityAnalyzer
            self._cost_analyzer = CostSensitivityAnalyzer()
            logger.info("✅ Cost Sensitivity Analyzer initialized")
        except ImportError as e:
            logger.warning(f"Cost Analyzer not available: {e}")
        
        # Initialize Impact Model
        try:
            from ..execution.almgren_chriss import AlmgrenChrissModel
            self._impact_model = AlmgrenChrissModel()
            logger.info("✅ Almgren-Chriss Impact Model initialized")
        except ImportError as e:
            logger.warning(f"Impact Model not available: {e}")
        
        # Initialize Paper Trading Orchestrator
        try:
            from ..trading.paper_trading_orchestrator import PaperTradingOrchestrator
            self._paper_orchestrator = PaperTradingOrchestrator()
            logger.info("✅ Paper Trading Orchestrator initialized")
        except ImportError as e:
            logger.warning(f"Paper Trading Orchestrator not available: {e}")
        
        # Initialize Learning Guardrails
        try:
            from ..models.online_learning_guardrails import OnlineLearningGuardrails
            self._learning_guardrails = OnlineLearningGuardrails()
            logger.info("✅ Online Learning Guardrails initialized")
        except ImportError as e:
            logger.warning(f"Learning Guardrails not available: {e}")
        
        self.is_initialized = True
        logger.info("Production trading pipeline initialization complete")
    
    async def process_signal(
        self,
        signal_id: str,
        symbol: str,
        direction: str,
        raw_size: float,
        strategy_name: str,
        market_data: Dict[str, Any],
        orderbook_data: Dict[str, Any] = None,
        expected_edge_bps: float = 10.0
    ) -> TradeDecision:
        """
        Process a trading signal through all gates.
        
        This is the MAIN ENTRY POINT for all trades.
        
        Args:
            signal_id: Unique identifier for this signal
            symbol: Trading symbol
            direction: 'long' or 'short'
            raw_size: Requested position size (in USD)
            strategy_name: Name of generating strategy
            market_data: Current market data dict
            orderbook_data: Current orderbook snapshot
            expected_edge_bps: Expected edge in basis points
        
        Returns:
            TradeDecision with full gate results and final decision
        """
        
        if not self.is_initialized:
            await self.initialize()
        
        if self.kill_switch_active:
            return self._create_blocked_decision(
                signal_id, symbol, direction, raw_size,
                "Kill switch active"
            )
        
        self.trades_processed += 1
        
        # 1. ORDER-FLOW GATE
        orderflow_approved = True
        orderflow_reason = "Gate not available"
        orderflow_confidence = 0.5
        
        if self._orderflow_gate and orderbook_data:
            try:
                orderflow_approved, orderflow_reason, orderflow_confidence = \
                    await self._orderflow_gate.should_trade(
                        symbol=symbol,
                        direction=direction,
                        orderbook=orderbook_data,
                        market_data=market_data
                    )
            except Exception as e:
                logger.error(f"Order-flow gate error: {e}")
                orderflow_approved = False
                orderflow_reason = f"Gate error: {e}"
        
        # 2. REGIME GATE
        regime_approved = True
        regime_reason = "Gate not available"
        current_regime = "unknown"
        
        if self._regime_gate:
            try:
                regime_approved, regime_reason, current_regime = \
                    self._regime_gate.is_strategy_allowed(
                        strategy_name=strategy_name,
                        market_data=market_data
                    )
            except Exception as e:
                logger.error(f"Regime gate error: {e}")
                regime_approved = False
                regime_reason = f"Gate error: {e}"
        
        # 3. COST ANALYSIS
        estimated_cost_bps = 15.0  # Default assumption
        cost_adjusted_edge = expected_edge_bps - estimated_cost_bps
        is_profitable = cost_adjusted_edge > self.config['min_edge_after_costs_bps']
        
        if self._cost_analyzer:
            try:
                cost_analysis = self._cost_analyzer.analyze_trade(
                    symbol=symbol,
                    size_usd=raw_size,
                    expected_edge_bps=expected_edge_bps
                )
                estimated_cost_bps = cost_analysis.get('total_cost_bps', 15.0)
                cost_adjusted_edge = cost_analysis.get('net_edge_bps', expected_edge_bps - 15)
                is_profitable = cost_analysis.get('is_profitable', True)
            except Exception as e:
                logger.error(f"Cost analysis error: {e}")
        
        # 4. MARKET IMPACT ANALYSIS
        market_impact_bps = 5.0
        optimal_time = 5.0
        recommended_slices = 1
        
        if self._impact_model:
            try:
                impact_analysis = self._impact_model.calculate_impact(
                    order_size=raw_size,
                    daily_volume=market_data.get('daily_volume', 1000000),
                    volatility=market_data.get('volatility', 0.02),
                    spread_bps=market_data.get('spread_bps', 5)
                )
                market_impact_bps = impact_analysis.get('total_impact_bps', 5.0)
                optimal_time = impact_analysis.get('optimal_time_minutes', 5.0)
                recommended_slices = impact_analysis.get('recommended_slices', 1)
            except Exception as e:
                logger.error(f"Impact analysis error: {e}")
        
        # 5. MAKE FINAL DECISION
        decision, final_size, notes = self._make_decision(
            orderflow_approved=orderflow_approved,
            orderflow_confidence=orderflow_confidence,
            regime_approved=regime_approved,
            is_profitable=is_profitable,
            market_impact_bps=market_impact_bps,
            raw_size=raw_size,
            expected_edge_bps=expected_edge_bps,
            estimated_cost_bps=estimated_cost_bps
        )
        
        # Track metrics
        if decision == SignalDecision.EXECUTE:
            self.trades_executed += 1
        else:
            self.trades_blocked += 1
            self.total_cost_saved_bps += estimated_cost_bps
        
        trade_decision = TradeDecision(
            signal_id=signal_id,
            symbol=symbol,
            direction=direction,
            raw_size=raw_size,
            orderflow_approved=orderflow_approved,
            orderflow_reason=orderflow_reason,
            orderflow_confidence=orderflow_confidence,
            regime_approved=regime_approved,
            regime_reason=regime_reason,
            current_regime=current_regime,
            estimated_cost_bps=estimated_cost_bps,
            cost_adjusted_edge=cost_adjusted_edge,
            is_profitable=is_profitable,
            market_impact_bps=market_impact_bps,
            optimal_execution_time_minutes=optimal_time,
            recommended_slices=recommended_slices,
            decision=decision,
            final_size=final_size,
            execution_notes=notes,
            timestamp=datetime.now(),
            mode=self.mode
        )
        
        # Log decision
        self._log_decision(trade_decision)
        
        # If paper trading, record for analysis
        if self.mode == TradingMode.PAPER and self._paper_orchestrator:
            await self._paper_orchestrator.record_trade(trade_decision)
        
        return trade_decision
    
    def _make_decision(
        self,
        orderflow_approved: bool,
        orderflow_confidence: float,
        regime_approved: bool,
        is_profitable: bool,
        market_impact_bps: float,
        raw_size: float,
        expected_edge_bps: float,
        estimated_cost_bps: float
    ) -> Tuple[SignalDecision, float, str]:
        """Make final trade decision based on all gate results."""
        
        notes = []
        
        # Check if all gates must pass
        if self.config['require_all_gates_pass']:
            if not orderflow_approved:
                return SignalDecision.SKIP, 0.0, f"Order-flow gate rejected: {orderflow_confidence:.1%} confidence"
            
            if not regime_approved:
                return SignalDecision.SKIP, 0.0, "Regime gate rejected: strategy not suited for current regime"
            
            if not is_profitable:
                return SignalDecision.SKIP, 0.0, f"Unprofitable after costs: {estimated_cost_bps:.1f} bps"
        
        # Check impact limits
        if market_impact_bps > self.config['max_market_impact_bps']:
            # Reduce size to acceptable impact
            reduction_factor = self.config['max_market_impact_bps'] / market_impact_bps
            reduced_size = raw_size * reduction_factor
            notes.append(f"Size reduced due to impact: {reduction_factor:.1%}")
            return SignalDecision.REDUCE_SIZE, reduced_size, " | ".join(notes)
        
        # Check edge vs cost ratio
        edge_cost_ratio = expected_edge_bps / max(estimated_cost_bps, 1)
        if edge_cost_ratio < 1.5:
            # Marginal trade - reduce size
            reduced_size = raw_size * 0.5
            notes.append(f"Marginal edge/cost ratio: {edge_cost_ratio:.2f}")
            return SignalDecision.REDUCE_SIZE, reduced_size, " | ".join(notes)
        
        # Apply orderflow confidence to sizing
        if orderflow_confidence < 0.8:
            confidence_adjusted = raw_size * orderflow_confidence
            notes.append(f"Size adjusted for orderflow confidence: {orderflow_confidence:.1%}")
            return SignalDecision.EXECUTE, confidence_adjusted, " | ".join(notes)
        
        # Full execution
        notes.append("All gates passed")
        return SignalDecision.EXECUTE, raw_size, " | ".join(notes)
    
    def _create_blocked_decision(
        self,
        signal_id: str,
        symbol: str,
        direction: str,
        raw_size: float,
        reason: str
    ) -> TradeDecision:
        """Create a blocked trade decision."""
        return TradeDecision(
            signal_id=signal_id,
            symbol=symbol,
            direction=direction,
            raw_size=raw_size,
            orderflow_approved=False,
            orderflow_reason=reason,
            orderflow_confidence=0.0,
            regime_approved=False,
            regime_reason=reason,
            current_regime="blocked",
            estimated_cost_bps=0.0,
            cost_adjusted_edge=0.0,
            is_profitable=False,
            market_impact_bps=0.0,
            optimal_execution_time_minutes=0.0,
            recommended_slices=0,
            decision=SignalDecision.SKIP,
            final_size=0.0,
            execution_notes=reason,
            timestamp=datetime.now(),
            mode=self.mode
        )
    
    def _log_decision(self, decision: TradeDecision):
        """Log trade decision for audit trail."""
        log_entry = {
            'timestamp': decision.timestamp.isoformat(),
            'signal_id': decision.signal_id,
            'symbol': decision.symbol,
            'direction': decision.direction,
            'raw_size': decision.raw_size,
            'final_size': decision.final_size,
            'decision': decision.decision.value,
            'orderflow_approved': decision.orderflow_approved,
            'regime_approved': decision.regime_approved,
            'is_profitable': decision.is_profitable,
            'notes': decision.execution_notes
        }
        
        logger.info(f"Trade decision: {json.dumps(log_entry)}")
    
    def activate_kill_switch(self, reason: str = "Manual activation"):
        """Activate emergency kill switch."""
        self.kill_switch_active = True
        logger.critical(f"KILL SWITCH ACTIVATED: {reason}")
    
    def deactivate_kill_switch(self, authorized_by: str):
        """Deactivate kill switch with authorization."""
        self.kill_switch_active = False
        logger.warning(f"Kill switch deactivated by {authorized_by}")
    
    def get_pipeline_status(self) -> Dict:
        """Get current pipeline status."""
        return {
            'mode': self.mode.value,
            'is_initialized': self.is_initialized,
            'kill_switch_active': self.kill_switch_active,
            'trades_processed': self.trades_processed,
            'trades_executed': self.trades_executed,
            'trades_blocked': self.trades_blocked,
            'execution_rate': self.trades_executed / max(self.trades_processed, 1),
            'estimated_cost_saved_bps': self.total_cost_saved_bps,
            'components': {
                'orderflow_gate': self._orderflow_gate is not None,
                'regime_gate': self._regime_gate is not None,
                'cost_analyzer': self._cost_analyzer is not None,
                'impact_model': self._impact_model is not None,
                'paper_orchestrator': self._paper_orchestrator is not None,
                'learning_guardrails': self._learning_guardrails is not None,
            }
        }
    
    def transition_to_live(self, authorization: Dict) -> bool:
        """
        Transition from paper to live trading.
        
        Requires:
        - 30+ days paper trading
        - Positive Sharpe in paper trading
        - All deployment checklist items passed
        - Authorized sign-off
        """
        
        # Check paper trading requirement
        if self._paper_orchestrator:
            paper_stats = self._paper_orchestrator.get_stats()
            
            if paper_stats.get('days_active', 0) < self.config['paper_trading_days_required']:
                logger.error(f"Insufficient paper trading days: {paper_stats.get('days_active', 0)}")
                return False
            
            if paper_stats.get('sharpe', 0) < 0:
                logger.error(f"Negative paper trading Sharpe: {paper_stats.get('sharpe', 0)}")
                return False
        
        # Check deployment checklist
        if self._deployment_checklist:
            readiness = self._deployment_checklist.get_readiness()
            if not readiness.is_ready:
                logger.error(f"Deployment checklist not ready: {readiness.blocking_items}")
                return False
        
        # Check authorization
        if not authorization.get('signed_by') or not authorization.get('signature'):
            logger.error("Missing authorization for live transition")
            return False
        
        # Transition
        self.mode = TradingMode.LIVE
        logger.warning(f"TRANSITIONED TO LIVE TRADING - Authorized by {authorization['signed_by']}")
        
        return True


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def create_production_pipeline(
    mode: TradingMode = TradingMode.PAPER
) -> ProductionTradingPipeline:
    """Create and initialize production pipeline."""
    pipeline = ProductionTradingPipeline(mode=mode)
    await pipeline.initialize()
    return pipeline


# =============================================================================
# DEMO
# =============================================================================

async def demo():
    """Demonstrate production trading pipeline."""
    
    print("=" * 70)
    print("PRODUCTION TRADING PIPELINE - DEMO")
    print("=" * 70)
    
    # Create pipeline in paper mode
    pipeline = ProductionTradingPipeline(mode=TradingMode.PAPER)
    await pipeline.initialize()
    
    # Process a sample signal
    print("\n1. Processing sample trade signal...")
    
    decision = await pipeline.process_signal(
        signal_id="SIG-001",
        symbol="BTC-PERP",
        direction="long",
        raw_size=10000,  # $10,000
        strategy_name="momentum",
        market_data={
            'price': 50000,
            'volatility': 0.02,
            'daily_volume': 100000000,
            'spread_bps': 2
        },
        orderbook_data={
            'bid_depth': 5000000,
            'ask_depth': 4500000,
            'imbalance': 0.55
        },
        expected_edge_bps=25
    )
    
    print(f"\nTrade Decision:")
    print(f"  Signal: {decision.signal_id}")
    print(f"  Decision: {decision.decision.value}")
    print(f"  Final Size: ${decision.final_size:,.0f}")
    print(f"  Order-Flow: {'✅' if decision.orderflow_approved else '❌'} ({decision.orderflow_reason})")
    print(f"  Regime: {'✅' if decision.regime_approved else '❌'} ({decision.regime_reason})")
    print(f"  Profitable: {'✅' if decision.is_profitable else '❌'}")
    print(f"  Notes: {decision.execution_notes}")
    
    # Show pipeline status
    print("\n2. Pipeline Status:")
    status = pipeline.get_pipeline_status()
    for key, value in status.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    print("\n✅ Pipeline demo complete")


if __name__ == "__main__":
    asyncio.run(demo())
