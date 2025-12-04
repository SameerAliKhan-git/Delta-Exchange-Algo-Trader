"""
Integration Tests for Production Trading Pipeline
==================================================
Tests all critical modules together.
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestOrderFlowGate:
    """Test Order-Flow Gate functionality."""
    
    def test_imbalance_detection(self):
        """Test that imbalance is correctly detected."""
        from src.execution.orderflow_gate import OrderFlowGate
        
        gate = OrderFlowGate(
            imbalance_threshold=0.6,
            cvd_confirmation=True,
            footprint_analysis=False
        )
        
        # Strong buy imbalance
        orderbook = {
            'bids': [[50000, 100], [49900, 80], [49800, 60]],
            'asks': [[50100, 30], [50200, 20], [50300, 10]]
        }
        
        imbalance = gate._calculate_imbalance(orderbook)
        assert imbalance > 0.6, "Should detect strong buy imbalance"
    
    def test_gate_blocks_against_flow(self):
        """Test that gate blocks trades against order flow."""
        from src.execution.orderflow_gate import OrderFlowGate
        
        gate = OrderFlowGate(imbalance_threshold=0.6)
        
        # Strong sell imbalance but trying to go long
        result = gate.evaluate_sync(
            direction='long',
            imbalance=-0.7,  # Strong selling
            cvd_delta=-1000,
            footprint_signal=None
        )
        
        assert not result[0], "Should block long when strong sell pressure"
    
    def test_gate_allows_with_flow(self):
        """Test that gate allows trades with order flow."""
        from src.execution.orderflow_gate import OrderFlowGate
        
        gate = OrderFlowGate(imbalance_threshold=0.6)
        
        # Strong buy imbalance and going long
        result = gate.evaluate_sync(
            direction='long',
            imbalance=0.7,  # Strong buying
            cvd_delta=1000,
            footprint_signal=1
        )
        
        assert result[0], "Should allow long when strong buy pressure"


class TestRegimeGate:
    """Test Regime Gate functionality."""
    
    def test_regime_detection(self):
        """Test regime is correctly detected."""
        from src.strategies.regime_gate import RegimeGate
        
        gate = RegimeGate()
        
        # Create trending market data
        np.random.seed(42)
        prices = 50000 * (1 + np.cumsum(np.random.uniform(0.001, 0.005, 50)))
        
        market_data = {
            'prices': prices.tolist(),
            'returns': np.diff(prices) / prices[:-1],
            'volatility': 0.02
        }
        
        regime = gate.detect_regime(market_data)
        assert regime in ['TRENDING', 'MEAN_REVERTING', 'VOLATILE', 'CRISIS']
    
    def test_strategy_blocking(self):
        """Test that incompatible strategies are blocked."""
        from src.strategies.regime_gate import RegimeGate
        
        gate = RegimeGate()
        
        # Mean reversion strategy in trending market
        is_allowed, reason, _ = gate.is_strategy_allowed_sync(
            strategy_name='mean_reversion',
            regime='TRENDING'
        )
        
        assert not is_allowed, "Mean reversion should be blocked in trending"
    
    def test_compatible_strategies(self):
        """Test that compatible strategies are allowed."""
        from src.strategies.regime_gate import RegimeGate
        
        gate = RegimeGate()
        
        # Momentum strategy in trending market
        is_allowed, reason, _ = gate.is_strategy_allowed_sync(
            strategy_name='momentum',
            regime='TRENDING'
        )
        
        assert is_allowed, "Momentum should be allowed in trending"


class TestCostSensitivity:
    """Test Cost Sensitivity Analyzer."""
    
    def test_break_even_calculation(self):
        """Test break-even cost calculation."""
        from src.utils.cost_sensitivity import CostSensitivityAnalyzer
        
        analyzer = CostSensitivityAnalyzer()
        
        # Create returns with known characteristics
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        
        result = analyzer.calculate_break_even_cost(
            returns=returns,
            turnover_rate=1.0
        )
        
        assert result > 0, "Break-even cost should be positive"
        assert result < 100, "Break-even cost should be reasonable"
    
    def test_profitability_at_cost_levels(self):
        """Test profitability at different cost levels."""
        from src.utils.cost_sensitivity import CostSensitivityAnalyzer
        
        analyzer = CostSensitivityAnalyzer()
        
        # Returns with ~50 bps/day edge
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.005, 0.02, 100))
        
        # Should be profitable at low costs
        is_profitable_low = analyzer.is_profitable_at_cost(
            returns=returns,
            cost_bps=5,
            turnover_rate=1.0
        )
        assert is_profitable_low, "Should be profitable at 5 bps"
        
        # Should not be profitable at very high costs
        is_profitable_high = analyzer.is_profitable_at_cost(
            returns=returns,
            cost_bps=200,
            turnover_rate=1.0
        )
        assert not is_profitable_high, "Should not be profitable at 200 bps"


class TestAlmgrenChriss:
    """Test Almgren-Chriss Market Impact Model."""
    
    def test_impact_increases_with_size(self):
        """Test that impact increases with order size."""
        from src.execution.almgren_chriss import AlmgrenChrissModel
        
        model = AlmgrenChrissModel()
        
        small_impact = model.calculate_total_impact(
            order_size=10000,
            adv=10000000,
            volatility=0.02,
            execution_time=5
        )
        
        large_impact = model.calculate_total_impact(
            order_size=100000,
            adv=10000000,
            volatility=0.02,
            execution_time=5
        )
        
        assert large_impact > small_impact, "Larger orders should have more impact"
    
    def test_optimal_execution_time(self):
        """Test optimal execution time calculation."""
        from src.execution.almgren_chriss import AlmgrenChrissModel
        
        model = AlmgrenChrissModel()
        
        optimal_time = model.calculate_optimal_execution_time(
            order_size=50000,
            adv=10000000,
            volatility=0.02,
            urgency=0.5
        )
        
        assert optimal_time > 0, "Optimal time should be positive"
        assert optimal_time < 480, "Should complete within a trading day"
    
    def test_slicing_recommendation(self):
        """Test order slicing recommendation."""
        from src.execution.almgren_chriss import AlmgrenChrissModel
        
        model = AlmgrenChrissModel()
        
        slices = model.recommend_slices(
            order_size=100000,
            adv=5000000,
            min_slice_size=1000
        )
        
        assert slices >= 1, "Should have at least one slice"
        assert slices <= 100, "Should not have excessive slices"


class TestOnlineLearningGuardrails:
    """Test Online Learning Guardrails."""
    
    def test_update_throttling(self):
        """Test that updates are throttled correctly."""
        from src.models.online_learning_guardrails import UpdateThrottler
        
        throttler = UpdateThrottler(
            min_interval_seconds=60,
            max_updates_per_hour=10
        )
        
        # First update should be allowed
        assert throttler.can_update(), "First update should be allowed"
        throttler.record_update()
        
        # Immediate second update should be blocked
        assert not throttler.can_update(), "Immediate update should be blocked"
    
    def test_validation_gate(self):
        """Test validation gate functionality."""
        from src.models.online_learning_guardrails import ValidationGate
        
        gate = ValidationGate(
            min_sharpe_improvement=0.05,
            max_drawdown_increase=0.02
        )
        
        old_metrics = {'sharpe': 1.0, 'max_drawdown': 0.10}
        
        # Better model should pass
        better_metrics = {'sharpe': 1.2, 'max_drawdown': 0.09}
        assert gate.should_accept(old_metrics, better_metrics)
        
        # Worse model should fail
        worse_metrics = {'sharpe': 0.8, 'max_drawdown': 0.15}
        assert not gate.should_accept(old_metrics, worse_metrics)
    
    def test_automatic_rollback(self):
        """Test automatic rollback on degradation."""
        from src.models.online_learning_guardrails import OnlineLearningGuardrails
        
        guards = OnlineLearningGuardrails()
        
        # Simulate model update followed by poor performance
        guards.record_model_update("v1.0", metrics={'sharpe': 1.2})
        guards.record_model_update("v1.1", metrics={'sharpe': 0.5})
        
        should_rollback = guards.check_for_rollback(
            current_metrics={'sharpe': 0.3, 'drawdown': 0.2}
        )
        
        assert should_rollback, "Should trigger rollback on poor performance"


class TestProductionPipeline:
    """Test full production pipeline integration."""
    
    @pytest.mark.asyncio
    async def test_pipeline_initialization(self):
        """Test pipeline initializes correctly."""
        from src.trading.production_pipeline import ProductionTradingPipeline, TradingMode
        
        pipeline = ProductionTradingPipeline(mode=TradingMode.PAPER)
        await pipeline.initialize()
        
        assert pipeline.is_initialized, "Pipeline should be initialized"
        assert not pipeline.kill_switch_active, "Kill switch should be off"
    
    @pytest.mark.asyncio
    async def test_signal_processing(self):
        """Test signal processing through pipeline."""
        from src.trading.production_pipeline import ProductionTradingPipeline, TradingMode, SignalDecision
        
        pipeline = ProductionTradingPipeline(mode=TradingMode.PAPER)
        await pipeline.initialize()
        
        decision = await pipeline.process_signal(
            signal_id="TEST-001",
            symbol="BTC-PERP",
            direction="long",
            raw_size=10000,
            strategy_name="momentum",
            market_data={'price': 50000, 'volatility': 0.02, 'daily_volume': 100000000},
            expected_edge_bps=25
        )
        
        assert decision.signal_id == "TEST-001"
        assert decision.decision in [SignalDecision.EXECUTE, SignalDecision.SKIP, SignalDecision.REDUCE_SIZE]
    
    @pytest.mark.asyncio
    async def test_kill_switch(self):
        """Test kill switch blocks all trades."""
        from src.trading.production_pipeline import ProductionTradingPipeline, TradingMode, SignalDecision
        
        pipeline = ProductionTradingPipeline(mode=TradingMode.PAPER)
        await pipeline.initialize()
        
        # Activate kill switch
        pipeline.activate_kill_switch("Test")
        
        decision = await pipeline.process_signal(
            signal_id="TEST-002",
            symbol="ETH-PERP",
            direction="short",
            raw_size=5000,
            strategy_name="momentum",
            market_data={'price': 3000}
        )
        
        assert decision.decision == SignalDecision.SKIP, "Kill switch should block all trades"


class TestDeploymentChecklist:
    """Test Deployment Checklist functionality."""
    
    def test_checklist_creation(self):
        """Test checklist is created with all items."""
        from src.ops.deployment_checklist import DeploymentChecklist, PRODUCTION_CHECKLIST
        
        checklist = DeploymentChecklist(state_file="./test_checklist.json")
        
        assert len(checklist.checklist) == len(PRODUCTION_CHECKLIST)
    
    def test_item_update(self):
        """Test updating checklist items."""
        from src.ops.deployment_checklist import DeploymentChecklist, CheckStatus
        
        checklist = DeploymentChecklist(state_file="./test_checklist2.json")
        
        checklist.update_item("RISK-001", CheckStatus.PASSED, "Kill switch verified")
        
        item = checklist.get_item("RISK-001")
        assert item.status == CheckStatus.PASSED
    
    def test_readiness_calculation(self):
        """Test readiness assessment."""
        from src.ops.deployment_checklist import DeploymentChecklist, CheckStatus
        
        checklist = DeploymentChecklist(state_file="./test_checklist3.json")
        
        # Mark all critical items as passed
        for item in checklist.checklist:
            if item.priority.value == "critical":
                checklist.update_item(item.id, CheckStatus.PASSED, "Test")
                if item.requires_signoff:
                    checklist.sign_off(item.id, "tester")
        
        readiness = checklist.get_readiness()
        
        assert readiness.critical_items_passed == readiness.critical_items_total


class TestDurabilityScoring:
    """Test Profitability Durability Scoring."""
    
    def test_durability_calculation(self):
        """Test durability score calculation."""
        from src.analytics.profitability_durability import ProfitabilityDurabilityEngine
        
        engine = ProfitabilityDurabilityEngine()
        
        # Create synthetic data
        np.random.seed(42)
        n_days = 180
        
        returns = pd.Series(
            np.random.normal(0.001, 0.015, n_days),
            index=pd.date_range(end=datetime.now(), periods=n_days, freq='D')
        )
        
        signal_strength = pd.Series(
            np.random.uniform(0.3, 0.7, n_days),
            index=returns.index
        )
        
        market_returns = pd.Series(
            np.random.normal(0.0003, 0.02, n_days),
            index=returns.index
        )
        
        score = engine.calculate_durability_score(
            returns=returns,
            signal_strength=signal_strength,
            market_returns=market_returns,
            backtest_sharpe=2.0,
            live_sharpe=1.6,
            strategy_params={'num_parameters': 15, 'num_signals': 5},
            transaction_costs_bps=15,
            current_aum=100000,
            adv_traded=10000000
        )
        
        assert 0 <= score.overall_score <= 100
        assert score.rating is not None
        assert len(score.key_risks) >= 0
    
    def test_alpha_decay_detection(self):
        """Test alpha decay detection."""
        from src.analytics.profitability_durability import ProfitabilityDurabilityEngine
        
        engine = ProfitabilityDurabilityEngine()
        
        # Create decaying signal
        np.random.seed(42)
        n_days = 180
        decay_factor = np.exp(-np.arange(n_days) * 0.003)  # Clear decay
        
        signal_strength = pd.Series(
            np.random.uniform(0.5, 0.8, n_days) * decay_factor,
            index=pd.date_range(end=datetime.now(), periods=n_days, freq='D')
        )
        
        returns = pd.Series(
            np.random.normal(0.001, 0.015, n_days),
            index=signal_strength.index
        )
        
        decay = engine._analyze_alpha_decay(signal_strength, returns)
        
        assert decay.decay_rate_annual > 0, "Should detect decay"


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
