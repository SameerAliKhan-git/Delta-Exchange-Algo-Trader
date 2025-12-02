"""
Aladdin Trading Bot - Unit Tests

Run: pytest tests/ -v
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch


class TestBacktestModule:
    """Tests for backtest module"""
    
    def test_event_engine_import(self):
        """Test event engine imports correctly"""
        from backtest.event_engine import EventEngine, Event, EventType
        assert EventEngine is not None
        assert Event is not None
        assert EventType is not None
    
    def test_event_engine_publish_subscribe(self):
        """Test event pub/sub pattern"""
        from backtest.event_engine import EventEngine, Event, EventType
        
        engine = EventEngine()
        received = []
        
        def handler(event):
            received.append(event)
        
        engine.subscribe(EventType.BAR, handler)
        engine.publish(Event(event_type=EventType.BAR, data={'test': True}))
        engine.process_all()
        
        assert len(received) == 1
        assert received[0].data['test'] == True
    
    def test_data_handler_synthetic(self):
        """Test synthetic data generation"""
        from backtest.data_handler import DataHandler
        
        handler = DataHandler()
        data = handler.generate_synthetic(
            symbol="TEST",
            timeframe="1h",
            start_price=100,
            num_bars=100
        )
        
        assert len(data) == 100
        assert data.symbol == "TEST"
        assert data.timeframe == "1h"


class TestSignalFusion:
    """Tests for signal fusion engine"""
    
    def test_signal_fusion_import(self):
        """Test signal fusion imports"""
        from signals.fusion_engine import SignalFusion, SignalComponent
        assert SignalFusion is not None
    
    def test_signal_aggregation(self):
        """Test signal aggregation"""
        from signals.fusion_engine import SignalFusion, SignalComponent
        
        fusion = SignalFusion()
        
        # Add bullish signals
        fusion.add_component(SignalComponent(
            component_type="technical",
            name="test_ta",
            signal=1,
            strength=0.8,
            confidence=0.9
        ))
        
        fused = fusion.fuse()
        assert fused is not None
        assert fused.direction in [-1, 0, 1]


class TestOrderManager:
    """Tests for order management"""
    
    def test_order_manager_import(self):
        """Test order manager imports"""
        from orders.order_manager import OrderManager, Order, OrderType
        assert OrderManager is not None
    
    def test_order_creation(self):
        """Test order creation"""
        from orders.order_manager import OrderManager
        
        manager = OrderManager()
        
        order = manager.create_order(
            symbol="BTCUSD",
            side="buy",
            quantity=0.1,
            price=100000
        )
        
        assert order is not None
        assert order.symbol == "BTCUSD"
        assert order.side == "buy"


class TestRiskDesk:
    """Tests for risk management"""
    
    def test_risk_desk_import(self):
        """Test risk desk imports"""
        from aladdin.risk_desk import RiskDesk, CircuitBreaker
        assert RiskDesk is not None
        assert CircuitBreaker is not None
    
    def test_circuit_breaker(self):
        """Test circuit breaker functionality"""
        from aladdin.risk_desk import CircuitBreaker
        
        breaker = CircuitBreaker(
            name="test_breaker",
            threshold=10.0,
            cooldown_seconds=60
        )
        
        # Should not be tripped initially
        assert not breaker.is_tripped
        
        # Trip it
        breaker.check(15.0)
        assert breaker.is_tripped


class TestProductCatalog:
    """Tests for product catalog"""
    
    def test_catalog_import(self):
        """Test catalog imports"""
        from catalog.product_catalog import ProductCatalog, Product
        assert ProductCatalog is not None
    
    def test_liquidity_filter_import(self):
        """Test liquidity filter imports"""
        from catalog.liquidity_filter import LiquidityFilter, LiquidityMetrics
        assert LiquidityFilter is not None


class TestOptionsModule:
    """Tests for options module"""
    
    def test_options_scanner_import(self):
        """Test options scanner imports"""
        from options.options_scanner import OptionsScanner
        assert OptionsScanner is not None
    
    def test_iv_analyzer_import(self):
        """Test IV analyzer imports"""
        from options.iv_analyzer import IVAnalyzer, IVPoint, IVSurface
        assert IVAnalyzer is not None
    
    def test_spread_strategies_import(self):
        """Test spread strategies imports"""
        from options.spread_strategies import OptionSpreadBuilder
        assert OptionSpreadBuilder is not None


class TestMonitoring:
    """Tests for monitoring module"""
    
    def test_metrics_exporter_import(self):
        """Test metrics exporter imports"""
        from monitoring.exporter import MetricsExporter, TradeMetrics
        assert MetricsExporter is not None
    
    def test_alerting_import(self):
        """Test alerting imports"""
        from monitoring.alerting import AlertManager, Alert, AlertLevel
        assert AlertManager is not None
        assert AlertLevel.CRITICAL.value > AlertLevel.INFO.value
    
    def test_alert_rate_limiting(self):
        """Test alert rate limiting"""
        from monitoring.alerting import AlertManager, AlertLevel
        
        manager = AlertManager(rate_limit_seconds=60)
        
        # First alert should go through
        result1 = manager.info("Test", "First message")
        
        # Second identical alert should be rate limited
        result2 = manager.info("Test", "Second message")
        
        assert result2 == False  # Rate limited


class TestIngestPipeline:
    """Tests for data ingestion"""
    
    def test_sentiment_engine_import(self):
        """Test sentiment engine imports"""
        from ingest.sentiment_engine import SentimentEngine
        assert SentimentEngine is not None
    
    def test_news_aggregator_import(self):
        """Test news aggregator imports"""
        from ingest.news_aggregator import NewsAggregator, NewsItem
        assert NewsAggregator is not None


class TestBacktestMetrics:
    """Tests for performance metrics"""
    
    def test_metrics_calculation(self):
        """Test metrics calculation"""
        from backtest.metrics import calculate_metrics, Trade, PerformanceMetrics
        
        # Create sample trades
        trades = [
            Trade(
                entry_time=1700000000,
                exit_time=1700003600,
                entry_price=100,
                exit_price=105,
                size=1.0,
                side="long",
                pnl=5.0,
                pnl_pct=5.0
            ),
            Trade(
                entry_time=1700007200,
                exit_time=1700010800,
                entry_price=105,
                exit_price=103,
                size=1.0,
                side="long",
                pnl=-2.0,
                pnl_pct=-1.9
            )
        ]
        
        equity = np.array([100, 105, 103])
        
        metrics = calculate_metrics(trades, equity, 100)
        
        assert metrics.total_trades == 2
        assert metrics.winning_trades == 1
        assert metrics.losing_trades == 1
        assert metrics.win_rate == 50.0
    
    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation"""
        from backtest.metrics import calculate_sharpe
        
        # Positive returns
        returns = np.array([0.01, 0.02, -0.005, 0.015, 0.01])
        sharpe = calculate_sharpe(returns)
        
        assert sharpe > 0  # Positive excess returns
    
    def test_max_drawdown(self):
        """Test max drawdown calculation"""
        from backtest.metrics import calculate_max_drawdown
        
        # Equity curve with 20% drawdown
        equity = np.array([100, 110, 120, 96, 100, 95])
        max_dd = calculate_max_drawdown(equity)
        
        assert max_dd > 0
        assert max_dd <= 1.0


class TestConfiguration:
    """Tests for configuration"""
    
    def test_config_exists(self):
        """Test config file exists"""
        import os
        
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yml')
        assert os.path.exists(config_path), "config.yml should exist"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
