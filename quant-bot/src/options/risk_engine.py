"""
Options Risk Engine - Crypto-Specific Risk Management
======================================================
Extended risk controls for options trading including gamma risk,
vega exposure, volatility crush, and gap risk management.

Author: Quant Bot
Version: 1.0.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum
from datetime import datetime, timedelta
import logging

from .pricing_engine import Greeks, OptionType, OptionsPricingEngine
from .strategies import StrategyPosition

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskType(Enum):
    GAMMA = "gamma"
    VEGA = "vega"
    THETA = "theta"
    DELTA = "delta"
    VOLATILITY_CRUSH = "vol_crush"
    GAP = "gap"
    LIQUIDITY = "liquidity"
    EXPIRATION = "expiration"


@dataclass
class RiskLimit:
    """Risk limit definition."""
    risk_type: RiskType
    limit_value: float
    current_value: float
    utilization: float
    level: RiskLevel
    breach: bool
    
    def __post_init__(self):
        self.utilization = abs(self.current_value / self.limit_value) if self.limit_value != 0 else 0
        self.breach = self.utilization > 1.0
        
        if self.utilization > 0.9:
            self.level = RiskLevel.CRITICAL
        elif self.utilization > 0.7:
            self.level = RiskLevel.HIGH
        elif self.utilization > 0.5:
            self.level = RiskLevel.MEDIUM
        else:
            self.level = RiskLevel.LOW


@dataclass
class VolatilityCrushRisk:
    """Risk from IV collapse after events."""
    event_type: str
    event_time: datetime
    current_iv: float
    expected_post_iv: float
    iv_drop_pct: float
    vega_exposure: float
    potential_loss: float
    
    @property
    def risk_level(self) -> RiskLevel:
        if self.potential_loss > 0.05:  # >5% of position
            return RiskLevel.CRITICAL
        elif self.potential_loss > 0.02:
            return RiskLevel.HIGH
        elif self.potential_loss > 0.01:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW


@dataclass
class GapRiskAnalysis:
    """Gap risk analysis for options."""
    symbol: str
    current_price: float
    expected_gap_pct: float
    gap_scenarios: Dict[float, float]  # gap_size -> pnl
    worst_case_loss: float
    var_95: float
    var_99: float
    
    @property
    def at_risk(self) -> bool:
        return self.worst_case_loss < -0.10  # >10% loss potential


@dataclass
class RiskReport:
    """Comprehensive risk report."""
    timestamp: datetime
    portfolio_value: float
    total_delta: float
    total_gamma: float
    total_vega: float
    total_theta: float
    
    # Dollar exposures
    delta_exposure_usd: float
    gamma_exposure_1pct: float  # P&L for 1% spot move
    vega_exposure_1vol: float   # P&L for 1vol point move
    theta_daily: float
    
    # Risk limits
    limits: List[RiskLimit]
    
    # Special risks
    vol_crush_risk: Optional[VolatilityCrushRisk]
    gap_risk: Optional[GapRiskAnalysis]
    
    # Summary
    overall_risk_level: RiskLevel
    alerts: List[str]


class OptionsRiskEngine:
    """
    Production-grade options risk management system.
    
    Crypto-specific features:
    - 24/7 market gap risk (weekends, news events)
    - High volatility regime handling
    - Volatility crush around events (halving, ETF, etc.)
    - Gamma squeeze detection
    - Cross-margin risk aggregation
    """
    
    # Default risk limits (as fraction of portfolio)
    DEFAULT_LIMITS = {
        RiskType.DELTA: 0.50,      # Max 50% delta exposure
        RiskType.GAMMA: 0.10,      # Max 10% gamma exposure per 1% move
        RiskType.VEGA: 0.15,       # Max 15% vega exposure per 1vol
        RiskType.THETA: 0.02,      # Max 2% daily theta decay
    }
    
    def __init__(
        self,
        pricing_engine: Optional[OptionsPricingEngine] = None,
        portfolio_value: float = 100000,
        risk_limits: Optional[Dict[RiskType, float]] = None
    ):
        self.pricing_engine = pricing_engine or OptionsPricingEngine()
        self.portfolio_value = portfolio_value
        self.risk_limits = risk_limits or self.DEFAULT_LIMITS.copy()
        
        # Position tracking
        self.positions: List[StrategyPosition] = []
        self.aggregate_greeks = Greeks()
        
        # Event calendar
        self.upcoming_events: List[Dict] = []
        
        logger.info("OptionsRiskEngine initialized")
    
    def set_portfolio_value(self, value: float) -> None:
        """Update portfolio value."""
        self.portfolio_value = value
    
    def add_position(self, position: StrategyPosition) -> None:
        """Add position to risk tracking."""
        self.positions.append(position)
        self._update_aggregate_greeks()
    
    def remove_position(self, symbol: str) -> None:
        """Remove position from tracking."""
        self.positions = [p for p in self.positions if p.symbol != symbol]
        self._update_aggregate_greeks()
    
    def _update_aggregate_greeks(self) -> None:
        """Recalculate aggregate portfolio Greeks."""
        total = Greeks()
        for pos in self.positions:
            total = total + pos.net_greeks
        self.aggregate_greeks = total
    
    # ==================== GAMMA RISK ====================
    
    def calculate_gamma_risk(
        self,
        spot_price: float
    ) -> Dict:
        """
        Calculate gamma risk metrics.
        
        Gamma risk = rapid P&L changes from spot moves
        High gamma = large swings, need frequent rebalancing
        """
        gamma = self.aggregate_greeks.gamma
        
        # Dollar gamma (P&L change for 1% spot move)
        dollar_gamma = 0.5 * gamma * (spot_price * 0.01) ** 2
        
        # Gamma per contract
        gamma_1pct = dollar_gamma / self.portfolio_value if self.portfolio_value > 0 else 0
        
        # Scenarios
        scenarios = {}
        for move_pct in [-5, -3, -2, -1, 1, 2, 3, 5]:
            move = spot_price * (move_pct / 100)
            gamma_pnl = 0.5 * gamma * move ** 2
            delta_pnl = self.aggregate_greeks.delta * move
            scenarios[move_pct] = {
                'spot_change': move,
                'delta_pnl': delta_pnl,
                'gamma_pnl': gamma_pnl,
                'total_pnl': delta_pnl + gamma_pnl
            }
        
        # Risk assessment
        is_short_gamma = gamma < 0
        
        return {
            'raw_gamma': gamma,
            'dollar_gamma': dollar_gamma,
            'gamma_pct_portfolio': gamma_1pct,
            'is_short_gamma': is_short_gamma,
            'gamma_risk_level': self._assess_gamma_level(gamma_1pct),
            'scenarios': scenarios,
            'hedging_frequency': self._recommend_hedge_frequency(gamma, spot_price)
        }
    
    def _assess_gamma_level(self, gamma_pct: float) -> RiskLevel:
        """Assess gamma risk level."""
        if abs(gamma_pct) > 0.10:
            return RiskLevel.CRITICAL
        elif abs(gamma_pct) > 0.05:
            return RiskLevel.HIGH
        elif abs(gamma_pct) > 0.02:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW
    
    def _recommend_hedge_frequency(
        self,
        gamma: float,
        spot_price: float
    ) -> str:
        """Recommend hedging frequency based on gamma."""
        # Calculate expected delta change per 0.5% move
        delta_change_per_half_pct = abs(gamma * spot_price * 0.005)
        
        if delta_change_per_half_pct > 0.10:
            return "Continuous (every price tick)"
        elif delta_change_per_half_pct > 0.05:
            return "Every 0.5% spot move"
        elif delta_change_per_half_pct > 0.02:
            return "Every 1% spot move"
        elif delta_change_per_half_pct > 0.01:
            return "Every 2% spot move"
        return "Daily rebalancing sufficient"
    
    # ==================== VEGA RISK ====================
    
    def calculate_vega_risk(
        self,
        current_iv: float
    ) -> Dict:
        """
        Calculate vega (volatility) risk metrics.
        
        Vega risk = P&L sensitivity to implied volatility changes
        """
        vega = self.aggregate_greeks.vega
        
        # P&L for 1 vol point (1%) move
        vega_1vol = vega * 1  # Vega already normalized to 1%
        vega_pct = vega_1vol / self.portfolio_value if self.portfolio_value > 0 else 0
        
        # Scenarios
        vol_scenarios = {}
        for vol_change in [-20, -10, -5, 5, 10, 20]:
            pnl = vega * vol_change
            vol_scenarios[vol_change] = {
                'new_iv': current_iv * (1 + vol_change / 100),
                'pnl': pnl,
                'pnl_pct': pnl / self.portfolio_value if self.portfolio_value > 0 else 0
            }
        
        # Long/short vega
        is_long_vega = vega > 0
        
        return {
            'raw_vega': vega,
            'vega_per_1vol': vega_1vol,
            'vega_pct_portfolio': vega_pct,
            'is_long_vega': is_long_vega,
            'vega_risk_level': self._assess_vega_level(vega_pct),
            'vol_scenarios': vol_scenarios,
            'recommendation': self._vega_recommendation(is_long_vega, current_iv)
        }
    
    def _assess_vega_level(self, vega_pct: float) -> RiskLevel:
        """Assess vega risk level."""
        if abs(vega_pct) > 0.15:
            return RiskLevel.CRITICAL
        elif abs(vega_pct) > 0.10:
            return RiskLevel.HIGH
        elif abs(vega_pct) > 0.05:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW
    
    def _vega_recommendation(
        self,
        is_long_vega: bool,
        current_iv: float
    ) -> str:
        """Recommend vega management action."""
        if is_long_vega:
            if current_iv > 0.80:
                return "CAUTION: Long vega at high IV - consider reducing"
            else:
                return "Long vega: Benefits from volatility increase"
        else:
            if current_iv < 0.30:
                return "CAUTION: Short vega at low IV - limited upside"
            else:
                return "Short vega: Benefits from volatility decrease"
    
    # ==================== VOLATILITY CRUSH RISK ====================
    
    def add_upcoming_event(
        self,
        event_type: str,
        event_time: datetime,
        expected_iv_drop_pct: float
    ) -> None:
        """Add an upcoming event that may cause IV crush."""
        self.upcoming_events.append({
            'type': event_type,
            'time': event_time,
            'iv_drop': expected_iv_drop_pct
        })
    
    def calculate_vol_crush_risk(
        self,
        current_iv: float
    ) -> Optional[VolatilityCrushRisk]:
        """
        Calculate risk from volatility crush after events.
        
        Events like earnings, halvings, ETF decisions cause IV run-up
        followed by sharp drop (IV crush).
        """
        if not self.upcoming_events:
            return None
        
        # Find nearest event
        now = datetime.now()
        upcoming = [e for e in self.upcoming_events if e['time'] > now]
        
        if not upcoming:
            return None
        
        nearest = min(upcoming, key=lambda e: e['time'])
        
        # Calculate potential loss from IV drop
        expected_post_iv = current_iv * (1 - nearest['iv_drop'] / 100)
        iv_drop_pct = nearest['iv_drop']
        
        # Vega P&L
        vega = self.aggregate_greeks.vega
        potential_loss = -vega * iv_drop_pct  # Negative if long vega
        potential_loss_pct = potential_loss / self.portfolio_value if self.portfolio_value > 0 else 0
        
        return VolatilityCrushRisk(
            event_type=nearest['type'],
            event_time=nearest['time'],
            current_iv=current_iv,
            expected_post_iv=expected_post_iv,
            iv_drop_pct=iv_drop_pct,
            vega_exposure=vega,
            potential_loss=potential_loss_pct
        )
    
    # ==================== GAP RISK ====================
    
    def calculate_gap_risk(
        self,
        spot_price: float,
        historical_gaps: Optional[List[float]] = None
    ) -> GapRiskAnalysis:
        """
        Calculate gap risk for crypto markets.
        
        Unlike traditional markets, crypto trades 24/7 but gaps can
        occur during low liquidity periods or major news events.
        """
        # Default gap distribution if not provided
        if historical_gaps is None:
            # Typical crypto gap sizes (as fractions)
            historical_gaps = [
                0.05, 0.03, 0.02, 0.01, -0.01, -0.02, -0.03, -0.05,
                0.10, -0.10, 0.15, -0.15  # Extreme scenarios
            ]
        
        # Calculate P&L for each gap scenario
        gap_scenarios = {}
        
        for gap_pct in sorted(set(historical_gaps)):
            gap_size = spot_price * gap_pct
            new_price = spot_price + gap_size
            
            # Delta P&L
            delta_pnl = self.aggregate_greeks.delta * gap_size
            
            # Gamma P&L (significant for large gaps)
            gamma_pnl = 0.5 * self.aggregate_greeks.gamma * gap_size ** 2
            
            total_pnl = delta_pnl + gamma_pnl
            gap_scenarios[gap_pct] = total_pnl
        
        # Risk metrics
        pnls = list(gap_scenarios.values())
        worst_case = min(pnls) / self.portfolio_value if self.portfolio_value > 0 else min(pnls)
        
        # VaR
        sorted_pnls = sorted(pnls)
        var_95 = sorted_pnls[int(len(sorted_pnls) * 0.05)] / self.portfolio_value
        var_99 = sorted_pnls[int(len(sorted_pnls) * 0.01)] / self.portfolio_value
        
        # Expected gap based on historical
        expected_gap = np.std(historical_gaps)
        
        return GapRiskAnalysis(
            symbol="PORTFOLIO",
            current_price=spot_price,
            expected_gap_pct=expected_gap,
            gap_scenarios=gap_scenarios,
            worst_case_loss=worst_case,
            var_95=var_95,
            var_99=var_99
        )
    
    # ==================== EXPIRATION RISK ====================
    
    def calculate_expiration_risk(
        self,
        spot_price: float,
        days_to_expiry: float
    ) -> Dict:
        """
        Calculate risks specific to approaching expiration.
        
        Near expiration:
        - Gamma becomes extreme for ATM options
        - Theta decay accelerates
        - Pin risk (price pins to strike)
        """
        # Identify positions near expiry
        near_expiry = [p for p in self.positions 
                      if any(leg.expiry <= days_to_expiry / 365 for leg in p.option_legs)]
        
        if not near_expiry:
            return {'at_risk': False}
        
        # Aggregate Greeks for near-expiry positions
        near_greeks = Greeks()
        for pos in near_expiry:
            near_greeks = near_greeks + pos.net_greeks
        
        # Check for extreme gamma (ATM options)
        gamma_risk = abs(near_greeks.gamma * spot_price * 0.01)  # For 1% move
        
        # Theta decay
        daily_decay = abs(near_greeks.theta)
        
        # Pin risk (estimate probability of expiring near a strike)
        strikes = set()
        for pos in near_expiry:
            for leg in pos.option_legs:
                strikes.add(leg.strike)
        
        nearest_strike = min(strikes, key=lambda k: abs(k - spot_price))
        distance_to_pin = abs(spot_price - nearest_strike) / spot_price
        
        return {
            'at_risk': True,
            'positions_count': len(near_expiry),
            'near_expiry_gamma': near_greeks.gamma,
            'gamma_risk_1pct': gamma_risk,
            'daily_theta_decay': daily_decay,
            'nearest_strike': nearest_strike,
            'distance_to_pin_pct': distance_to_pin * 100,
            'pin_risk': distance_to_pin < 0.02,  # Within 2% of strike
            'recommendation': self._expiry_recommendation(days_to_expiry, gamma_risk)
        }
    
    def _expiry_recommendation(
        self,
        days: float,
        gamma_risk: float
    ) -> str:
        """Recommend action for expiration risk."""
        if days <= 1:
            if gamma_risk > 0.05:
                return "URGENT: Close or roll positions before expiry"
            return "Monitor closely - same day expiry"
        elif days <= 3:
            if gamma_risk > 0.10:
                return "Consider rolling to later expiry"
            return "Monitor for pin risk"
        return "Normal expiration risk"
    
    # ==================== COMPREHENSIVE RISK REPORT ====================
    
    def generate_risk_report(
        self,
        spot_price: float,
        current_iv: float
    ) -> RiskReport:
        """
        Generate comprehensive risk report.
        """
        # Dollar exposures
        delta_exposure = self.aggregate_greeks.delta * spot_price
        gamma_exposure = 0.5 * self.aggregate_greeks.gamma * (spot_price * 0.01) ** 2 * 100
        vega_exposure = self.aggregate_greeks.vega
        theta_daily = self.aggregate_greeks.theta
        
        # Check limits
        limits = []
        
        # Delta limit
        delta_pct = abs(delta_exposure) / self.portfolio_value if self.portfolio_value > 0 else 0
        limits.append(RiskLimit(
            RiskType.DELTA,
            self.risk_limits[RiskType.DELTA],
            delta_pct,
            0, RiskLevel.LOW, False
        ))
        
        # Gamma limit
        gamma_pct = abs(gamma_exposure) / self.portfolio_value if self.portfolio_value > 0 else 0
        limits.append(RiskLimit(
            RiskType.GAMMA,
            self.risk_limits[RiskType.GAMMA],
            gamma_pct,
            0, RiskLevel.LOW, False
        ))
        
        # Vega limit
        vega_pct = abs(vega_exposure) / self.portfolio_value if self.portfolio_value > 0 else 0
        limits.append(RiskLimit(
            RiskType.VEGA,
            self.risk_limits[RiskType.VEGA],
            vega_pct,
            0, RiskLevel.LOW, False
        ))
        
        # Theta limit
        theta_pct = abs(theta_daily) / self.portfolio_value if self.portfolio_value > 0 else 0
        limits.append(RiskLimit(
            RiskType.THETA,
            self.risk_limits[RiskType.THETA],
            theta_pct,
            0, RiskLevel.LOW, False
        ))
        
        # Special risks
        vol_crush = self.calculate_vol_crush_risk(current_iv)
        gap_risk = self.calculate_gap_risk(spot_price)
        
        # Generate alerts
        alerts = []
        for limit in limits:
            if limit.breach:
                alerts.append(f"⚠️ {limit.risk_type.value.upper()} LIMIT BREACH: {limit.utilization:.1%}")
            elif limit.level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                alerts.append(f"🔶 {limit.risk_type.value} at {limit.utilization:.1%} of limit")
        
        if vol_crush and vol_crush.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            alerts.append(f"📉 Vol crush risk: {vol_crush.event_type} in {(vol_crush.event_time - datetime.now()).days}d")
        
        if gap_risk.at_risk:
            alerts.append(f"⚡ Gap risk elevated: worst case {gap_risk.worst_case_loss:.1%}")
        
        # Overall risk level
        critical_count = sum(1 for l in limits if l.level == RiskLevel.CRITICAL)
        high_count = sum(1 for l in limits if l.level == RiskLevel.HIGH)
        
        if critical_count > 0 or any(l.breach for l in limits):
            overall = RiskLevel.CRITICAL
        elif high_count >= 2:
            overall = RiskLevel.HIGH
        elif high_count >= 1:
            overall = RiskLevel.MEDIUM
        else:
            overall = RiskLevel.LOW
        
        return RiskReport(
            timestamp=datetime.now(),
            portfolio_value=self.portfolio_value,
            total_delta=self.aggregate_greeks.delta,
            total_gamma=self.aggregate_greeks.gamma,
            total_vega=self.aggregate_greeks.vega,
            total_theta=self.aggregate_greeks.theta,
            delta_exposure_usd=delta_exposure,
            gamma_exposure_1pct=gamma_exposure,
            vega_exposure_1vol=vega_exposure,
            theta_daily=theta_daily,
            limits=limits,
            vol_crush_risk=vol_crush,
            gap_risk=gap_risk,
            overall_risk_level=overall,
            alerts=alerts
        )
    
    # ==================== HEDGING RECOMMENDATIONS ====================
    
    def get_hedging_recommendation(
        self,
        spot_price: float,
        current_iv: float
    ) -> Dict:
        """
        Get specific hedging recommendations.
        """
        recommendations = []
        
        # Delta hedge
        delta = self.aggregate_greeks.delta
        if abs(delta) > 0.1:
            hedge_qty = -delta
            recommendations.append({
                'type': 'DELTA_HEDGE',
                'action': 'BUY' if hedge_qty > 0 else 'SELL',
                'quantity': abs(hedge_qty),
                'instrument': 'SPOT/PERP',
                'urgency': 'HIGH' if abs(delta) > 0.3 else 'MEDIUM'
            })
        
        # Gamma reduction
        gamma = self.aggregate_greeks.gamma
        if abs(gamma) > 0.001:
            recommendations.append({
                'type': 'GAMMA_HEDGE',
                'action': 'Buy options' if gamma < 0 else 'Sell options',
                'target': 'Reduce gamma to < 0.0005',
                'instrument': 'ATM OPTIONS',
                'urgency': 'MEDIUM'
            })
        
        # Vega hedge
        vega = self.aggregate_greeks.vega
        vol_crush = self.calculate_vol_crush_risk(current_iv)
        
        if vol_crush and vega > 0 and vol_crush.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            recommendations.append({
                'type': 'VEGA_HEDGE',
                'action': 'SELL VEGA',
                'reason': f'Upcoming {vol_crush.event_type} - expect {vol_crush.iv_drop_pct:.0f}% IV drop',
                'instrument': 'SHORT STRADDLE/STRANGLE',
                'urgency': 'HIGH'
            })
        
        return {
            'recommendations': recommendations,
            'current_greeks': {
                'delta': delta,
                'gamma': gamma,
                'vega': vega,
                'theta': self.aggregate_greeks.theta
            },
            'target_greeks': {
                'delta': 0,
                'gamma': 0,
                'vega': 'neutral',
                'theta': 'acceptable'
            }
        }


# ==================== CLI ====================

if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Options Risk Engine")
    parser.add_argument("--portfolio-value", type=float, default=100000)
    parser.add_argument("--spot", type=float, default=100000)
    parser.add_argument("--iv", type=float, default=0.60)
    parser.add_argument("--delta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.001)
    parser.add_argument("--vega", type=float, default=100)
    parser.add_argument("--theta", type=float, default=-50)
    
    args = parser.parse_args()
    
    engine = OptionsRiskEngine(portfolio_value=args.portfolio_value)
    
    # Set aggregate Greeks manually for demo
    engine.aggregate_greeks = Greeks(
        delta=args.delta,
        gamma=args.gamma,
        vega=args.vega,
        theta=args.theta
    )
    
    # Add upcoming event
    engine.add_upcoming_event(
        "Bitcoin Halving",
        datetime.now() + timedelta(days=30),
        expected_iv_drop_pct=25
    )
    
    # Generate report
    report = engine.generate_risk_report(args.spot, args.iv)
    
    print(f"\n{'='*60}")
    print("OPTIONS RISK REPORT")
    print(f"{'='*60}")
    print(f"Timestamp: {report.timestamp}")
    print(f"Portfolio Value: ${report.portfolio_value:,.2f}")
    print(f"\nAggregate Greeks:")
    print(f"  Delta: {report.total_delta:+.4f}")
    print(f"  Gamma: {report.total_gamma:.6f}")
    print(f"  Vega:  {report.total_vega:.2f}")
    print(f"  Theta: {report.theta_daily:.2f}/day")
    print(f"\nDollar Exposures:")
    print(f"  Delta: ${report.delta_exposure_usd:,.2f}")
    print(f"  Gamma (1% move): ${report.gamma_exposure_1pct:,.2f}")
    print(f"  Vega (1vol): ${report.vega_exposure_1vol:,.2f}")
    print(f"  Theta daily: ${report.theta_daily:,.2f}")
    print(f"\nRisk Limits:")
    for limit in report.limits:
        status = "❌ BREACH" if limit.breach else f"✓ {limit.utilization:.0%}"
        print(f"  {limit.risk_type.value}: {status}")
    print(f"\nOverall Risk: {report.overall_risk_level.value.upper()}")
    if report.alerts:
        print(f"\nAlerts:")
        for alert in report.alerts:
            print(f"  {alert}")
    
    # Get hedging recommendations
    recs = engine.get_hedging_recommendation(args.spot, args.iv)
    if recs['recommendations']:
        print(f"\nHedging Recommendations:")
        for rec in recs['recommendations']:
            print(f"  [{rec['urgency']}] {rec['type']}: {rec['action']}")
