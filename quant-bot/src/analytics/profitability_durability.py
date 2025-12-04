"""
Profitability Durability Scoring Engine
========================================
Quantifies how DURABLE your strategy's edge is.

This module provides:
- Multi-factor durability assessment
- Stress testing profitability across regimes
- Decay rate estimation for alpha signals
- Competitive moat analysis
- Overall "bankability" score

Reference: Quantitative equity alpha typically decays 10-30% per year due to crowding.
Crypto may decay faster (50%+) or offer new opportunities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
from scipy import stats
from collections import defaultdict

logger = logging.getLogger(__name__)


class DurabilityRating(Enum):
    """Overall durability rating."""
    EXCELLENT = "excellent"     # >80 - Deploy with confidence
    GOOD = "good"              # 60-80 - Deploy with monitoring
    MARGINAL = "marginal"      # 40-60 - Deploy small, iterate
    POOR = "poor"              # 20-40 - Do not deploy, rebuild
    UNVIABLE = "unviable"      # <20 - Abandon strategy


@dataclass
class AlphaDecayAnalysis:
    """Analysis of alpha signal decay over time."""
    half_life_days: float           # Time for signal to lose half its power
    decay_rate_annual: float        # Annual decay percentage
    is_accelerating: bool           # Is decay speeding up?
    estimated_viable_months: int    # How long before signal is useless
    confidence_interval: Tuple[float, float]  # 95% CI for decay rate


@dataclass
class RegimeStability:
    """Strategy stability across market regimes."""
    trending_sharpe: float
    mean_reverting_sharpe: float
    volatile_sharpe: float
    crisis_sharpe: float
    
    # Consistency metrics
    regime_consistency_score: float  # 0-100, higher = more consistent across regimes
    worst_regime: str
    best_regime: str


@dataclass
class CapacityAnalysis:
    """Strategy capacity analysis."""
    estimated_capacity_usd: float      # Max capital before impact kills edge
    current_utilization_pct: float     # How much capacity are we using
    marginal_impact_per_million: float # Impact cost per additional $1M
    is_constrained: bool               # Are we near capacity limits


@dataclass
class DurabilityScore:
    """Complete durability assessment."""
    
    # Overall score
    overall_score: float  # 0-100
    rating: DurabilityRating
    
    # Component scores (0-100 each)
    alpha_decay_score: float
    regime_stability_score: float
    transaction_robustness_score: float
    capacity_score: float
    complexity_penalty_score: float
    live_vs_backtest_score: float
    
    # Detailed analysis
    alpha_decay: AlphaDecayAnalysis
    regime_stability: RegimeStability
    capacity: CapacityAnalysis
    
    # Risk metrics
    probability_of_negative_sharpe_1yr: float
    expected_drawdown_during_decay: float
    
    # Recommendations
    deployment_recommendation: str
    key_risks: List[str]
    improvements_needed: List[str]
    
    # Metadata
    calculated_at: datetime = field(default_factory=datetime.now)


class ProfitabilityDurabilityEngine:
    """
    Assess how durable a strategy's profitability is.
    
    This goes beyond Sharpe ratio to ask: "Will this still work in 6 months?"
    
    Key Questions Answered:
    1. How fast is the alpha decaying?
    2. How consistent is performance across regimes?
    3. What's the strategy capacity?
    4. How robust to transaction costs?
    5. How much does backtest overfit reality?
    """
    
    def __init__(
        self,
        min_history_days: int = 90,
        regime_window: int = 20,
        alpha_decay_lookback_days: int = 180,
        complexity_factors: List[str] = None
    ):
        self.min_history_days = min_history_days
        self.regime_window = regime_window
        self.alpha_decay_lookback_days = alpha_decay_lookback_days
        self.complexity_factors = complexity_factors or [
            'num_parameters',
            'num_signals',
            'lookback_period',
            'num_filters'
        ]
    
    def calculate_durability_score(
        self,
        returns: pd.Series,
        signal_strength: pd.Series,
        market_returns: pd.Series,
        backtest_sharpe: float,
        live_sharpe: float,
        strategy_params: Dict[str, Any],
        transaction_costs_bps: float,
        current_aum: float,
        adv_traded: float  # Average daily volume traded
    ) -> DurabilityScore:
        """
        Calculate comprehensive durability score.
        
        Args:
            returns: Strategy daily returns
            signal_strength: Daily signal strength/confidence
            market_returns: Market benchmark returns
            backtest_sharpe: Sharpe from backtesting
            live_sharpe: Sharpe from live/paper trading
            strategy_params: Dict of strategy parameters
            transaction_costs_bps: Current transaction cost assumption
            current_aum: Current assets under management
            adv_traded: Average daily volume traded
        
        Returns:
            Complete DurabilityScore assessment
        """
        
        # 1. Alpha Decay Analysis
        alpha_decay = self._analyze_alpha_decay(signal_strength, returns)
        alpha_score = self._score_alpha_decay(alpha_decay)
        
        # 2. Regime Stability Analysis
        regime_stability = self._analyze_regime_stability(returns, market_returns)
        regime_score = regime_stability.regime_consistency_score
        
        # 3. Transaction Cost Robustness
        tx_robustness = self._analyze_transaction_robustness(
            returns, transaction_costs_bps
        )
        
        # 4. Capacity Analysis
        capacity = self._analyze_capacity(
            returns, current_aum, adv_traded
        )
        capacity_score = self._score_capacity(capacity)
        
        # 5. Complexity Penalty
        complexity_penalty = self._calculate_complexity_penalty(strategy_params)
        complexity_score = 100 - complexity_penalty
        
        # 6. Live vs Backtest Gap
        live_vs_backtest = self._score_live_vs_backtest(backtest_sharpe, live_sharpe)
        
        # Calculate Overall Score
        weights = {
            'alpha_decay': 0.25,
            'regime_stability': 0.20,
            'tx_robustness': 0.15,
            'capacity': 0.10,
            'complexity': 0.10,
            'live_vs_backtest': 0.20
        }
        
        overall_score = (
            alpha_score * weights['alpha_decay'] +
            regime_score * weights['regime_stability'] +
            tx_robustness * weights['tx_robustness'] +
            capacity_score * weights['capacity'] +
            complexity_score * weights['complexity'] +
            live_vs_backtest * weights['live_vs_backtest']
        )
        
        # Determine rating
        rating = self._get_rating(overall_score)
        
        # Calculate risk metrics
        prob_neg_sharpe = self._probability_negative_sharpe(returns, months=12)
        expected_dd = self._expected_drawdown_during_decay(returns, alpha_decay)
        
        # Generate recommendations
        recommendation, risks, improvements = self._generate_recommendations(
            overall_score=overall_score,
            alpha_decay=alpha_decay,
            regime_stability=regime_stability,
            tx_robustness=tx_robustness,
            capacity=capacity,
            complexity_penalty=complexity_penalty,
            live_vs_backtest=live_vs_backtest
        )
        
        return DurabilityScore(
            overall_score=overall_score,
            rating=rating,
            alpha_decay_score=alpha_score,
            regime_stability_score=regime_score,
            transaction_robustness_score=tx_robustness,
            capacity_score=capacity_score,
            complexity_penalty_score=complexity_score,
            live_vs_backtest_score=live_vs_backtest,
            alpha_decay=alpha_decay,
            regime_stability=regime_stability,
            capacity=capacity,
            probability_of_negative_sharpe_1yr=prob_neg_sharpe,
            expected_drawdown_during_decay=expected_dd,
            deployment_recommendation=recommendation,
            key_risks=risks,
            improvements_needed=improvements
        )
    
    def _analyze_alpha_decay(
        self,
        signal_strength: pd.Series,
        returns: pd.Series
    ) -> AlphaDecayAnalysis:
        """Analyze how fast the alpha signal is decaying."""
        
        # Calculate rolling information coefficient (IC)
        window = min(20, len(signal_strength) // 5)
        if window < 5:
            window = 5
        
        rolling_ic = signal_strength.rolling(window).corr(returns.shift(-1))
        rolling_ic = rolling_ic.dropna()
        
        if len(rolling_ic) < 30:
            return AlphaDecayAnalysis(
                half_life_days=365,  # Assume stable if not enough data
                decay_rate_annual=0.0,
                is_accelerating=False,
                estimated_viable_months=24,
                confidence_interval=(0.0, 0.2)
            )
        
        # Fit exponential decay
        x = np.arange(len(rolling_ic))
        y = rolling_ic.abs().values
        
        # Linear regression on log scale
        y_log = np.log(y + 1e-10)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y_log)
        
        # Calculate decay metrics
        if slope >= 0:
            half_life = 365  # Not decaying
            decay_rate = 0
        else:
            half_life = -np.log(2) / slope
            decay_rate = 1 - np.exp(slope * 252)  # Annualized
        
        # Check if decay is accelerating
        mid_point = len(rolling_ic) // 2
        early_slope, _, _, _, _ = stats.linregress(x[:mid_point], y_log[:mid_point])
        late_slope, _, _, _, _ = stats.linregress(x[mid_point:], y_log[mid_point:])
        is_accelerating = late_slope < early_slope - 0.001
        
        # Estimate viable months
        if decay_rate > 0:
            # When will IC drop below 0.02?
            current_ic = abs(rolling_ic.iloc[-1])
            if current_ic > 0.02:
                months_to_threshold = np.log(0.02 / current_ic) / slope / 21  # 21 trading days per month
                viable_months = max(0, int(months_to_threshold))
            else:
                viable_months = 0
        else:
            viable_months = 24
        
        # Confidence interval
        ci_low = max(0, decay_rate - 2 * std_err * 252)
        ci_high = decay_rate + 2 * std_err * 252
        
        return AlphaDecayAnalysis(
            half_life_days=half_life,
            decay_rate_annual=decay_rate,
            is_accelerating=is_accelerating,
            estimated_viable_months=viable_months,
            confidence_interval=(ci_low, ci_high)
        )
    
    def _score_alpha_decay(self, decay: AlphaDecayAnalysis) -> float:
        """Score alpha decay (0-100, higher is better/slower decay)."""
        
        # Score based on half-life
        if decay.half_life_days >= 180:
            half_life_score = 100
        elif decay.half_life_days >= 90:
            half_life_score = 80
        elif decay.half_life_days >= 60:
            half_life_score = 60
        elif decay.half_life_days >= 30:
            half_life_score = 40
        else:
            half_life_score = 20
        
        # Penalty for accelerating decay
        if decay.is_accelerating:
            half_life_score *= 0.8
        
        return half_life_score
    
    def _analyze_regime_stability(
        self,
        returns: pd.Series,
        market_returns: pd.Series
    ) -> RegimeStability:
        """Analyze strategy performance across market regimes."""
        
        # Detect regimes
        regimes = self._detect_regimes(market_returns)
        
        # Calculate Sharpe by regime
        regime_sharpes = {}
        for regime in ['trending', 'mean_reverting', 'volatile', 'crisis']:
            mask = regimes == regime
            if mask.sum() > 20:  # Need enough data
                regime_returns = returns[mask]
                sharpe = regime_returns.mean() / regime_returns.std() * np.sqrt(252)
                regime_sharpes[regime] = sharpe
            else:
                regime_sharpes[regime] = np.nan
        
        # Fill NaN with overall Sharpe
        overall_sharpe = returns.mean() / returns.std() * np.sqrt(252)
        for regime in regime_sharpes:
            if np.isnan(regime_sharpes[regime]):
                regime_sharpes[regime] = overall_sharpe
        
        # Calculate consistency score
        sharpe_values = [v for v in regime_sharpes.values() if not np.isnan(v)]
        if len(sharpe_values) > 1:
            # Consistency = 100 - coefficient of variation
            cv = np.std(sharpe_values) / (np.mean(sharpe_values) + 0.01) * 100
            consistency_score = max(0, 100 - cv)
        else:
            consistency_score = 50
        
        # Identify best/worst regimes
        worst_regime = min(regime_sharpes, key=regime_sharpes.get)
        best_regime = max(regime_sharpes, key=regime_sharpes.get)
        
        return RegimeStability(
            trending_sharpe=regime_sharpes['trending'],
            mean_reverting_sharpe=regime_sharpes['mean_reverting'],
            volatile_sharpe=regime_sharpes['volatile'],
            crisis_sharpe=regime_sharpes['crisis'],
            regime_consistency_score=consistency_score,
            worst_regime=worst_regime,
            best_regime=best_regime
        )
    
    def _detect_regimes(self, market_returns: pd.Series) -> pd.Series:
        """Detect market regimes from returns."""
        
        regimes = pd.Series(index=market_returns.index, dtype=str)
        
        # Rolling statistics
        window = self.regime_window
        rolling_vol = market_returns.rolling(window).std() * np.sqrt(252)
        rolling_trend = market_returns.rolling(window).mean() * 252
        rolling_autocorr = market_returns.rolling(window).apply(
            lambda x: x.autocorr() if len(x) > 5 else 0
        )
        
        # Classify regimes
        vol_median = rolling_vol.median()
        
        for i in range(len(market_returns)):
            if i < window:
                regimes.iloc[i] = 'unknown'
                continue
            
            vol = rolling_vol.iloc[i]
            trend = rolling_trend.iloc[i]
            autocorr = rolling_autocorr.iloc[i]
            
            # Crisis: very high vol + negative trend
            if vol > vol_median * 2 and trend < -0.1:
                regimes.iloc[i] = 'crisis'
            # Volatile: high vol
            elif vol > vol_median * 1.5:
                regimes.iloc[i] = 'volatile'
            # Trending: clear direction, positive autocorrelation
            elif abs(trend) > 0.2 and autocorr > 0.1:
                regimes.iloc[i] = 'trending'
            # Mean reverting: negative autocorrelation
            elif autocorr < -0.1:
                regimes.iloc[i] = 'mean_reverting'
            # Default to trending
            else:
                regimes.iloc[i] = 'trending'
        
        return regimes
    
    def _analyze_transaction_robustness(
        self,
        returns: pd.Series,
        current_cost_bps: float
    ) -> float:
        """Analyze robustness to transaction costs."""
        
        # Estimate turnover from returns
        daily_return_mag = returns.abs().mean()
        estimated_turnover = daily_return_mag * 100  # Rough proxy
        
        # Calculate break-even cost
        gross_return = returns.mean() * 252  # Annualized
        if estimated_turnover > 0:
            breakeven_cost = gross_return / (estimated_turnover * 252) * 10000  # Convert to bps
        else:
            breakeven_cost = 1000  # Very high if no turnover estimate
        
        # Score based on cushion above current costs
        cushion = breakeven_cost - current_cost_bps
        
        if cushion > 50:  # More than 50 bps cushion
            return 100
        elif cushion > 30:
            return 80
        elif cushion > 15:
            return 60
        elif cushion > 5:
            return 40
        elif cushion > 0:
            return 20
        else:
            return 0  # Already unprofitable
    
    def _analyze_capacity(
        self,
        returns: pd.Series,
        current_aum: float,
        adv_traded: float
    ) -> CapacityAnalysis:
        """Analyze strategy capacity constraints."""
        
        # Estimate capacity using ADV rule of thumb
        # Assume can trade 1% of ADV without significant impact
        capacity_estimate = adv_traded * 0.01 * 252  # Annualized capacity
        
        if current_aum > 0:
            utilization = current_aum / capacity_estimate * 100
        else:
            utilization = 0
        
        # Marginal impact estimate (simplified Almgren-Chriss)
        base_impact = 10  # bps base impact
        marginal_impact = base_impact * (current_aum / capacity_estimate) ** 0.5
        
        is_constrained = utilization > 50
        
        return CapacityAnalysis(
            estimated_capacity_usd=capacity_estimate,
            current_utilization_pct=utilization,
            marginal_impact_per_million=marginal_impact,
            is_constrained=is_constrained
        )
    
    def _score_capacity(self, capacity: CapacityAnalysis) -> float:
        """Score capacity situation."""
        
        if capacity.current_utilization_pct < 10:
            return 100  # Plenty of room
        elif capacity.current_utilization_pct < 30:
            return 80
        elif capacity.current_utilization_pct < 50:
            return 60
        elif capacity.current_utilization_pct < 70:
            return 40
        elif capacity.current_utilization_pct < 90:
            return 20
        else:
            return 0  # Over capacity
    
    def _calculate_complexity_penalty(self, params: Dict[str, Any]) -> float:
        """Calculate complexity penalty (0-100, higher = more complex = worse)."""
        
        penalties = []
        
        # Number of parameters
        num_params = params.get('num_parameters', 10)
        if num_params > 50:
            penalties.append(30)
        elif num_params > 30:
            penalties.append(20)
        elif num_params > 15:
            penalties.append(10)
        
        # Number of signals
        num_signals = params.get('num_signals', 3)
        if num_signals > 20:
            penalties.append(25)
        elif num_signals > 10:
            penalties.append(15)
        elif num_signals > 5:
            penalties.append(5)
        
        # Lookback period complexity
        lookback = params.get('lookback_period', 50)
        if lookback > 500:
            penalties.append(20)
        elif lookback > 200:
            penalties.append(10)
        
        # Number of filters
        num_filters = params.get('num_filters', 2)
        if num_filters > 10:
            penalties.append(15)
        elif num_filters > 5:
            penalties.append(10)
        
        return min(100, sum(penalties))
    
    def _score_live_vs_backtest(
        self,
        backtest_sharpe: float,
        live_sharpe: float
    ) -> float:
        """Score the live vs backtest gap."""
        
        if backtest_sharpe <= 0:
            return 0
        
        ratio = live_sharpe / backtest_sharpe
        
        if ratio >= 0.9:
            return 100  # Excellent - live nearly matches backtest
        elif ratio >= 0.75:
            return 80  # Good
        elif ratio >= 0.5:
            return 50  # Concerning
        elif ratio >= 0.25:
            return 25  # Poor
        else:
            return 0  # Major overfit
    
    def _probability_negative_sharpe(
        self,
        returns: pd.Series,
        months: int = 12
    ) -> float:
        """Calculate probability of negative Sharpe over next N months."""
        
        # Use historical distribution
        monthly_returns = returns.resample('M').sum()
        mu = monthly_returns.mean()
        sigma = monthly_returns.std()
        
        if sigma == 0:
            return 0.0 if mu > 0 else 1.0
        
        # Probability that mean of N months is negative
        # Using normal approximation
        n_months_sigma = sigma / np.sqrt(months)
        prob_negative = stats.norm.cdf(0, mu, n_months_sigma)
        
        return prob_negative
    
    def _expected_drawdown_during_decay(
        self,
        returns: pd.Series,
        decay: AlphaDecayAnalysis
    ) -> float:
        """Estimate expected drawdown as alpha decays."""
        
        current_sharpe = returns.mean() / returns.std() * np.sqrt(252)
        current_vol = returns.std() * np.sqrt(252)
        
        # Project Sharpe decay over 6 months
        months = 6
        decay_factor = np.exp(-decay.decay_rate_annual * months / 12)
        future_sharpe = current_sharpe * decay_factor
        
        # Estimate max drawdown (simplified formula)
        # Max DD ≈ 2 * vol / sharpe for reasonable holding periods
        if future_sharpe > 0.2:
            expected_dd = current_vol / future_sharpe
        else:
            expected_dd = current_vol * 5  # Large DD if Sharpe collapses
        
        return min(expected_dd, 1.0)  # Cap at 100%
    
    def _get_rating(self, score: float) -> DurabilityRating:
        """Convert score to rating."""
        if score >= 80:
            return DurabilityRating.EXCELLENT
        elif score >= 60:
            return DurabilityRating.GOOD
        elif score >= 40:
            return DurabilityRating.MARGINAL
        elif score >= 20:
            return DurabilityRating.POOR
        else:
            return DurabilityRating.UNVIABLE
    
    def _generate_recommendations(
        self,
        overall_score: float,
        alpha_decay: AlphaDecayAnalysis,
        regime_stability: RegimeStability,
        tx_robustness: float,
        capacity: CapacityAnalysis,
        complexity_penalty: float,
        live_vs_backtest: float
    ) -> Tuple[str, List[str], List[str]]:
        """Generate deployment recommendations."""
        
        risks = []
        improvements = []
        
        # Alpha decay risks
        if alpha_decay.decay_rate_annual > 0.3:
            risks.append(f"High alpha decay ({alpha_decay.decay_rate_annual*100:.0f}% annually)")
        if alpha_decay.is_accelerating:
            risks.append("Alpha decay is accelerating")
            improvements.append("Investigate cause of signal degradation")
        if alpha_decay.estimated_viable_months < 6:
            risks.append(f"Signal may only be viable for {alpha_decay.estimated_viable_months} more months")
            improvements.append("Develop new signals to replace decaying ones")
        
        # Regime stability
        if regime_stability.regime_consistency_score < 50:
            risks.append(f"Inconsistent across regimes (worst: {regime_stability.worst_regime})")
            improvements.append(f"Add regime-specific adjustments for {regime_stability.worst_regime}")
        if regime_stability.crisis_sharpe < 0:
            risks.append(f"Loses money in crisis (Sharpe: {regime_stability.crisis_sharpe:.2f})")
            improvements.append("Add crisis hedging or position reduction rules")
        
        # Transaction costs
        if tx_robustness < 40:
            risks.append("Vulnerable to transaction cost increases")
            improvements.append("Reduce turnover or negotiate better execution")
        
        # Capacity
        if capacity.is_constrained:
            risks.append(f"Near capacity limits ({capacity.current_utilization_pct:.0f}% utilized)")
            improvements.append("Scale strategy or find additional liquidity venues")
        
        # Complexity
        if complexity_penalty > 30:
            risks.append("High complexity increases overfit risk")
            improvements.append("Simplify strategy, remove redundant parameters")
        
        # Live vs backtest gap
        if live_vs_backtest < 50:
            risks.append("Large gap between backtest and live performance")
            improvements.append("Review execution, check for look-ahead bias")
        
        # Generate recommendation
        if overall_score >= 80:
            recommendation = "✅ DEPLOY WITH CONFIDENCE - Monitor alpha decay"
        elif overall_score >= 60:
            recommendation = "🟡 DEPLOY WITH MONITORING - Address risks within 90 days"
        elif overall_score >= 40:
            recommendation = "⚠️ DEPLOY SMALL - Significant improvements needed"
        elif overall_score >= 20:
            recommendation = "❌ DO NOT DEPLOY - Major rebuild required"
        else:
            recommendation = "🚫 ABANDON STRATEGY - Start fresh"
        
        return recommendation, risks, improvements
    
    def generate_report(self, score: DurabilityScore) -> str:
        """Generate human-readable durability report."""
        
        lines = [
            "=" * 70,
            "PROFITABILITY DURABILITY REPORT",
            "=" * 70,
            f"Generated: {score.calculated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "-" * 70,
            "OVERALL ASSESSMENT",
            "-" * 70,
            f"Durability Score: {score.overall_score:.0f}/100",
            f"Rating: {score.rating.value.upper()}",
            f"Recommendation: {score.deployment_recommendation}",
            "",
            "-" * 70,
            "COMPONENT SCORES",
            "-" * 70,
            f"Alpha Decay:           {score.alpha_decay_score:.0f}/100",
            f"Regime Stability:      {score.regime_stability_score:.0f}/100",
            f"Transaction Robustness:{score.transaction_robustness_score:.0f}/100",
            f"Capacity:              {score.capacity_score:.0f}/100",
            f"Complexity Penalty:    {score.complexity_penalty_score:.0f}/100 (100 = low complexity)",
            f"Live vs Backtest:      {score.live_vs_backtest_score:.0f}/100",
            "",
            "-" * 70,
            "ALPHA DECAY ANALYSIS",
            "-" * 70,
            f"Half-Life:             {score.alpha_decay.half_life_days:.0f} days",
            f"Annual Decay Rate:     {score.alpha_decay.decay_rate_annual*100:.1f}%",
            f"Decay Accelerating:    {'YES ⚠️' if score.alpha_decay.is_accelerating else 'No'}",
            f"Estimated Viable:      {score.alpha_decay.estimated_viable_months} months",
            "",
            "-" * 70,
            "REGIME STABILITY",
            "-" * 70,
            f"Trending Markets:      Sharpe {score.regime_stability.trending_sharpe:.2f}",
            f"Mean Reverting:        Sharpe {score.regime_stability.mean_reverting_sharpe:.2f}",
            f"Volatile Markets:      Sharpe {score.regime_stability.volatile_sharpe:.2f}",
            f"Crisis Periods:        Sharpe {score.regime_stability.crisis_sharpe:.2f}",
            f"Consistency Score:     {score.regime_stability.regime_consistency_score:.0f}/100",
            f"Best Regime:           {score.regime_stability.best_regime}",
            f"Worst Regime:          {score.regime_stability.worst_regime}",
            "",
            "-" * 70,
            "CAPACITY ANALYSIS",
            "-" * 70,
            f"Estimated Capacity:    ${score.capacity.estimated_capacity_usd:,.0f}",
            f"Current Utilization:   {score.capacity.current_utilization_pct:.1f}%",
            f"Marginal Impact:       {score.capacity.marginal_impact_per_million:.1f} bps per $1M",
            f"Constrained:           {'YES ⚠️' if score.capacity.is_constrained else 'No'}",
            "",
            "-" * 70,
            "RISK METRICS",
            "-" * 70,
            f"P(Negative Sharpe 1Y): {score.probability_of_negative_sharpe_1yr*100:.1f}%",
            f"Expected DD (decay):   {score.expected_drawdown_during_decay*100:.1f}%",
            "",
        ]
        
        if score.key_risks:
            lines.extend([
                "-" * 70,
                "⚠️ KEY RISKS",
                "-" * 70,
            ])
            for risk in score.key_risks:
                lines.append(f"  • {risk}")
            lines.append("")
        
        if score.improvements_needed:
            lines.extend([
                "-" * 70,
                "📋 IMPROVEMENTS NEEDED",
                "-" * 70,
            ])
            for improvement in score.improvements_needed:
                lines.append(f"  • {improvement}")
            lines.append("")
        
        lines.extend([
            "=" * 70,
            "END OF REPORT",
            "=" * 70,
        ])
        
        return "\n".join(lines)


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate durability scoring engine."""
    
    print("=" * 70)
    print("PROFITABILITY DURABILITY ENGINE - DEMO")
    print("=" * 70)
    
    # Generate synthetic data
    np.random.seed(42)
    n_days = 365
    
    # Create returns with some decay in performance
    base_sharpe = 1.5
    decay_factor = np.exp(-np.arange(n_days) * 0.001)  # Slow decay
    
    daily_returns = np.random.normal(
        base_sharpe * 0.15 / np.sqrt(252) * decay_factor,
        0.15 / np.sqrt(252),
        n_days
    )
    
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    returns = pd.Series(daily_returns, index=dates)
    
    # Signal strength (also decaying)
    signal_strength = pd.Series(
        np.random.uniform(0.3, 0.7, n_days) * decay_factor,
        index=dates
    )
    
    # Market returns
    market_returns = pd.Series(
        np.random.normal(0.0003, 0.015, n_days),
        index=dates
    )
    
    # Strategy parameters
    params = {
        'num_parameters': 25,
        'num_signals': 8,
        'lookback_period': 100,
        'num_filters': 4
    }
    
    # Create engine and score
    engine = ProfitabilityDurabilityEngine()
    
    score = engine.calculate_durability_score(
        returns=returns,
        signal_strength=signal_strength,
        market_returns=market_returns,
        backtest_sharpe=2.0,
        live_sharpe=1.5,
        strategy_params=params,
        transaction_costs_bps=15,
        current_aum=500000,
        adv_traded=50000000
    )
    
    # Print report
    report = engine.generate_report(score)
    print(report)
    
    print(f"\n{'✅' if score.rating in [DurabilityRating.EXCELLENT, DurabilityRating.GOOD] else '❌'}")
    print(f"FINAL ASSESSMENT: {score.rating.value.upper()}")


if __name__ == "__main__":
    demo()
