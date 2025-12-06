"""
Institutional-Grade Trading Components
======================================

This module contains production-grade components for institutional trading:

1. AMRC - Autonomous Meta-Risk Controller
   - Sub-50ms global halt capability
   - Multi-source risk monitoring
   
2. CAML - Capital Allocation Meta-Learner
   - Thompson Sampling with GP prior
   - Capacity-aware allocation
   
3. Execution RL - Quote-Level RL Agent
   - LOB-aware execution
   - Slippage minimization
   
4. Security - Zero-Trust Security Mesh
   - Vault integration
   - 2-person authentication
   
5. Audit - Regulatory-Ready Audit Bus
   - Immutable logging
   - 7-year retention
   
6. Regime - Bayesian Changepoint Detection
   - Drift rejection
   - False alarm filtering
   
7. Capacity - Liquidity Forecasting
   - Kyle-lambda estimation
   - TFT prediction
   
8. Explainability - SHAP/XAI
   - Model explanation
   - Feature importance
"""

from .amrc import AutonomousMetaRiskController, AMRCStatus
from .caml import CapitalAllocationMetaLearner, AllocationState
from .execution_rl import ExecutionRLAgent, LOBEnvironment
from .security import SecurityMesh, VaultClient
from .audit import AuditBus, TradeJustification
from .regime import BayesianChangepoint, DriftRejector
from .capacity import KyleLambdaEstimator, LiquidityForecaster
from .explainability import SHAPExplainer, ModelCard

__all__ = [
    'AutonomousMetaRiskController',
    'AMRCStatus',
    'CapitalAllocationMetaLearner',
    'AllocationState',
    'ExecutionRLAgent',
    'LOBEnvironment',
    'SecurityMesh',
    'VaultClient',
    'AuditBus',
    'TradeJustification',
    'BayesianChangepoint',
    'DriftRejector',
    'KyleLambdaEstimator',
    'LiquidityForecaster',
    'SHAPExplainer',
    'ModelCard',
]
