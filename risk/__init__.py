"""
Risk Module - Risk management and position sizing

Provides:
- Capital-aware position sizing
- Daily loss limits
- Drawdown protection
- Kill switch
"""

from .manager import (
    RiskManager,
    RiskConfig,
    RiskState,
    PositionSize,
    compute_futures_size,
    compute_option_contracts
)

__all__ = [
    "RiskManager",
    "RiskConfig",
    "RiskState",
    "PositionSize",
    "compute_futures_size",
    "compute_option_contracts",
]
