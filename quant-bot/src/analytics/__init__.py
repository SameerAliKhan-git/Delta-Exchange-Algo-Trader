"""
Analytics Module
================
Profitability analysis and durability scoring.
"""

from .profitability_durability import (
    ProfitabilityDurabilityEngine,
    DurabilityScore,
    DurabilityRating,
    AlphaDecayAnalysis,
    RegimeStability,
    CapacityAnalysis
)

__all__ = [
    'ProfitabilityDurabilityEngine',
    'DurabilityScore',
    'DurabilityRating',
    'AlphaDecayAnalysis',
    'RegimeStability',
    'CapacityAnalysis'
]
