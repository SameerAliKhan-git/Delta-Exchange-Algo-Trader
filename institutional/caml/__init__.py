"""
CAML - Capital Allocation Meta-Learner
======================================

Thompson Sampling-based capital allocation across strategies.
"""

from .allocator import CapitalAllocationMetaLearner, AllocationState, CAMLConfig
from .state_manager import StateManager, StrategyPerformance

__all__ = [
    'CapitalAllocationMetaLearner',
    'AllocationState',
    'CAMLConfig',
    'StateManager',
    'StrategyPerformance',
]
