"""
AMRC - Autonomous Meta-Risk Controller
=======================================

Sub-50ms global trading halt capability for institutional safety.
"""

from .controller import AutonomousMetaRiskController, AMRCStatus, AMRCConfig
from .shared_memory import SharedMemoryFlag
from .chaos_tests import AMRCChaosTest

__all__ = [
    'AutonomousMetaRiskController',
    'AMRCStatus',
    'AMRCConfig',
    'SharedMemoryFlag',
    'AMRCChaosTest',
]
