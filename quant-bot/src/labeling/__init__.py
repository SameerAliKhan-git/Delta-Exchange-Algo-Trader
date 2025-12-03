"""Labeling module for quant-bot."""

from .afml_labeling import (
    TripleBarrierLabeler,
    TripleBarrierConfig,
    MetaLabeler,
    compute_sample_weights,
    compute_return_attribution,
    trend_scanning_labels,
    to_binary_labels,
    to_multiclass_labels,
    analyze_labels
)

__all__ = [
    'TripleBarrierLabeler',
    'TripleBarrierConfig',
    'MetaLabeler',
    'compute_sample_weights',
    'compute_return_attribution',
    'trend_scanning_labels',
    'to_binary_labels',
    'to_multiclass_labels',
    'analyze_labels'
]
