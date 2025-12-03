"""Utils module for quant-bot."""

from .metrics import (
    PerformanceReport,
    calculate_returns,
    calculate_log_returns,
    calculate_total_return,
    calculate_cagr,
    calculate_annualized_volatility,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_var,
    calculate_cvar,
    calculate_calmar_ratio,
    calculate_omega_ratio,
    calculate_profit_factor,
    calculate_win_rate,
    calculate_expectancy,
    generate_performance_report,
    print_performance_report
)

__all__ = [
    'PerformanceReport',
    'calculate_returns',
    'calculate_log_returns',
    'calculate_total_return',
    'calculate_cagr',
    'calculate_annualized_volatility',
    'calculate_sharpe_ratio',
    'calculate_sortino_ratio',
    'calculate_max_drawdown',
    'calculate_var',
    'calculate_cvar',
    'calculate_calmar_ratio',
    'calculate_omega_ratio',
    'calculate_profit_factor',
    'calculate_win_rate',
    'calculate_expectancy',
    'generate_performance_report',
    'print_performance_report'
]
