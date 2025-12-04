/**
 * src/dashboard/react/components/Dashboard.tsx
 * 
 * Main Dashboard Component
 * - System status overview
 * - Position cards
 * - Strategy performance
 * - Meta-learner control
 * - Real-time P&L
 */

import React from 'react';
import {
  useSystemStatus,
  usePositions,
  useStrategies,
  useMetaLearner,
  useMetrics,
  useAlerts,
  usePnLHistory,
} from '../hooks/useApi';

// ============================================================================
// Status Badge
// ============================================================================

interface StatusBadgeProps {
  status: 'healthy' | 'degraded' | 'error' | 'paper' | 'canary' | 'production';
}

function StatusBadge({ status }: StatusBadgeProps) {
  const colors: Record<string, string> = {
    healthy: 'bg-green-500',
    degraded: 'bg-yellow-500',
    error: 'bg-red-500',
    paper: 'bg-blue-500',
    canary: 'bg-yellow-500',
    production: 'bg-green-500',
  };

  return (
    <span
      className={`px-2 py-1 text-xs font-semibold text-white rounded ${colors[status] || 'bg-gray-500'}`}
    >
      {status.toUpperCase()}
    </span>
  );
}

// ============================================================================
// System Status Card
// ============================================================================

function SystemStatusCard() {
  const { status, loading, error } = useSystemStatus();

  if (loading) return <div className="animate-pulse h-32 bg-gray-700 rounded" />;
  if (error) return <div className="text-red-500">Error: {error.message}</div>;
  if (!status) return null;

  const uptimeHours = Math.floor(status.uptime_seconds / 3600);
  const uptimeMinutes = Math.floor((status.uptime_seconds % 3600) / 60);

  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-bold text-white">System Status</h2>
        <div className="flex gap-2">
          <StatusBadge status={status.status} />
          <StatusBadge status={status.mode} />
        </div>
      </div>
      
      <div className="grid grid-cols-2 gap-4 text-gray-300">
        <div>
          <span className="text-gray-500">Uptime:</span>
          <span className="ml-2">{uptimeHours}h {uptimeMinutes}m</span>
        </div>
        <div>
          <span className="text-gray-500">Active Strategies:</span>
          <span className="ml-2">{status.active_strategies.length}</span>
        </div>
        <div>
          <span className="text-gray-500">Meta-Learner:</span>
          <span className={`ml-2 ${status.meta_learner_enabled ? 'text-green-400' : 'text-gray-500'}`}>
            {status.meta_learner_enabled ? 'Enabled' : 'Disabled'}
          </span>
        </div>
        <div>
          <span className="text-gray-500">Errors:</span>
          <span className={`ml-2 ${status.error_count > 0 ? 'text-red-400' : 'text-green-400'}`}>
            {status.error_count}
          </span>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Positions Card
// ============================================================================

function PositionsCard() {
  const { positions, loading, error } = usePositions();

  if (loading) return <div className="animate-pulse h-48 bg-gray-700 rounded" />;
  if (error) return <div className="text-red-500">Error: {error.message}</div>;

  const totalUnrealizedPnL = positions.reduce((sum, p) => sum + p.unrealized_pnl, 0);

  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-bold text-white">Open Positions</h2>
        <span className={`text-lg font-semibold ${totalUnrealizedPnL >= 0 ? 'text-green-400' : 'text-red-400'}`}>
          {totalUnrealizedPnL >= 0 ? '+' : ''}${totalUnrealizedPnL.toFixed(2)}
        </span>
      </div>

      {positions.length === 0 ? (
        <p className="text-gray-500">No open positions</p>
      ) : (
        <div className="space-y-3">
          {positions.map((pos) => (
            <div
              key={pos.symbol}
              className="flex justify-between items-center p-3 bg-gray-700 rounded"
            >
              <div>
                <span className="font-semibold text-white">{pos.symbol}</span>
                <span
                  className={`ml-2 text-sm ${pos.side === 'long' ? 'text-green-400' : 'text-red-400'}`}
                >
                  {pos.side.toUpperCase()} {pos.leverage}x
                </span>
              </div>
              <div className="text-right">
                <div className="text-white">{pos.size.toFixed(4)}</div>
                <div className={pos.unrealized_pnl >= 0 ? 'text-green-400' : 'text-red-400'}>
                  {pos.unrealized_pnl >= 0 ? '+' : ''}${pos.unrealized_pnl.toFixed(2)}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Strategies Card
// ============================================================================

function StrategiesCard() {
  const { strategies, loading, error, toggle } = useStrategies();

  if (loading) return <div className="animate-pulse h-64 bg-gray-700 rounded" />;
  if (error) return <div className="text-red-500">Error: {error.message}</div>;

  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <h2 className="text-xl font-bold text-white mb-4">Strategies</h2>
      
      <div className="space-y-3">
        {strategies.map((strat) => (
          <div
            key={strat.name}
            className={`p-4 rounded ${strat.enabled ? 'bg-gray-700' : 'bg-gray-700/50'}`}
          >
            <div className="flex justify-between items-center mb-2">
              <div className="flex items-center gap-2">
                <span className="font-semibold text-white">{strat.name}</span>
                <button
                  onClick={() => toggle(strat.name)}
                  className={`px-2 py-0.5 text-xs rounded ${
                    strat.enabled
                      ? 'bg-green-500/20 text-green-400'
                      : 'bg-gray-600 text-gray-400'
                  }`}
                >
                  {strat.enabled ? 'ON' : 'OFF'}
                </button>
              </div>
              <span className={strat.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'}>
                ${strat.total_pnl.toFixed(0)}
              </span>
            </div>
            
            <div className="grid grid-cols-3 gap-2 text-sm text-gray-400">
              <div>
                <span className="text-gray-500">Win Rate: </span>
                <span className="text-white">{(strat.win_rate * 100).toFixed(0)}%</span>
              </div>
              <div>
                <span className="text-gray-500">Sharpe: </span>
                <span className="text-white">{strat.sharpe_ratio.toFixed(2)}</span>
              </div>
              <div>
                <span className="text-gray-500">Trades: </span>
                <span className="text-white">{strat.trades}</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ============================================================================
// Meta-Learner Card
// ============================================================================

function MetaLearnerCard() {
  const { meta, loading, error, enable, disable, configure } = useMetaLearner();

  if (loading) return <div className="animate-pulse h-48 bg-gray-700 rounded" />;
  if (error) return <div className="text-red-500">Error: {error.message}</div>;
  if (!meta) return null;

  const handleToggle = () => {
    if (meta.enabled) {
      disable();
    } else {
      enable();
    }
  };

  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-bold text-white">Meta-Learner</h2>
        <button
          onClick={handleToggle}
          className={`px-4 py-2 rounded font-semibold transition ${
            meta.enabled
              ? 'bg-green-500 hover:bg-green-600 text-white'
              : 'bg-gray-600 hover:bg-gray-500 text-gray-300'
          }`}
        >
          {meta.enabled ? 'Enabled' : 'Disabled'}
        </button>
      </div>

      {meta.enabled && (
        <>
          <div className="grid grid-cols-2 gap-4 text-gray-300 mb-4">
            <div>
              <span className="text-gray-500">Mode:</span>
              <span className="ml-2 capitalize">{meta.mode}</span>
            </div>
            <div>
              <span className="text-gray-500">Current:</span>
              <span className="ml-2 text-green-400">{meta.current_strategy || 'N/A'}</span>
            </div>
            <div>
              <span className="text-gray-500">Regime:</span>
              <span className="ml-2 capitalize">{meta.regime}</span>
            </div>
            <div>
              <span className="text-gray-500">Exploration:</span>
              <span className="ml-2">{(meta.exploration_rate * 100).toFixed(0)}%</span>
            </div>
          </div>

          <div className="border-t border-gray-700 pt-4">
            <h3 className="text-sm font-semibold text-gray-400 mb-2">Arm Statistics</h3>
            <div className="space-y-2">
              {Object.entries(meta.arm_stats).map(([name, stats]) => {
                const winRate = stats.alpha / (stats.alpha + stats.beta);
                return (
                  <div key={name} className="flex justify-between items-center text-sm">
                    <span className="text-gray-300">{name}</span>
                    <div className="flex gap-4 text-gray-400">
                      <span>α={stats.alpha}</span>
                      <span>β={stats.beta}</span>
                      <span className="text-white">{(winRate * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

// ============================================================================
// Metrics Card
// ============================================================================

function MetricsCard() {
  const { metrics, loading, error } = useMetrics();

  if (loading) return <div className="animate-pulse h-32 bg-gray-700 rounded" />;
  if (error) return <div className="text-red-500">Error: {error.message}</div>;
  if (!metrics) return null;

  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <h2 className="text-xl font-bold text-white mb-4">Performance Metrics</h2>
      
      <div className="grid grid-cols-2 md:grid-cols-3 gap-6">
        <div>
          <div className="text-gray-500 text-sm">Total P&L</div>
          <div className={`text-2xl font-bold ${metrics.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            ${metrics.total_pnl.toFixed(0)}
          </div>
        </div>
        <div>
          <div className="text-gray-500 text-sm">Win Rate</div>
          <div className="text-2xl font-bold text-white">
            {(metrics.avg_win_rate * 100).toFixed(0)}%
          </div>
        </div>
        <div>
          <div className="text-gray-500 text-sm">Sharpe Ratio</div>
          <div className="text-2xl font-bold text-white">
            {metrics.sharpe_ratio.toFixed(2)}
          </div>
        </div>
        <div>
          <div className="text-gray-500 text-sm">Total Trades</div>
          <div className="text-2xl font-bold text-white">
            {metrics.total_trades}
          </div>
        </div>
        <div>
          <div className="text-gray-500 text-sm">Active Strategies</div>
          <div className="text-2xl font-bold text-white">
            {metrics.active_strategies}
          </div>
        </div>
        <div>
          <div className="text-gray-500 text-sm">Current DD</div>
          <div className="text-2xl font-bold text-yellow-400">
            {(metrics.current_drawdown * 100).toFixed(1)}%
          </div>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Alerts Card
// ============================================================================

function AlertsCard() {
  const { alerts, loading, error, acknowledge } = useAlerts();

  if (loading) return <div className="animate-pulse h-32 bg-gray-700 rounded" />;
  if (error) return <div className="text-red-500">Error: {error.message}</div>;

  const unacknowledged = alerts.filter((a) => !a.acknowledged);

  const severityColors: Record<string, string> = {
    info: 'border-blue-500',
    warning: 'border-yellow-500',
    critical: 'border-red-500',
  };

  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-bold text-white">Alerts</h2>
        {unacknowledged.length > 0 && (
          <span className="bg-red-500 text-white text-xs px-2 py-1 rounded-full">
            {unacknowledged.length}
          </span>
        )}
      </div>

      {unacknowledged.length === 0 ? (
        <p className="text-gray-500">No active alerts</p>
      ) : (
        <div className="space-y-2 max-h-48 overflow-y-auto">
          {unacknowledged.map((alert) => (
            <div
              key={alert.id}
              className={`p-3 bg-gray-700 rounded border-l-4 ${severityColors[alert.severity]}`}
            >
              <div className="flex justify-between items-start">
                <p className="text-gray-300 text-sm">{alert.message}</p>
                <button
                  onClick={() => acknowledge(alert.id)}
                  className="text-gray-500 hover:text-white text-xs"
                >
                  ✕
                </button>
              </div>
              <p className="text-gray-500 text-xs mt-1">
                {new Date(alert.timestamp).toLocaleTimeString()}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Main Dashboard
// ============================================================================

export function Dashboard() {
  return (
    <div className="min-h-screen bg-gray-900 p-6">
      <div className="max-w-7xl mx-auto">
        <header className="mb-8">
          <h1 className="text-3xl font-bold text-white">Quant Bot Dashboard</h1>
          <p className="text-gray-400">Real-time trading system monitor</p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Row 1 */}
          <SystemStatusCard />
          <MetricsCard />

          {/* Row 2 */}
          <PositionsCard />
          <AlertsCard />

          {/* Row 3 */}
          <StrategiesCard />
          <MetaLearnerCard />
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
