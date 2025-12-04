/**
 * src/dashboard/react/hooks/useApi.ts
 * 
 * React hooks for connecting to the backend API
 * - REST API calls
 * - WebSocket connection with auto-reconnect
 * - Real-time data subscriptions
 */

import { useState, useEffect, useCallback, useRef } from 'react';

// API Configuration
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8080';
const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8080/ws';

// ============================================================================
// Types
// ============================================================================

export interface SystemStatus {
  status: 'healthy' | 'degraded' | 'error';
  uptime_seconds: number;
  mode: 'paper' | 'canary' | 'production';
  active_strategies: string[];
  meta_learner_enabled: boolean;
  last_trade_time: string | null;
  error_count: number;
}

export interface Position {
  symbol: string;
  side: 'long' | 'short';
  size: number;
  entry_price: number;
  current_price: number;
  unrealized_pnl: number;
  realized_pnl: number;
  leverage: number;
}

export interface Trade {
  id: string;
  timestamp: string;
  symbol: string;
  side: string;
  size: number;
  price: number;
  pnl: number;
  fees: number;
  strategy: string;
}

export interface StrategyStats {
  name: string;
  enabled: boolean;
  trades: number;
  win_rate: number;
  sharpe_ratio: number;
  total_pnl: number;
  last_signal: string | null;
  regime: string;
}

export interface MetaLearnerStatus {
  enabled: boolean;
  mode: string;
  current_strategy: string | null;
  exploration_rate: number;
  regime: string;
  arm_stats: Record<string, { alpha: number; beta: number; pulls: number }>;
}

export interface Alert {
  id: string;
  severity: 'info' | 'warning' | 'critical';
  message: string;
  timestamp: string;
  acknowledged: boolean;
}

export interface PnLHistory {
  date: string;
  daily_pnl: number;
  cumulative_pnl: number;
}

export interface Metrics {
  total_pnl: number;
  total_trades: number;
  avg_win_rate: number;
  active_strategies: number;
  current_drawdown: number;
  sharpe_ratio: number;
}

// ============================================================================
// API Client
// ============================================================================

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || `API Error: ${response.status}`);
    }

    return response.json();
  }

  // System
  async getStatus(): Promise<SystemStatus> {
    return this.request('/api/status');
  }

  async getMode(): Promise<{ mode: string }> {
    return this.request('/api/mode');
  }

  async setMode(mode: string): Promise<{ mode: string }> {
    return this.request(`/api/mode/${mode}`, { method: 'POST' });
  }

  // Positions & Trades
  async getPositions(): Promise<{ positions: Position[] }> {
    return this.request('/api/positions');
  }

  async getTrades(limit = 50): Promise<{ trades: Trade[] }> {
    return this.request(`/api/trades?limit=${limit}`);
  }

  // Strategies
  async getStrategies(): Promise<{ strategies: StrategyStats[] }> {
    return this.request('/api/strategies');
  }

  async toggleStrategy(name: string): Promise<{ strategy: string; enabled: boolean }> {
    return this.request(`/api/strategies/${name}/toggle`, { method: 'POST' });
  }

  // Meta-learner
  async getMetaStatus(): Promise<MetaLearnerStatus> {
    return this.request('/api/meta/status');
  }

  async enableMeta(): Promise<{ status: string; mode: string }> {
    return this.request('/api/meta/enable', { method: 'POST' });
  }

  async disableMeta(): Promise<{ status: string }> {
    return this.request('/api/meta/disable', { method: 'POST' });
  }

  async configureMeta(config: {
    enabled: boolean;
    mode?: string;
    exploration_rate?: number;
  }): Promise<MetaLearnerStatus> {
    return this.request('/api/meta/configure', {
      method: 'POST',
      body: JSON.stringify(config),
    });
  }

  async selectStrategy(): Promise<{
    strategy: string;
    confidence: number;
    regime: string;
  }> {
    return this.request('/api/meta/select');
  }

  // Alerts
  async getAlerts(acknowledged?: boolean): Promise<{ alerts: Alert[] }> {
    const params = acknowledged !== undefined ? `?acknowledged=${acknowledged}` : '';
    return this.request(`/api/alerts${params}`);
  }

  async acknowledgeAlert(alertId: string): Promise<{ status: string }> {
    return this.request(`/api/alerts/${alertId}/acknowledge`, { method: 'POST' });
  }

  // Metrics
  async getMetrics(): Promise<Metrics> {
    return this.request('/api/metrics');
  }

  async getPnLHistory(days = 30): Promise<{ history: PnLHistory[] }> {
    return this.request(`/api/pnl/history?days=${days}`);
  }
}

export const api = new ApiClient();

// ============================================================================
// WebSocket Hook
// ============================================================================

type WebSocketMessage = {
  channel: string;
  data: unknown;
  ts: number;
};

type MessageHandler = (data: unknown) => void;

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const handlersRef = useRef<Map<string, Set<MessageHandler>>>(new Map());
  const [connected, setConnected] = useState(false);
  const [reconnectAttempt, setReconnectAttempt] = useState(0);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket connected');
      setConnected(true);
      setReconnectAttempt(0);
    };

    ws.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);
        const handlers = handlersRef.current.get(message.channel);
        
        if (handlers) {
          handlers.forEach((handler) => handler(message.data));
        }
        
        // Also call "all" handlers
        const allHandlers = handlersRef.current.get('all');
        if (allHandlers) {
          allHandlers.forEach((handler) => handler(message));
        }
      } catch (e) {
        console.error('WebSocket message parse error:', e);
      }
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setConnected(false);
      
      // Auto-reconnect with exponential backoff
      const delay = Math.min(1000 * Math.pow(2, reconnectAttempt), 30000);
      reconnectTimeoutRef.current = setTimeout(() => {
        setReconnectAttempt((a) => a + 1);
        connect();
      }, delay);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }, [reconnectAttempt]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    wsRef.current?.close();
  }, []);

  const subscribe = useCallback((channel: string, handler: MessageHandler) => {
    if (!handlersRef.current.has(channel)) {
      handlersRef.current.set(channel, new Set());
    }
    handlersRef.current.get(channel)!.add(handler);

    // Send subscribe message to server
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({ action: 'subscribe', channels: [channel] })
      );
    }

    // Return unsubscribe function
    return () => {
      handlersRef.current.get(channel)?.delete(handler);
    };
  }, []);

  const send = useCallback((data: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  useEffect(() => {
    connect();
    return () => disconnect();
  }, [connect, disconnect]);

  return { connected, subscribe, send, reconnectAttempt };
}

// ============================================================================
// Data Hooks
// ============================================================================

export function useSystemStatus() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const { subscribe } = useWebSocket();

  useEffect(() => {
    // Initial fetch
    api.getStatus()
      .then(setStatus)
      .catch(setError)
      .finally(() => setLoading(false));

    // Subscribe to updates
    return subscribe('status', (data) => {
      setStatus(data as SystemStatus);
    });
  }, [subscribe]);

  return { status, loading, error };
}

export function usePositions() {
  const [positions, setPositions] = useState<Position[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const { subscribe } = useWebSocket();

  useEffect(() => {
    api.getPositions()
      .then((data) => setPositions(data.positions))
      .catch(setError)
      .finally(() => setLoading(false));

    return subscribe('positions', (data) => {
      const payload = data as { positions: Position[] };
      setPositions(payload.positions);
    });
  }, [subscribe]);

  return { positions, loading, error };
}

export function useStrategies() {
  const [strategies, setStrategies] = useState<StrategyStats[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const { subscribe } = useWebSocket();

  const refresh = useCallback(async () => {
    const data = await api.getStrategies();
    setStrategies(data.strategies);
  }, []);

  const toggle = useCallback(async (name: string) => {
    await api.toggleStrategy(name);
    await refresh();
  }, [refresh]);

  useEffect(() => {
    refresh().catch(setError).finally(() => setLoading(false));

    return subscribe('strategies', (data) => {
      const payload = data as { strategies: StrategyStats[] };
      setStrategies(payload.strategies);
    });
  }, [subscribe, refresh]);

  return { strategies, loading, error, toggle, refresh };
}

export function useMetaLearner() {
  const [meta, setMeta] = useState<MetaLearnerStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const { subscribe } = useWebSocket();

  const refresh = useCallback(async () => {
    const data = await api.getMetaStatus();
    setMeta(data);
  }, []);

  const enable = useCallback(async () => {
    await api.enableMeta();
    await refresh();
  }, [refresh]);

  const disable = useCallback(async () => {
    await api.disableMeta();
    await refresh();
  }, [refresh]);

  const configure = useCallback(
    async (config: { enabled: boolean; mode?: string; exploration_rate?: number }) => {
      await api.configureMeta(config);
      await refresh();
    },
    [refresh]
  );

  useEffect(() => {
    refresh().catch(setError).finally(() => setLoading(false));

    return subscribe('meta', (data) => {
      setMeta(data as MetaLearnerStatus);
    });
  }, [subscribe, refresh]);

  return { meta, loading, error, enable, disable, configure, refresh };
}

export function useAlerts() {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const { subscribe } = useWebSocket();

  const refresh = useCallback(async () => {
    const data = await api.getAlerts();
    setAlerts(data.alerts);
  }, []);

  const acknowledge = useCallback(async (alertId: string) => {
    await api.acknowledgeAlert(alertId);
    await refresh();
  }, [refresh]);

  useEffect(() => {
    refresh().catch(setError).finally(() => setLoading(false));

    return subscribe('alerts', (data) => {
      const payload = data as { alerts: Alert[] };
      setAlerts(payload.alerts);
    });
  }, [subscribe, refresh]);

  return { alerts, loading, error, acknowledge, refresh };
}

export function useMetrics() {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    api.getMetrics()
      .then(setMetrics)
      .catch(setError)
      .finally(() => setLoading(false));

    // Refresh every 30 seconds
    const interval = setInterval(() => {
      api.getMetrics().then(setMetrics).catch(setError);
    }, 30000);

    return () => clearInterval(interval);
  }, []);

  return { metrics, loading, error };
}

export function usePnLHistory(days = 30) {
  const [history, setHistory] = useState<PnLHistory[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    api.getPnLHistory(days)
      .then((data) => setHistory(data.history))
      .catch(setError)
      .finally(() => setLoading(false));
  }, [days]);

  return { history, loading, error };
}

export function useTrades(limit = 50) {
  const [trades, setTrades] = useState<Trade[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const { subscribe } = useWebSocket();

  const refresh = useCallback(async () => {
    const data = await api.getTrades(limit);
    setTrades(data.trades);
  }, [limit]);

  useEffect(() => {
    refresh().catch(setError).finally(() => setLoading(false));

    return subscribe('trades', (data) => {
      const payload = data as { trades: Trade[] };
      setTrades(payload.trades);
    });
  }, [subscribe, refresh]);

  return { trades, loading, error, refresh };
}
