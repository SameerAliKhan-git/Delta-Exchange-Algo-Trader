import React, { useState, useEffect } from 'react';
import { Activity, BarChart3, Terminal, Shield, Zap, Settings, Play, Square } from 'lucide-react';

function App() {
  const [status, setStatus] = useState('STOPPED');
  const [logs, setLogs] = useState([]);
  
  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col font-sans">
      {/* Top Bar */}
      <header className="h-16 border-b border-border flex items-center px-6 justify-between bg-card/50 backdrop-blur-sm fixed w-full top-0 z-10">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded bg-primary/20 flex items-center justify-center text-primary">
            <Zap size={20} />
          </div>
          <h1 className="text-xl font-bold tracking-tight">ALADDIN <span className="text-primary">AI</span></h1>
        </div>
        
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-green-500/10 text-green-500 border border-green-500/20">
            <Activity size={14} />
            <span className="text-sm font-medium">+5.4% P&L</span>
          </div>
          <div className="h-8 w-px bg-border mx-2"></div>
          <button className="p-2 hover:bg-secondary rounded-md transition-colors">
            <Settings size={20} />
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 pt-16 flex">
        {/* Sidebar */}
        <aside className="w-64 border-r border-border bg-card/30 hidden md:flex flex-col">
          <div className="p-4 space-y-2">
            <div className="text-xs font-semibold text-muted-foreground mb-2 uppercase tracking-wider">Strategies</div>
            <StrategyToggle name="Medallion Fund" active={true} />
            <StrategyToggle name="ML Engine" active={true} />
            <StrategyToggle name="Options Alpha" active={false} />
            <StrategyToggle name="Stat Arb" active={true} />
          </div>
          
          <div className="mt-auto p-4 border-t border-border">
            <div className="text-xs font-semibold text-muted-foreground mb-2 uppercase tracking-wider">System Status</div>
            <div className="flex items-center gap-2 text-sm text-green-500 mb-1">
              <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
              API Connected
            </div>
            <div className="flex items-center gap-2 text-sm text-green-500">
              <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
              Risk Checks Pass
            </div>
          </div>
        </aside>

        {/* Dashboard Grid */}
        <div className="flex-1 p-6 grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Chart Area */}
          <div className="lg:col-span-2 bg-card/50 border border-border rounded-xl p-6 min-h-[400px] flex flex-col">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-lg font-semibold flex items-center gap-2">
                <BarChart3 size={18} className="text-primary" />
                Market Overview
              </h2>
              <div className="flex gap-2">
                 <button className="px-3 py-1 text-xs rounded bg-secondary hover:bg-secondary/80">1H</button>
                 <button className="px-3 py-1 text-xs rounded bg-primary text-primary-foreground">4H</button>
                 <button className="px-3 py-1 text-xs rounded bg-secondary hover:bg-secondary/80">1D</button>
              </div>
            </div>
            <div className="flex-1 flex items-center justify-center text-muted-foreground border-2 border-dashed border-border/50 rounded-lg">
              Chart Placeholder (Recharts)
            </div>
          </div>

          {/* Control Panel */}
          <div className="space-y-6">
            {/* Action Card */}
            <div className="bg-card/50 border border-border rounded-xl p-6">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Shield size={18} className="text-primary" />
                Control Center
              </h2>
              
              <div className="grid grid-cols-2 gap-4 mb-6">
                <div className="p-3 bg-secondary/50 rounded-lg">
                  <div className="text-xs text-muted-foreground">Daily P&L</div>
                  <div className="text-xl font-bold text-green-500">+$1,240.50</div>
                </div>
                <div className="p-3 bg-secondary/50 rounded-lg">
                  <div className="text-xs text-muted-foreground">Open Risk</div>
                  <div className="text-xl font-bold text-orange-500">1.2%</div>
                </div>
              </div>

              <div className="flex gap-3">
                <button 
                  onClick={() => setStatus('RUNNING')}
                  className={`flex-1 py-3 rounded-lg font-medium flex items-center justify-center gap-2 transition-all ${status === 'RUNNING' ? 'bg-green-500/20 text-green-500 border border-green-500/50' : 'bg-green-600 hover:bg-green-700 text-white'}`}
                >
                  <Play size={18} />
                  {status === 'RUNNING' ? 'Running' : 'Start Bot'}
                </button>
                <button 
                  onClick={() => setStatus('STOPPED')}
                  className={`flex-1 py-3 rounded-lg font-medium flex items-center justify-center gap-2 transition-all ${status === 'STOPPED' ? 'bg-red-500/20 text-red-500 border border-red-500/50' : 'bg-red-600 hover:bg-red-700 text-white'}`}
                >
                  <Square size={18} />
                  Stop
                </button>
              </div>
            </div>

            {/* Terminal */}
            <div className="bg-black/80 border border-border rounded-xl p-4 h-[300px] flex flex-col font-mono text-xs">
              <div className="flex items-center gap-2 text-muted-foreground mb-2 border-b border-border/50 pb-2">
                <Terminal size={14} />
                <span>System Logs</span>
              </div>
              <div className="flex-1 overflow-y-auto space-y-1 text-gray-300">
                <div className="text-green-500">➜ [SYSTEM] Dashboard initialized...</div>
                <div className="text-blue-400">➜ [CONNECT] Connecting to Delta Exchange...</div>
                <div>➜ [INFO] Market data stream active</div>
                <div>➜ [STRATEGY] Medallion: Scanning for patterns...</div>
                <div className="text-yellow-500">➜ [WARN] High volatility detected in BTC-PERP</div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}

function StrategyToggle({ name, active }) {
  return (
    <div className="flex items-center justify-between p-3 rounded-lg bg-secondary/30 hover:bg-secondary/50 transition-colors cursor-pointer group">
      <div className="flex items-center gap-3">
        <div className={`w-2 h-2 rounded-full ${active ? 'bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.5)]' : 'bg-gray-600'}`}></div>
        <span className="text-sm font-medium group-hover:text-primary transition-colors">{name}</span>
      </div>
      <div className={`w-8 h-4 rounded-full relative transition-colors ${active ? 'bg-primary/20' : 'bg-gray-700'}`}>
        <div className={`absolute top-0.5 w-3 h-3 rounded-full bg-current transition-all ${active ? 'left-4.5 text-primary' : 'left-0.5 text-gray-400'}`}></div>
      </div>
    </div>
  )
}

export default App
