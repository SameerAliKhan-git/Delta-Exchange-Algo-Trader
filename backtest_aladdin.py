#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    ALADDIN BACKTESTER - Historical Performance                 ║
║                                                                               ║
║  Test your strategy on historical data before risking real money!             ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import requests
import json
import time
from datetime import datetime, timedelta
from collections import deque
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple
import random

# =============================================================================
# 📊 STRATEGY PARAMETERS - LONG BIAS TREND-FOLLOWING
# =============================================================================
STOP_LOSS_PCT = 0.01          # 1% stop (wider for noise)
TARGET_PCT = 0.03             # 3% target (3:1 R:R after fees)
ROUND_TRIP_FEE = 0.001        # 0.1%
MIN_CONFIRMATIONS = 3         # Need 3 signals for confirmation
MIN_CONFIDENCE = 30           # Higher confidence required
BREAKEVEN_TRIGGER = 0.01      # Move to BE at 1%
TRAIL_START = 0.015           # Start trailing at 1.5%
TRAIL_DISTANCE = 0.005        # Trail 0.5% behind
LONG_BIAS = True              # Only take LONG trades (follow the macro uptrend)

# =============================================================================
# 📈 TECHNICAL ANALYSIS (Same functions as main bot)
# =============================================================================

def calculate_rsi(prices_list: List[float], period: int = 14) -> float:
    if len(prices_list) < period + 1:
        return 50
    changes = [prices_list[i] - prices_list[i-1] for i in range(1, len(prices_list))]
    recent = changes[-(period):]
    gains = [c for c in recent if c > 0]
    losses = [-c for c in recent if c < 0]
    avg_gain = sum(gains) / period if gains else 0
    avg_loss = sum(losses) / period if losses else 0.0001
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_momentum(prices_list: List[float], period: int = 10) -> float:
    if len(prices_list) < period:
        return 0
    return ((prices_list[-1] - prices_list[-period]) / prices_list[-period]) * 100

def calculate_vwap_signal(prices_list: List[float]) -> float:
    if len(prices_list) < 20:
        return 0
    vwap = mean(prices_list[-50:]) if len(prices_list) >= 50 else mean(prices_list)
    current = prices_list[-1]
    deviation = ((current - vwap) / vwap) * 100
    return -deviation

def calculate_trend(prices_list: List[float]) -> Tuple[str, float]:
    if len(prices_list) < 20:
        return 'NEUTRAL', 0
    short_ma = mean(prices_list[-5:])
    long_ma = mean(prices_list[-20:])
    diff_pct = ((short_ma - long_ma) / long_ma) * 100
    # More sensitive trend detection (0.05% threshold)
    if diff_pct > 0.05:
        return 'BULLISH', min(abs(diff_pct) * 50, 100)
    elif diff_pct < -0.05:
        return 'BEARISH', min(abs(diff_pct) * 50, 100)
    return 'NEUTRAL', 0

def get_signals(prices_list: List[float], fear_greed: int = 50) -> Dict[str, Tuple[str, float]]:
    """Get all strategy signals - OPTIMIZED FOR MORE TRADES"""
    signals = {}
    
    if len(prices_list) < 30:
        return signals
    
    # RSI - More sensitive thresholds
    rsi = calculate_rsi(prices_list)
    if rsi < 35:  # Was 25
        signals['RSI'] = ('LONG', min((35 - rsi) * 3, 100))
    elif rsi > 65:  # Was 75
        signals['RSI'] = ('SHORT', min((rsi - 65) * 3, 100))
    
    # Momentum - Lower threshold
    momentum = calculate_momentum(prices_list, 10)
    if momentum > 0.05:  # Was 0.15
        signals['MOMENTUM'] = ('LONG', min(momentum * 100, 100))
    elif momentum < -0.05:
        signals['MOMENTUM'] = ('SHORT', min(abs(momentum) * 100, 100))
    
    # VWAP - Lower threshold
    vwap_signal = calculate_vwap_signal(prices_list)
    if vwap_signal > 0.15:  # Was 0.4
        signals['VWAP'] = ('LONG', min(vwap_signal * 50, 100))
    elif vwap_signal < -0.15:
        signals['VWAP'] = ('SHORT', min(abs(vwap_signal) * 50, 100))
    
    # Trend - Lower threshold
    trend, strength = calculate_trend(prices_list)
    if trend == 'BULLISH' and strength > 20:  # Was 40
        signals['TREND'] = ('LONG', strength)
    elif trend == 'BEARISH' and strength > 20:
        signals['TREND'] = ('SHORT', strength)
    
    # Sentiment (always active for backtest)
    if fear_greed < 40:  # Was 20
        signals['SENTIMENT'] = ('LONG', min((40 - fear_greed) * 2, 100))
    elif fear_greed > 60:  # Was 80
        signals['SENTIMENT'] = ('SHORT', min((fear_greed - 60) * 2, 100))
    
    # Orderbook simulation based on short-term momentum
    short_momentum = calculate_momentum(prices_list, 3)
    if short_momentum > 0.02:
        signals['ORDERBOOK'] = ('LONG', min(abs(short_momentum) * 200, 100))
    elif short_momentum < -0.02:
        signals['ORDERBOOK'] = ('SHORT', min(abs(short_momentum) * 200, 100))
    
    return signals

def get_combined_signal(prices_list: List[float]) -> Tuple[Optional[str], float, int]:
    """Get combined signal - LONG BIAS TREND-FOLLOWING"""
    signals = get_signals(prices_list)
    
    if not signals:
        return None, 0, 0
    
    trend, trend_strength = calculate_trend(prices_list)
    
    long_count = sum(1 for s, _ in signals.values() if s == 'LONG')
    short_count = sum(1 for s, _ in signals.values() if s == 'SHORT')
    
    weights = {'RSI': 20, 'MOMENTUM': 15, 'ORDERBOOK': 25, 'VWAP': 15, 'TREND': 20, 'SENTIMENT': 5}
    
    long_score = sum(strength * weights.get(s, 10) / 100 for s, (d, strength) in signals.items() if d == 'LONG')
    short_score = sum(strength * weights.get(s, 10) / 100 for s, (d, strength) in signals.items() if d == 'SHORT')
    
    # LONG BIAS: Only take LONG trades when trend is bullish OR neutral with signals
    # Short trades tend to get stopped out in crypto markets
    if LONG_BIAS:
        if trend == 'BULLISH' and long_count >= 1:
            return 'LONG', long_score + trend_strength, long_count
        # Also buy dips in neutral market if RSI oversold
        if trend == 'NEUTRAL' and long_count >= 2:
            return 'LONG', long_score, long_count
        # Skip shorts entirely
        return None, 0, 0
    
    # Original logic for non-biased trading
    if trend == 'BULLISH' and long_count >= 2 and long_count > short_count:
        return 'LONG', long_score + trend_strength, long_count
    elif trend == 'BEARISH' and short_count >= 2 and short_count > long_count:
        return 'SHORT', short_score + trend_strength, short_count
    
    return None, 0, max(long_count, short_count)

# =============================================================================
# 📊 FETCH HISTORICAL DATA
# =============================================================================

def fetch_historical_data(symbol: str = 'XRPUSD', days: int = 7) -> List[dict]:
    """Fetch historical OHLCV data from Delta Exchange"""
    print(f"\n📊 Fetching {days} days of {symbol} historical data...")
    
    product_ids = {'XRPUSD': 14969, 'ETHUSD': 3136, 'BTCUSD': 139}
    product_id = product_ids.get(symbol, 14969)
    
    end_time = int(time.time())
    start_time = end_time - (days * 24 * 60 * 60)
    
    url = f"https://api.india.delta.exchange/v2/history/candles"
    params = {
        'resolution': '1m',  # 1 minute candles
        'symbol': symbol,
        'start': start_time,
        'end': end_time
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        
        if data.get('success') and data.get('result'):
            candles = data['result']
            print(f"✅ Fetched {len(candles)} candles")
            # Debug: Show first candle structure
            if candles:
                print(f"   Sample candle: {candles[0]}")
            return candles
        else:
            print(f"❌ API Error: {data}")
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
    
    # Generate synthetic data if API fails
    print("⚠️ Generating synthetic data for testing...")
    return generate_synthetic_data(days)

def generate_synthetic_data(days: int = 7) -> List[dict]:
    """Generate realistic synthetic price data for backtesting"""
    candles = []
    base_price = 2.15  # XRP starting price
    
    num_candles = days * 24 * 60  # 1 minute candles
    current_time = int(time.time()) - (days * 24 * 60 * 60)
    
    for i in range(num_candles):
        # Random walk with mean reversion
        change = random.gauss(0, 0.001)  # 0.1% std dev
        
        # Add some trending behavior
        if i % 60 < 30:  # First 30 mins of each hour tend up
            change += 0.0001
        else:
            change -= 0.0001
        
        base_price *= (1 + change)
        
        # Keep price realistic
        base_price = max(1.5, min(3.0, base_price))
        
        high = base_price * (1 + abs(random.gauss(0, 0.002)))
        low = base_price * (1 - abs(random.gauss(0, 0.002)))
        open_price = base_price * (1 + random.gauss(0, 0.001))
        close = base_price
        
        candles.append({
            'time': current_time + i * 60,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': random.randint(10000, 100000)
        })
    
    print(f"✅ Generated {len(candles)} synthetic candles")
    return candles

# =============================================================================
# 🧪 BACKTESTER
# =============================================================================

class Backtester:
    def __init__(self, starting_balance: float = 0.12, leverage: int = 50):
        self.starting_balance = starting_balance
        self.balance = starting_balance
        self.leverage = leverage
        self.position = None
        self.trades = []
        self.equity_curve = []
        self.peak_balance = starting_balance
        self.max_drawdown = 0
        
    def run(self, candles: List[dict]) -> dict:
        """Run backtest on historical data"""
        print(f"\n🧪 Running Backtest...")
        print(f"   Starting Balance: ${self.starting_balance:.4f}")
        print(f"   Leverage: {self.leverage}x")
        print(f"   Candles: {len(candles)}")
        
        prices_history = deque(maxlen=100)
        
        signal_count = 0
        for i, candle in enumerate(candles):
            price = candle['close']
            prices_history.append(price)
            
            # Track equity
            self.equity_curve.append({
                'time': candle['time'],
                'balance': self.balance,
                'price': price
            })
            
            # Update drawdown
            if self.balance > self.peak_balance:
                self.peak_balance = self.balance
            dd = (self.peak_balance - self.balance) / self.peak_balance
            self.max_drawdown = max(self.max_drawdown, dd)
            
            # Check if we have enough data
            if len(prices_history) < 30:
                continue
            
            prices_list = list(prices_history)
            
            # Manage existing position
            if self.position:
                self._manage_position(price)
                continue
            
            # Look for new signals
            direction, confidence, confirmations = get_combined_signal(prices_list)
            
            # Debug: Show signal info occasionally
            if i == 100 or (i % 500 == 0 and i > 0):
                signals = get_signals(prices_list)
                print(f"   Candle {i}: Price={price:.4f}, Signals={len(signals)}, Conf={confidence:.1f}, Confirms={confirmations}")
                if signals:
                    print(f"      Signals: {list(signals.keys())}")
            
            if direction and confirmations >= 2:  # Need 2 confirmations for quality
                signal_count += 1
                # Calculate position size
                position_value = self.balance * 0.5 * self.leverage
                lots = int(position_value / price)
                
                if lots > 0:
                    self._open_position(direction, price, lots, candle['time'])
        
        print(f"   Total signals detected: {signal_count}")
        
        # Close any remaining position
        if self.position:
            self._close_position(candles[-1]['close'], "Backtest End", candles[-1]['time'])
        
        return self._generate_report()
    
    def _open_position(self, direction: str, price: float, lots: int, timestamp: int):
        self.position = {
            'direction': direction,
            'entry_price': price,
            'lots': lots,
            'entry_time': timestamp,
            'peak_pnl': 0
        }
    
    def _manage_position(self, current_price: float):
        if not self.position:
            return
        
        entry = self.position['entry_price']
        direction = self.position['direction']
        
        if direction == 'LONG':
            pnl_pct = (current_price - entry) / entry
        else:
            pnl_pct = (entry - current_price) / entry
        
        self.position['peak_pnl'] = max(self.position['peak_pnl'], pnl_pct)
        
        # Stop Loss
        if pnl_pct <= -STOP_LOSS_PCT:
            self._close_position(current_price, "Stop Loss", 0)
            return
        
        # Take Profit
        if pnl_pct >= TARGET_PCT:
            self._close_position(current_price, "Take Profit", 0)
            return
        
        # Trailing Stop
        if self.position['peak_pnl'] >= TRAIL_START:
            trail_level = self.position['peak_pnl'] - TRAIL_DISTANCE
            if pnl_pct < trail_level:
                self._close_position(current_price, "Trail Stop", 0)
                return
    
    def _close_position(self, price: float, reason: str, timestamp: int):
        if not self.position:
            return
        
        entry = self.position['entry_price']
        direction = self.position['direction']
        lots = self.position['lots']
        
        if direction == 'LONG':
            pnl_pct = (price - entry) / entry
        else:
            pnl_pct = (entry - price) / entry
        
        # Calculate actual PnL
        position_value = entry * lots / self.leverage
        gross_pnl = position_value * pnl_pct
        fees = position_value * ROUND_TRIP_FEE
        net_pnl = gross_pnl - fees
        
        self.balance += net_pnl
        
        self.trades.append({
            'direction': direction,
            'entry': entry,
            'exit': price,
            'pnl_pct': pnl_pct * 100,
            'net_pnl': net_pnl,
            'reason': reason,
            'win': net_pnl > 0
        })
        
        self.position = None
    
    def _generate_report(self) -> dict:
        """Generate backtest report"""
        wins = sum(1 for t in self.trades if t['win'])
        losses = len(self.trades) - wins
        
        total_pnl = sum(t['net_pnl'] for t in self.trades)
        gross_profit = sum(t['net_pnl'] for t in self.trades if t['win'])
        gross_loss = sum(t['net_pnl'] for t in self.trades if not t['win'])
        
        avg_win = gross_profit / wins if wins > 0 else 0
        avg_loss = gross_loss / losses if losses > 0 else 0
        
        profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')
        
        return {
            'starting_balance': self.starting_balance,
            'ending_balance': self.balance,
            'total_return_pct': (self.balance - self.starting_balance) / self.starting_balance * 100,
            'total_trades': len(self.trades),
            'wins': wins,
            'losses': losses,
            'win_rate': wins / len(self.trades) * 100 if self.trades else 0,
            'total_pnl': total_pnl,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': self.max_drawdown * 100,
            'trades': self.trades
        }

# =============================================================================
# 🚀 MAIN
# =============================================================================

def print_report(report: dict):
    """Print backtest report"""
    print("\n" + "═"*70)
    print("📊 BACKTEST RESULTS".center(70))
    print("═"*70)
    
    emoji = "🟢" if report['total_return_pct'] > 0 else "🔴"
    
    print(f"""
   💰 Starting Balance:    ${report['starting_balance']:.4f}
   💰 Ending Balance:      ${report['ending_balance']:.4f}
   {emoji} Total Return:        {report['total_return_pct']:+.2f}%
   
   📊 Total Trades:        {report['total_trades']}
   ✅ Wins:                {report['wins']}
   ❌ Losses:              {report['losses']}
   📈 Win Rate:            {report['win_rate']:.1f}%
   
   💵 Total P&L:           ${report['total_pnl']:.6f}
   💚 Gross Profit:        ${report['gross_profit']:.6f}
   💔 Gross Loss:          ${report['gross_loss']:.6f}
   
   📈 Avg Win:             ${report['avg_win']:.6f}
   📉 Avg Loss:            ${report['avg_loss']:.6f}
   ⚖️ Profit Factor:       {report['profit_factor']:.2f}
   
   📉 Max Drawdown:        {report['max_drawdown']:.2f}%
    """)
    
    print("═"*70)
    
    # Analysis
    print("\n📈 ANALYSIS:")
    
    if report['win_rate'] >= 50:
        print(f"   ✅ Win rate {report['win_rate']:.1f}% is GOOD (above 50%)")
    else:
        print(f"   ⚠️ Win rate {report['win_rate']:.1f}% is below 50% - needs improvement")
    
    if report['profit_factor'] >= 1.5:
        print(f"   ✅ Profit factor {report['profit_factor']:.2f} is EXCELLENT")
    elif report['profit_factor'] >= 1.0:
        print(f"   ⚠️ Profit factor {report['profit_factor']:.2f} is marginal")
    else:
        print(f"   ❌ Profit factor {report['profit_factor']:.2f} - strategy is LOSING money")
    
    if report['max_drawdown'] <= 5:
        print(f"   ✅ Max drawdown {report['max_drawdown']:.1f}% is LOW (good risk management)")
    elif report['max_drawdown'] <= 10:
        print(f"   ⚠️ Max drawdown {report['max_drawdown']:.1f}% is moderate")
    else:
        print(f"   ❌ Max drawdown {report['max_drawdown']:.1f}% is HIGH - risky!")
    
    # Projected monthly return
    if report['total_trades'] > 0:
        days_tested = 7  # Assuming 7 days of data
        daily_return = report['total_return_pct'] / days_tested
        monthly_return = daily_return * 30
        print(f"\n   📅 Projected Monthly Return: {monthly_return:+.1f}%")
        
        if monthly_return > 0:
            starting = report['starting_balance']
            month1 = starting * (1 + monthly_return/100)
            month3 = starting * ((1 + monthly_return/100) ** 3)
            month6 = starting * ((1 + monthly_return/100) ** 6)
            print(f"   📈 Compound Growth Projection:")
            print(f"      Month 1: ${month1:.4f}")
            print(f"      Month 3: ${month3:.4f}")
            print(f"      Month 6: ${month6:.4f}")
    
    print("\n" + "═"*70)

def main():
    print("\n" + "═"*70)
    print("🧪 ALADDIN STRATEGY BACKTESTER".center(70))
    print("═"*70)
    
    # Fetch historical data
    candles = fetch_historical_data('XRPUSD', days=7)
    
    if not candles:
        print("❌ No data available for backtesting")
        return
    
    # Run backtest with ₹10 balance (~$0.12)
    backtester = Backtester(starting_balance=0.12, leverage=50)
    report = backtester.run(candles)
    
    # Print results
    print_report(report)
    
    # Show sample trades
    if report['trades']:
        print("\n📝 SAMPLE TRADES (Last 10):")
        print("-" * 70)
        for trade in report['trades'][-10:]:
            emoji = "🟢" if trade['win'] else "🔴"
            print(f"   {emoji} {trade['direction']:5} | Entry: {trade['entry']:.4f} | Exit: {trade['exit']:.4f} | PnL: {trade['pnl_pct']:+.2f}% | {trade['reason']}")
        print("-" * 70)

if __name__ == "__main__":
    main()
