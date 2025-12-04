# 🤖 Delta Exchange Trading Bot - Complete Setup

## Quick Start

### 1. Install Dependencies
```powershell
cd "d:\Delta Exchange Algo\quant-bot"
pip install -r requirements.txt
```

### 2. Configure API Keys
Edit `.env` file with your Delta Exchange API credentials:
```
DELTA_API_KEY=your_api_key_here
DELTA_API_SECRET=your_api_secret_here
```

### 3. Download Data & Train
```powershell
# Full pipeline (download + train + backtest)
python run_delta_bot.py --mode full

# Or step by step:
python run_delta_bot.py --mode data      # Download historical data
python run_delta_bot.py --mode train     # Train ML models
python run_delta_bot.py --mode backtest  # Run backtest
```

### 4. Start Paper Trading
```powershell
python run_delta_bot.py --mode paper
```

### 5. Go Live (requires --confirm)
```powershell
python run_delta_bot.py --mode live --confirm
```

---

## 📊 Latest Backtest Results

| Metric | BTCUSD | ETHUSD |
|--------|--------|--------|
| Initial Capital | $100,000 | $100,000 |
| Final Capital | $133,215 | $131,657 |
| Total Return | **+33.22%** | **+31.66%** |
| Sharpe Ratio | **9.86** | **8.32** |
| Max Drawdown | 1.14% | 1.60% |
| Win Rate | 78.5% | 75.6% |
| Total Trades | 540 | 476 |
| Profit Factor | 7.04 | 6.82 |

---

## 🏗️ Architecture

```
run_delta_bot.py          # Master runner script
├── DataDownloader        # Downloads from Delta Exchange API
├── FeatureEngineer       # 63 technical features
├── ModelTrainer          # RF + GB + XGBoost ensemble
├── Backtester           # Event-driven backtest engine
└── PaperTrader          # Paper trading with live signals

src/
├── services/
│   └── delta_exchange.py  # Delta Exchange REST + WebSocket client
├── trading/
│   ├── production_pipeline.py   # Trade execution pipeline
│   └── paper_trading_orchestrator.py
├── strategies/           # Trading strategies
├── risk/                 # Risk management
├── ml/                   # ML models
│   ├── cost_aware_objective.py   # Net P&L optimization
│   └── meta_learner_bandit.py    # Thompson Sampling strategy selection
└── backtest/             # Backtest engine
```

---

## 🔧 Configuration

### Trading Pairs
Default: `['BTCUSD', 'ETHUSD']`

### Risk Limits
- Max Position: 20% of capital per trade
- Position Size: 2% risk per trade
- Max Drawdown: 15% circuit breaker
- Daily Loss Limit: 3%

### Model Settings
- Ensemble: Random Forest + Gradient Boosting + XGBoost
- Features: 63 technical indicators
- Training: 80/20 train/test split (no shuffling for time series)

---

## 📁 File Structure

```
quant-bot/
├── .env                  # API keys and configuration
├── run_delta_bot.py      # Master runner script
├── data/
│   └── ohlcv/           # Historical price data
├── models/
│   └── ensemble_model.pkl  # Trained models
├── reports/
│   └── backtest_*.json  # Backtest results
└── src/
    ├── services/        # Exchange clients
    ├── trading/         # Trading logic
    ├── strategies/      # Strategy implementations
    ├── risk/            # Risk management
    └── ml/              # Machine learning
```

---

## 🚀 Commands Reference

| Command | Description |
|---------|-------------|
| `python run_delta_bot.py --mode full` | Complete pipeline |
| `python run_delta_bot.py --mode data` | Download data only |
| `python run_delta_bot.py --mode train` | Train models only |
| `python run_delta_bot.py --mode backtest` | Backtest only |
| `python run_delta_bot.py --mode paper` | Start paper trading |
| `python run_delta_bot.py --mode live --confirm` | Start live trading |

### Additional Options
- `--symbols BTCUSD ETHUSD SOLUSD` - Specify trading pairs
- `--confirm` - Required for live trading

---

## ⚠️ Important Notes

1. **Paper Trading First**: Always test in paper mode before going live
2. **API Rate Limits**: Delta Exchange has rate limits (10 req/sec)
3. **Risk Management**: Never disable risk limits in production
4. **Monitoring**: Use the dashboard at http://localhost:8080

---

## 📞 Support

For issues with:
- **Delta Exchange API**: https://docs.delta.exchange
- **Trading Bot**: Check logs in `logs/` directory
