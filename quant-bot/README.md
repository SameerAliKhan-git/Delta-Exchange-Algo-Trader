# Quant-Bot: Financial Machine Learning Trading System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)]()

A production-ready quantitative trading system implementing concepts from **Advances in Financial Machine Learning** (López de Prado), market microstructure research, and modern ML techniques.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           QUANT-BOT ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌───────────┐  │
│  │   Data   │───▶│   Feature    │───▶│   Labeling  │───▶│   Model   │  │
│  │  Loader  │    │  Engineering │    │   (AFML)    │    │  Training │  │
│  └──────────┘    └──────────────┘    └─────────────┘    └─────┬─────┘  │
│                                                               │         │
│  ┌──────────────────────────────────────────────────────────┘         │
│  │                                                                     │
│  ▼                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                │
│  │  Backtest   │───▶│  Execution  │───▶│    Risk     │                │
│  │   Engine    │    │  Simulator  │    │   Manager   │                │
│  └─────────────┘    └─────────────┘    └─────────────┘                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/quant-bot.git
cd quant-bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run demo
python scripts/run_demo.py
```

### Docker

```bash
docker-compose up --build
```

## 📚 AFML Concepts Implemented

| Concept | Module | Description |
|---------|--------|-------------|
| **Triple-Barrier Labeling** | `src/labeling/afml_labeling.py` | Dynamic profit-taking/stop-loss with time barrier |
| **Sample Weights** | `src/labeling/afml_labeling.py` | Uniqueness-based weighting to reduce overfitting |
| **Meta-Labeling** | `src/labeling/afml_labeling.py` | Two-stage prediction: direction + bet sizing |
| **Purged K-Fold CV** | `src/models/train.py` | Cross-validation without lookahead bias |
| **Fractional Differentiation** | `src/features/feature_engineer.py` | Memory-preserving stationarity |
| **CUSUM Filter** | `src/features/feature_engineer.py` | Event-driven sampling |

## 📁 Project Structure

```
quant-bot/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── docker/
│   └── Dockerfile              # Container definition
├── docker-compose.yml          # Multi-container orchestration
├── src/
│   ├── data/
│   │   └── loader.py           # Data ingestion & caching
│   ├── features/
│   │   └── feature_engineer.py # Feature pipeline
│   ├── labeling/
│   │   └── afml_labeling.py    # AFML labeling methods
│   ├── models/
│   │   └── train.py            # Model training harness
│   ├── backtest/
│   │   └── engine.py           # Backtesting engine
│   ├── execution/
│   │   └── simulator.py        # Order execution simulator
│   ├── risk/
│   │   └── risk_manager.py     # Risk management
│   ├── utils/
│   │   └── metrics.py          # Performance metrics
│   └── config.py               # Configuration management
├── tests/
│   ├── test_loader.py
│   ├── test_labeling.py
│   └── test_backtester.py
├── notebooks/
│   ├── supervised_demo.ipynb
│   └── meta_labeling_demo.ipynb
├── docs/
│   └── ROADMAP.md              # 12-week implementation plan
├── data/
│   └── sample_ohlcv.csv        # Sample data
└── scripts/
    └── run_demo.py             # Demo runner
```

## 🔬 Key Features

### 1. Triple-Barrier Method
Labels trades based on which barrier is hit first:
- **Upper barrier**: Take-profit target reached
- **Lower barrier**: Stop-loss triggered  
- **Vertical barrier**: Time expiration

### 2. Meta-Labeling
Two-stage ML approach:
1. **Primary Model**: Predicts direction (long/short/neutral)
2. **Meta Model**: Predicts probability of primary model being correct

### 3. Slippage-Aware Backtesting
Realistic simulation including:
- Market impact modeling
- Latency simulation
- Commission structure
- Fill probability

### 4. Risk Management
- Position sizing (Kelly criterion, fixed fractional)
- Maximum exposure limits
- Drawdown circuit breakers
- Correlation-aware portfolio risk

## 📊 Example Usage

```python
from src.data.loader import CSVDataLoader
from src.features.feature_engineer import FeaturePipeline
from src.labeling.afml_labeling import TripleBarrierLabeler
from src.models.train import ModelTrainer
from src.backtest.engine import BacktestEngine

# Load data
loader = CSVDataLoader("data/sample_ohlcv.csv")
df = loader.load(start="2020-01-01", end="2023-12-31")

# Engineer features
pipeline = FeaturePipeline()
features = pipeline.fit_transform(df)

# Create labels
labeler = TripleBarrierLabeler(
    profit_taking=0.02,  # 2% take profit
    stop_loss=0.01,      # 1% stop loss
    max_holding=20       # 20 bars max
)
labels = labeler.fit_transform(df['close'])

# Train model
trainer = ModelTrainer(model_type='xgboost')
model = trainer.train(features, labels)

# Backtest
engine = BacktestEngine(initial_capital=100000)
results = engine.run(df, model, features)
print(results.summary())
```

## ⚠️ Disclaimers

- **Not Financial Advice**: This software is for educational and research purposes only.
- **No Warranty**: Use at your own risk. Past performance does not guarantee future results.
- **Live Trading**: Requires additional compliance, legal review, and risk controls.
- **API Keys**: Never commit credentials. Use environment variables or secure vaults.

## 📖 References

### Books
- López de Prado, M. (2018). *Advances in Financial Machine Learning*
- Jansen, S. (2020). *Machine Learning for Algorithmic Trading*
- Chan, E. (2017). *Machine Trading*
- Aronson, D. (2006). *Evidence-Based Technical Analysis*

### Papers
- Sirignano, J. (2019). *Deep Learning for Limit Order Books*
- López de Prado (2020). *10 Reasons Most ML Funds Fail*

### Open Source
- [mlfinlab](https://github.com/hudson-and-thames/mlfinlab)
- [FinRL](https://github.com/AI4Finance-Foundation/FinRL)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
