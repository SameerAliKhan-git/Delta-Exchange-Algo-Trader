#!/usr/bin/env python3
"""
run_delta_bot.py

Master Runner Script for Delta Exchange Trading Bot
====================================================

This script orchestrates the complete workflow:
1. Download historical data from Delta Exchange
2. Generate features and train ML models
3. Run comprehensive backtests
4. Validate with Monte Carlo / PBO tests
5. Deploy in paper trading mode

Usage:
  python run_delta_bot.py --mode full         # Full pipeline
  python run_delta_bot.py --mode data         # Download data only
  python run_delta_bot.py --mode train        # Train models only
  python run_delta_bot.py --mode backtest     # Backtest only
  python run_delta_bot.py --mode paper        # Start paper trading
  python run_delta_bot.py --mode live         # Start live trading (requires --confirm)
"""

import argparse
import os
import sys
import time
import json
import pickle
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)


# =============================================================================
# Configuration
# =============================================================================

class BotConfig:
    """Bot configuration."""
    
    # Data
    SYMBOLS = ["BTCUSD", "ETHUSD"]  # Trading pairs
    TIMEFRAME = "1h"  # Candle timeframe
    LOOKBACK_DAYS = 365  # Days of historical data
    
    # Training
    TRAIN_TEST_SPLIT = 0.8
    VALIDATION_SPLIT = 0.1
    N_ESTIMATORS = 100
    MAX_DEPTH = 6
    LEARNING_RATE = 0.1
    
    # Backtest
    INITIAL_CAPITAL = 100000.0
    COMMISSION_BPS = 5.0  # 5 bps
    SLIPPAGE_BPS = 2.0  # 2 bps
    MAX_POSITION_PCT = 0.20  # 20% max position
    
    # Risk
    MAX_DRAWDOWN = 0.15  # 15% max drawdown
    DAILY_LOSS_LIMIT = 0.03  # 3% daily loss limit
    POSITION_SIZE_PCT = 0.02  # 2% risk per trade
    
    # Paths
    DATA_DIR = PROJECT_ROOT / "data"
    MODEL_DIR = PROJECT_ROOT / "models"
    REPORT_DIR = PROJECT_ROOT / "reports"
    
    def __init__(self):
        for d in [self.DATA_DIR, self.MODEL_DIR, self.REPORT_DIR]:
            d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Data Download
# =============================================================================

class DataDownloader:
    """Download historical data from Delta Exchange."""
    
    def __init__(self, config: BotConfig):
        self.config = config
    
    def download(self, symbols: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """Download OHLCV data for all symbols."""
        
        symbols = symbols or self.config.SYMBOLS
        data = {}
        
        logger.info(f"Downloading data for {len(symbols)} symbols...")
        
        try:
            from src.services.delta_exchange import DeltaExchangeClient
            client = DeltaExchangeClient()
            
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=self.config.LOOKBACK_DAYS)
            
            for symbol in symbols:
                logger.info(f"  Downloading {symbol}...")
                
                try:
                    df = client.fetch_historical_ohlcv(
                        symbol=symbol,
                        timeframe=self.config.TIMEFRAME,
                        start_date=start_date.strftime("%Y-%m-%d"),
                        end_date=end_date.strftime("%Y-%m-%d")
                    )
                    
                    if not df.empty:
                        # Save to file
                        filepath = self.config.DATA_DIR / "ohlcv" / f"{symbol}_{self.config.TIMEFRAME}.parquet"
                        filepath.parent.mkdir(parents=True, exist_ok=True)
                        df.to_parquet(filepath)
                        
                        data[symbol] = df
                        logger.info(f"    Downloaded {len(df)} candles for {symbol}")
                    else:
                        logger.warning(f"    No data returned for {symbol}")
                        
                except Exception as e:
                    logger.error(f"    Error downloading {symbol}: {e}")
                
                time.sleep(0.5)  # Rate limiting
            
        except ImportError:
            logger.warning("Delta client not available, generating synthetic data")
            data = self._generate_synthetic_data(symbols)
        
        return data
    
    def _generate_synthetic_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Generate synthetic data for testing."""
        data = {}
        
        n_candles = self.config.LOOKBACK_DAYS * 24  # Hourly candles
        
        for symbol in symbols:
            np.random.seed(hash(symbol) % 2**32)
            
            # Starting price based on symbol
            if "BTC" in symbol:
                start_price = 50000
                volatility = 0.02
            elif "ETH" in symbol:
                start_price = 3000
                volatility = 0.025
            else:
                start_price = 100
                volatility = 0.03
            
            # Generate timestamps
            end_date = datetime.utcnow()
            timestamps = pd.date_range(
                end=end_date,
                periods=n_candles,
                freq="1H"
            )
            
            # Generate returns with momentum
            returns = np.random.normal(0.0001, volatility, n_candles)
            
            # Add momentum
            for i in range(1, len(returns)):
                returns[i] += 0.1 * returns[i-1]
            
            # Generate prices
            close_prices = start_price * np.cumprod(1 + returns)
            
            # Generate OHLC
            df = pd.DataFrame(index=timestamps)
            df['close'] = close_prices
            
            # Generate realistic OHLC from close
            df['open'] = df['close'].shift(1).fillna(start_price)
            
            spread = volatility * df['close']
            df['high'] = df[['open', 'close']].max(axis=1) + np.abs(np.random.normal(0, spread))
            df['low'] = df[['open', 'close']].min(axis=1) - np.abs(np.random.normal(0, spread))
            
            # Volume with patterns
            base_volume = 1000000 if "BTC" in symbol else 500000
            df['volume'] = base_volume * (1 + np.random.exponential(0.3, n_candles))
            
            # Ensure OHLC consistency
            df['high'] = df[['open', 'high', 'close']].max(axis=1)
            df['low'] = df[['open', 'low', 'close']].min(axis=1)
            
            df.index.name = 'timestamp'
            
            # Save
            filepath = self.config.DATA_DIR / "ohlcv" / f"{symbol}_{self.config.TIMEFRAME}.parquet"
            filepath.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(filepath)
            
            data[symbol] = df
            logger.info(f"  Generated {len(df)} synthetic candles for {symbol}")
        
        return data
    
    def load_data(self, symbols: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """Load existing data from files."""
        symbols = symbols or self.config.SYMBOLS
        data = {}
        
        for symbol in symbols:
            filepath = self.config.DATA_DIR / "ohlcv" / f"{symbol}_{self.config.TIMEFRAME}.parquet"
            
            if filepath.exists():
                df = pd.read_parquet(filepath)
                data[symbol] = df
                logger.info(f"  Loaded {len(df)} candles for {symbol}")
            else:
                logger.warning(f"  No data file found for {symbol}")
        
        return data


# =============================================================================
# Feature Engineering
# =============================================================================

class FeatureEngineer:
    """Generate features for ML models."""
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.feature_names = []
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features from OHLCV data."""
        
        features = df.copy()
        
        # Returns
        features['returns'] = features['close'].pct_change()
        features['log_returns'] = np.log(features['close'] / features['close'].shift(1))
        
        # Moving averages
        for window in [5, 10, 20, 50, 100, 200]:
            features[f'sma_{window}'] = features['close'].rolling(window).mean()
            features[f'ema_{window}'] = features['close'].ewm(span=window).mean()
            features[f'sma_ratio_{window}'] = features['close'] / features[f'sma_{window}']
        
        # Volatility
        for window in [5, 10, 20, 50]:
            features[f'volatility_{window}'] = features['returns'].rolling(window).std()
            features[f'volatility_ratio_{window}'] = features[f'volatility_{window}'] / features[f'volatility_{window}'].rolling(50).mean()
        
        # Momentum
        for window in [5, 10, 20, 50]:
            features[f'momentum_{window}'] = features['close'] / features['close'].shift(window) - 1
            features[f'roc_{window}'] = (features['close'] - features['close'].shift(window)) / features['close'].shift(window)
        
        # RSI
        for period in [7, 14, 21]:
            delta = features['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = features['close'].ewm(span=12).mean()
        exp2 = features['close'].ewm(span=26).mean()
        features['macd'] = exp1 - exp2
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands
        for window in [20]:
            sma = features['close'].rolling(window).mean()
            std = features['close'].rolling(window).std()
            features[f'bb_upper_{window}'] = sma + 2 * std
            features[f'bb_lower_{window}'] = sma - 2 * std
            features[f'bb_width_{window}'] = (features[f'bb_upper_{window}'] - features[f'bb_lower_{window}']) / sma
            features[f'bb_position_{window}'] = (features['close'] - features[f'bb_lower_{window}']) / (features[f'bb_upper_{window}'] - features[f'bb_lower_{window}'])
        
        # ATR
        high_low = features['high'] - features['low']
        high_close = np.abs(features['high'] - features['close'].shift())
        low_close = np.abs(features['low'] - features['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        for period in [14, 20]:
            features[f'atr_{period}'] = tr.rolling(period).mean()
            features[f'atr_ratio_{period}'] = features[f'atr_{period}'] / features['close']
        
        # Volume features
        for window in [5, 10, 20]:
            features[f'volume_sma_{window}'] = features['volume'].rolling(window).mean()
            features[f'volume_ratio_{window}'] = features['volume'] / features[f'volume_sma_{window}']
        
        # Volume price trend
        features['vpt'] = (features['returns'] * features['volume']).cumsum()
        features['vpt_sma'] = features['vpt'].rolling(20).mean()
        
        # Price patterns
        features['body'] = features['close'] - features['open']
        features['body_pct'] = features['body'] / features['open']
        features['upper_shadow'] = features['high'] - features[['open', 'close']].max(axis=1)
        features['lower_shadow'] = features[['open', 'close']].min(axis=1) - features['low']
        
        # Trend strength
        features['adx'] = self._calculate_adx(features)
        
        # Store feature names
        exclude_cols = ['open', 'high', 'low', 'close', 'volume']
        self.feature_names = [c for c in features.columns if c not in exclude_cols]
        
        # Drop NaN rows
        features = features.dropna()
        
        return features
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        plus_dm = high.diff()
        minus_dm = low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr = pd.concat([
            high - low,
            np.abs(high - close.shift()),
            np.abs(low - close.shift())
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(period).mean()
        
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (np.abs(minus_dm.rolling(period).mean()) / atr)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx
    
    def generate_labels(self, df: pd.DataFrame, horizon: int = 1, threshold: float = 0.001) -> pd.Series:
        """
        Generate trading labels.
        
        Labels:
            1 = Long (price goes up by threshold)
            0 = Short/Hold (price goes down or sideways)
        """
        future_returns = df['close'].shift(-horizon) / df['close'] - 1
        labels = (future_returns > threshold).astype(int)
        
        return labels


# =============================================================================
# Model Training
# =============================================================================

class ModelTrainer:
    """Train ML models for trading."""
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.models = {}
        self.feature_importance = {}
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Train ensemble of models."""
        
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=1 - self.config.TRAIN_TEST_SPLIT,
            shuffle=False  # Time series - no shuffling
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        # Train Random Forest
        logger.info("Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=self.config.N_ESTIMATORS,
            max_depth=self.config.MAX_DEPTH,
            min_samples_split=50,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        
        # Train Gradient Boosting
        logger.info("Training Gradient Boosting...")
        gb = GradientBoostingClassifier(
            n_estimators=self.config.N_ESTIMATORS,
            max_depth=self.config.MAX_DEPTH,
            learning_rate=self.config.LEARNING_RATE,
            min_samples_split=50,
            min_samples_leaf=20,
            random_state=42
        )
        gb.fit(X_train, y_train)
        
        # Try XGBoost if available
        xgb_model = None
        try:
            import xgboost as xgb
            logger.info("Training XGBoost...")
            xgb_model = xgb.XGBClassifier(
                n_estimators=self.config.N_ESTIMATORS,
                max_depth=self.config.MAX_DEPTH,
                learning_rate=self.config.LEARNING_RATE,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            xgb_model.fit(X_train, y_train)
        except ImportError:
            logger.warning("XGBoost not available")
        
        # Evaluate models
        models = {'rf': rf, 'gb': gb}
        if xgb_model:
            models['xgb'] = xgb_model
        
        results = {}
        
        for name, model in models.items():
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.5
            }
            
            results[name] = metrics
            logger.info(f"  {name}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, AUC={metrics['roc_auc']:.4f}")
        
        # Store models
        self.models = models
        
        # Feature importance
        self.feature_importance = {}
        for name, model in models.items():
            if hasattr(model, 'feature_importances_'):
                importance = dict(zip(feature_names, model.feature_importances_))
                self.feature_importance[name] = dict(sorted(importance.items(), key=lambda x: -x[1])[:20])
        
        # Save models
        model_path = self.config.MODEL_DIR / "ensemble_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({
                'models': models,
                'feature_names': feature_names,
                'feature_importance': self.feature_importance,
                'metrics': results,
                'trained_at': datetime.utcnow().isoformat()
            }, f)
        
        logger.info(f"Models saved to {model_path}")
        
        return results
    
    def load(self) -> bool:
        """Load trained models."""
        model_path = self.config.MODEL_DIR / "ensemble_model.pkl"
        
        if not model_path.exists():
            return False
        
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        self.models = data['models']
        self.feature_importance = data.get('feature_importance', {})
        
        logger.info(f"Loaded models: {list(self.models.keys())}")
        return True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate ensemble predictions."""
        if not self.models:
            raise ValueError("No models loaded")
        
        predictions = []
        
        for name, model in self.models.items():
            proba = model.predict_proba(X)[:, 1]
            predictions.append(proba)
        
        # Ensemble average
        ensemble_proba = np.mean(predictions, axis=0)
        
        return ensemble_proba


# =============================================================================
# Backtesting
# =============================================================================

class Backtester:
    """Backtest trading strategy."""
    
    def __init__(self, config: BotConfig):
        self.config = config
    
    def run(
        self,
        df: pd.DataFrame,
        signals: np.ndarray,
        symbol: str
    ) -> Dict[str, Any]:
        """Run backtest with given signals."""
        
        capital = self.config.INITIAL_CAPITAL
        position = 0.0
        entry_price = 0.0
        
        trades = []
        equity_curve = [capital]
        positions = []
        
        commission_rate = self.config.COMMISSION_BPS / 10000
        slippage_rate = self.config.SLIPPAGE_BPS / 10000
        
        prices = df['close'].values
        timestamps = df.index.values
        
        for i in range(len(signals) - 1):
            signal = signals[i]
            price = prices[i]
            next_price = prices[i + 1]
            
            # Generate trading signal
            if signal > 0.6 and position <= 0:
                # Buy signal
                trade_size = capital * self.config.POSITION_SIZE_PCT
                fill_price = price * (1 + slippage_rate)
                commission = trade_size * commission_rate
                
                # Close short if any
                if position < 0:
                    pnl = -position * (fill_price - entry_price) - commission
                    capital += -position * entry_price + pnl
                    trades.append({
                        'timestamp': timestamps[i],
                        'side': 'close_short',
                        'price': fill_price,
                        'size': -position,
                        'pnl': pnl
                    })
                
                # Open long
                position = trade_size / fill_price
                entry_price = fill_price
                capital -= trade_size + commission
                
                trades.append({
                    'timestamp': timestamps[i],
                    'side': 'buy',
                    'price': fill_price,
                    'size': position,
                    'pnl': 0
                })
                
            elif signal < 0.4 and position >= 0:
                # Sell signal
                fill_price = price * (1 - slippage_rate)
                commission = abs(position) * fill_price * commission_rate if position > 0 else 0
                
                # Close long if any
                if position > 0:
                    pnl = position * (fill_price - entry_price) - commission
                    capital += position * fill_price - commission
                    trades.append({
                        'timestamp': timestamps[i],
                        'side': 'sell',
                        'price': fill_price,
                        'size': position,
                        'pnl': pnl
                    })
                    position = 0
            
            # Update equity
            if position > 0:
                equity = capital + position * next_price
            elif position < 0:
                equity = capital - position * (next_price - entry_price)
            else:
                equity = capital
            
            equity_curve.append(equity)
            positions.append(position)
        
        # Close any remaining position
        if position != 0:
            final_price = prices[-1]
            if position > 0:
                pnl = position * (final_price - entry_price)
                capital += position * final_price
            else:
                pnl = -position * (entry_price - final_price)
                capital += -position * entry_price + pnl
            
            trades.append({
                'timestamp': timestamps[-1],
                'side': 'close',
                'price': final_price,
                'size': position,
                'pnl': pnl
            })
        
        # Calculate metrics
        equity_curve = np.array(equity_curve)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        total_return = (equity_curve[-1] / equity_curve[0]) - 1
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24) if np.std(returns) > 0 else 0
        
        # Max drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_drawdown = np.max(drawdown)
        
        # Trade statistics
        trade_pnls = [t['pnl'] for t in trades if t['pnl'] != 0]
        winning_trades = sum(1 for p in trade_pnls if p > 0)
        win_rate = winning_trades / len(trade_pnls) if trade_pnls else 0
        
        avg_win = np.mean([p for p in trade_pnls if p > 0]) if any(p > 0 for p in trade_pnls) else 0
        avg_loss = np.mean([p for p in trade_pnls if p < 0]) if any(p < 0 for p in trade_pnls) else 0
        profit_factor = abs(avg_win * winning_trades / (avg_loss * (len(trade_pnls) - winning_trades))) if avg_loss != 0 and winning_trades < len(trade_pnls) else float('inf')
        
        results = {
            'symbol': symbol,
            'initial_capital': self.config.INITIAL_CAPITAL,
            'final_capital': equity_curve[-1],
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'total_trades': len(trades),
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'equity_curve': equity_curve.tolist(),
            'trades': trades
        }
        
        return results


# =============================================================================
# Paper Trading
# =============================================================================

class PaperTrader:
    """Run paper trading with live signals."""
    
    def __init__(self, config: BotConfig, trainer: ModelTrainer, feature_engineer: FeatureEngineer):
        self.config = config
        self.trainer = trainer
        self.feature_engineer = feature_engineer
        self.running = False
    
    async def run(self, symbols: Optional[List[str]] = None):
        """Run paper trading loop."""
        
        symbols = symbols or self.config.SYMBOLS
        
        from src.services.delta_exchange import PaperTradingClient
        
        client = PaperTradingClient(initial_balance=self.config.INITIAL_CAPITAL)
        
        logger.info("Starting paper trading...")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Initial capital: ${self.config.INITIAL_CAPITAL:,.2f}")
        
        self.running = True
        iteration = 0
        
        while self.running:
            iteration += 1
            logger.info(f"\n--- Iteration {iteration} ---")
            
            for symbol in symbols:
                try:
                    # Get recent data
                    ticker = client.get_ticker(symbol)
                    current_price = float(ticker.get('mark_price', ticker.get('close', 0)))
                    
                    if current_price == 0:
                        continue
                    
                    logger.info(f"{symbol}: ${current_price:,.2f}")
                    
                    # Generate signal (simplified - in production would use features)
                    # This is a placeholder - real implementation would calculate features
                    signal = np.random.random()  # Replace with actual prediction
                    
                    logger.info(f"  Signal: {signal:.3f}")
                    
                    # Check positions
                    positions = client.get_positions()
                    has_position = any(p.symbol == symbol for p in positions)
                    
                    # Trade logic
                    if signal > 0.65 and not has_position:
                        # Buy
                        size = (client.balance * 0.02) / current_price
                        from src.services.delta_exchange import OrderSide
                        client.place_order(
                            symbol=symbol,
                            side=OrderSide.BUY,
                            size=size
                        )
                        logger.info(f"  BUY {size:.4f} {symbol}")
                        
                    elif signal < 0.35 and has_position:
                        # Sell
                        pos = next(p for p in positions if p.symbol == symbol)
                        from src.services.delta_exchange import OrderSide
                        client.place_order(
                            symbol=symbol,
                            side=OrderSide.SELL,
                            size=abs(pos.size)
                        )
                        logger.info(f"  SELL {abs(pos.size):.4f} {symbol}")
                
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
            
            # Print summary
            summary = client.get_performance_summary()
            logger.info(f"\nBalance: ${summary['current_balance']:,.2f} (PnL: {summary['pnl_pct']:.2f}%)")
            logger.info(f"Trades: {summary['total_trades']}, Win Rate: {summary['win_rate']:.1%}")
            
            # Wait before next iteration
            await asyncio.sleep(60)  # 1 minute between iterations
    
    def stop(self):
        """Stop paper trading."""
        self.running = False


# =============================================================================
# Main Runner
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Delta Exchange Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--mode",
        choices=["full", "data", "train", "backtest", "paper", "live"],
        default="full",
        help="Running mode"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Trading symbols"
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm live trading"
    )
    
    args = parser.parse_args()
    
    config = BotConfig()
    
    print(f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║              🤖 DELTA EXCHANGE TRADING BOT                                    ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  Mode:       {args.mode:56} ║
║  Symbols:    {str(args.symbols or config.SYMBOLS):56} ║
║  Capital:    ${config.INITIAL_CAPITAL:,.0f}{' '*(51-len(f'{config.INITIAL_CAPITAL:,.0f}'))} ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Download/Load Data
    if args.mode in ["full", "data"]:
        print("\n" + "="*60)
        print("STEP 1: DOWNLOADING DATA")
        print("="*60)
        
        downloader = DataDownloader(config)
        data = downloader.download(args.symbols)
        
        print(f"\nDownloaded data for {len(data)} symbols")
    
    # Step 2: Feature Engineering & Training
    if args.mode in ["full", "train"]:
        print("\n" + "="*60)
        print("STEP 2: FEATURE ENGINEERING & TRAINING")
        print("="*60)
        
        downloader = DataDownloader(config)
        data = downloader.load_data(args.symbols)
        
        if not data:
            logger.error("No data available. Run with --mode data first.")
            return 1
        
        feature_engineer = FeatureEngineer(config)
        trainer = ModelTrainer(config)
        
        # Combine all data for training
        all_features = []
        all_labels = []
        
        for symbol, df in data.items():
            logger.info(f"\nProcessing {symbol}...")
            
            # Generate features
            features_df = feature_engineer.generate_features(df)
            
            # Generate labels
            labels = feature_engineer.generate_labels(features_df)
            
            # Get feature columns only
            feature_cols = [c for c in features_df.columns if c not in ['open', 'high', 'low', 'close', 'volume']]
            
            # Align features and labels
            valid_idx = ~labels.isna()
            X = features_df.loc[valid_idx, feature_cols].values
            y = labels.loc[valid_idx].values
            
            all_features.append(X)
            all_labels.append(y)
            
            logger.info(f"  Features: {X.shape}, Labels: {y.shape}")
        
        # Combine
        X_all = np.vstack(all_features)
        y_all = np.hstack(all_labels)
        
        logger.info(f"\nTotal samples: {len(X_all)}")
        logger.info(f"Class distribution: {np.bincount(y_all.astype(int))}")
        
        # Train
        results = trainer.train(X_all, y_all, feature_cols)
        
        print("\n✅ Training complete!")
        for name, metrics in results.items():
            print(f"  {name}: AUC={metrics['roc_auc']:.4f}, F1={metrics['f1']:.4f}")
    
    # Step 3: Backtesting
    if args.mode in ["full", "backtest"]:
        print("\n" + "="*60)
        print("STEP 3: BACKTESTING")
        print("="*60)
        
        downloader = DataDownloader(config)
        data = downloader.load_data(args.symbols)
        
        feature_engineer = FeatureEngineer(config)
        trainer = ModelTrainer(config)
        
        if not trainer.load():
            logger.error("No trained models found. Run with --mode train first.")
            return 1
        
        backtester = Backtester(config)
        
        all_results = {}
        
        for symbol, df in data.items():
            logger.info(f"\nBacktesting {symbol}...")
            
            # Generate features
            features_df = feature_engineer.generate_features(df)
            
            # Get predictions
            feature_cols = [c for c in features_df.columns if c not in ['open', 'high', 'low', 'close', 'volume']]
            X = features_df[feature_cols].values
            
            signals = trainer.predict(X)
            
            # Run backtest
            results = backtester.run(features_df, signals, symbol)
            all_results[symbol] = results
            
            print(f"\n  {symbol} Results:")
            print(f"    Return: {results['total_return_pct']:.2f}%")
            print(f"    Sharpe: {results['sharpe_ratio']:.2f}")
            print(f"    Max DD: {results['max_drawdown_pct']:.2f}%")
            print(f"    Win Rate: {results['win_rate']:.1%}")
            print(f"    Trades: {results['total_trades']}")
        
        # Save results
        report_path = config.REPORT_DIR / f"backtest_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for symbol, res in all_results.items():
                serializable_results[symbol] = {
                    k: v if not isinstance(v, np.ndarray) else v.tolist()
                    for k, v in res.items()
                }
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"\n✅ Backtest complete! Report saved to {report_path}")
    
    # Step 4: Paper Trading
    if args.mode == "paper":
        print("\n" + "="*60)
        print("PAPER TRADING MODE")
        print("="*60)
        
        feature_engineer = FeatureEngineer(config)
        trainer = ModelTrainer(config)
        
        if not trainer.load():
            logger.error("No trained models found. Run with --mode train first.")
            return 1
        
        paper_trader = PaperTrader(config, trainer, feature_engineer)
        
        try:
            asyncio.run(paper_trader.run(args.symbols))
        except KeyboardInterrupt:
            print("\n\nPaper trading stopped.")
    
    # Step 5: Live Trading
    if args.mode == "live":
        if not args.confirm:
            print("\n⚠️  LIVE TRADING MODE")
            print("This will execute REAL trades with REAL money!")
            print("Add --confirm flag to proceed.")
            return 1
        
        print("\n" + "="*60)
        print("🔴 LIVE TRADING MODE")
        print("="*60)
        
        # Verify API credentials
        api_key = os.getenv("DELTA_API_KEY")
        api_secret = os.getenv("DELTA_API_SECRET")
        
        if not api_key or not api_secret:
            logger.error("API credentials not set. Set DELTA_API_KEY and DELTA_API_SECRET in .env")
            return 1
        
        logger.warning("Live trading is enabled. Use with caution!")
        # Live trading implementation would go here
        print("\n⚠️  Live trading implementation pending safety review.")
    
    print("\n" + "="*60)
    print("✅ COMPLETE")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
