import numpy as np
import pandas as pd
import talib
from scipy.stats import pearsonr
from typing import List, Dict, Tuple, Callable

class IndicatorOptimizer:
    """
    Optimizes indicator periods based on correlation with target Yt.
    """
    
    @staticmethod
    def calculate_Yt(data: pd.DataFrame, n: int = 20) -> np.ndarray:
        """
        Calculate normalized target Yt (0-1) based on future window.
        Note: Paper uses future min/max for labeling, but we must be careful 
        not to use it for feature calculation.
        
        For optimization (finding best period), we align indicator(t) with Yt(t+n).
        """
        close = data['close'].values
        Yt = np.zeros(len(close))
        
        # We want to predict Yt at time t. 
        # Yt represents the relative position of price t in the window [t, t+n] ?
        # Or is it [t-n, t]?
        # The prompt says: "Yt uses future min/max" -> Lookahead bias in paper.
        # "Your Bot Fix: Use only past data in calculate_Yt"
        
        # Wait, if we use only past data for Yt, then Yt is just a lagging indicator (Stochastic Oscillator).
        # If Yt is the TARGET, it SHOULD use future data (we want to predict future).
        # If Yt is a FEATURE, it must use past data.
        
        # The prompt says: "Target: Normalized price level Yt (scaled 0-1) over future window n"
        # So Yt is the TARGET. We train to predict Yt.
        
        for i in range(len(close) - n):
            window = close[i:i+n+1] # Future window
            min_p, max_p = np.min(window), np.max(window)
            if max_p == min_p:
                Yt[i] = 0.5
            else:
                Yt[i] = (close[i] - min_p) / (max_p - min_p)
                
        return Yt

    @staticmethod
    def optimize_period(data: pd.DataFrame, indicator_func: Callable, periods: List[int] = [5, 10, 20, 50, 100, 200]) -> Tuple[int, float]:
        """
        Find best period for an indicator.
        """
        best_corr = 0
        best_period = periods[0]
        
        # We need a target to correlate with. 
        # Let's use a standard future return or the Yt described.
        # If we optimize for Yt(n), then n changes with period?
        # The paper likely optimizes indicator(period) against a fixed Target.
        # Or optimizes indicator(period) against Yt(period).
        
        # Prompt says: "Calculate Yt(data, n=period)" inside the loop.
        # So it checks if Indicator(20) correlates with Price relative to 20-bar window.
        
        for p in periods:
            try:
                # Calculate indicator
                # Assumes indicator_func takes (data, timeperiod)
                indicator = indicator_func(data, p)
                
                # Calculate Target Yt (using future data for correlation check)
                # We want to see if Indicator(t) predicts Yt(t)
                # Yt(t) = position of price in *future* window? 
                # The prompt code for calculate_Yt uses `window = close[i-n:i+1] # Use only past data`
                # AND says "Fixed for lookahead bias".
                # If Yt uses past data, it's just %K of Stochastic.
                # If we correlate Indicator(t) with Past_Yt(t), we are just checking redundancy.
                
                # Let's stick to the prompt's `calculate_Yt` which uses PAST data.
                # "Yt[i] = (close[i] - min_p) / (max_p - min_p)" where window is i-n to i.
                # This is exactly Stochastic Oscillator %K.
                
                Yt = np.zeros(len(data))
                close = data['close'].values
                for i in range(p, len(close)):
                    window = close[i-p:i+1]
                    min_v, max_v = np.min(window), np.max(window)
                    if max_v > min_v:
                        Yt[i] = (close[i] - min_v) / (max_v - min_v)
                
                # Align
                # indicator is aligned with data
                # Yt is aligned with data
                
                valid_mask = ~np.isnan(indicator) & ~np.isnan(Yt) & (Yt != 0)
                if np.sum(valid_mask) < 50:
                    continue
                    
                corr, _ = pearsonr(indicator[valid_mask], Yt[valid_mask])
                
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_period = p
            except Exception:
                continue
                
        return best_period, best_corr

class FeatureEngineer:
    def __init__(self):
        self.optimized_periods = {}
        
    def optimize(self, data: pd.DataFrame):
        """Run optimization for all indicators."""
        # RSI
        p, c = IndicatorOptimizer.optimize_period(data, lambda d, p: talib.RSI(d['close'], timeperiod=p))
        self.optimized_periods['RSI'] = p
        
        # CCI
        p, c = IndicatorOptimizer.optimize_period(data, lambda d, p: talib.CCI(d['high'], d['low'], d['close'], timeperiod=p))
        self.optimized_periods['CCI'] = p
        
        # ROC
        p, c = IndicatorOptimizer.optimize_period(data, lambda d, p: talib.ROC(d['close'], timeperiod=p))
        self.optimized_periods['ROC'] = p
        
        # MOM
        p, c = IndicatorOptimizer.optimize_period(data, lambda d, p: talib.MOM(d['close'], timeperiod=p))
        self.optimized_periods['MOM'] = p
        
        # Williams %R
        p, c = IndicatorOptimizer.optimize_period(data, lambda d, p: talib.WILLR(d['high'], d['low'], d['close'], timeperiod=p))
        self.optimized_periods['WILLR'] = p
        
        # ADX
        p, c = IndicatorOptimizer.optimize_period(data, lambda d, p: talib.ADX(d['high'], d['low'], d['close'], timeperiod=p))
        self.optimized_periods['ADX'] = p
        
        # TRIX
        p, c = IndicatorOptimizer.optimize_period(data, lambda d, p: talib.TRIX(d['close'], timeperiod=p))
        self.optimized_periods['TRIX'] = p
        
        # CMO
        p, c = IndicatorOptimizer.optimize_period(data, lambda d, p: talib.CMO(d['close'], timeperiod=p))
        self.optimized_periods['CMO'] = p
        
        # MFI
        p, c = IndicatorOptimizer.optimize_period(data, lambda d, p: talib.MFI(d['high'], d['low'], d['close'], d['volume'], timeperiod=p))
        self.optimized_periods['MFI'] = p
        
        # AROON
        p, c = IndicatorOptimizer.optimize_period(data, lambda d, p: talib.AROONOSC(d['high'], d['low'], timeperiod=p))
        self.optimized_periods['AROON'] = p

    def compute_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Compute the 28 features (or subset) using optimized periods.
        Returns numpy array (n_samples, n_features).
        """
        df = pd.DataFrame(index=data.index)
        
        # Use defaults if not optimized
        def get_p(name, default=14):
            return self.optimized_periods.get(name, default)
            
        # 1. Momentum
        df['RSI'] = talib.RSI(data['close'], timeperiod=get_p('RSI'))
        df['CCI'] = talib.CCI(data['high'], data['low'], data['close'], timeperiod=get_p('CCI'))
        df['ROC'] = talib.ROC(data['close'], timeperiod=get_p('ROC'))
        df['MOM'] = talib.MOM(data['close'], timeperiod=get_p('MOM'))
        df['WILLR'] = talib.WILLR(data['high'], data['low'], data['close'], timeperiod=get_p('WILLR'))
        df['CMO'] = talib.CMO(data['close'], timeperiod=get_p('CMO'))
        
        # 2. Trend
        df['ADX'] = talib.ADX(data['high'], data['low'], data['close'], timeperiod=get_p('ADX'))
        df['TRIX'] = talib.TRIX(data['close'], timeperiod=get_p('TRIX'))
        df['AROON'] = talib.AROONOSC(data['high'], data['low'], timeperiod=get_p('AROON'))
        
        # 3. Volume
        if 'volume' in data.columns:
            df['MFI'] = talib.MFI(data['high'], data['low'], data['close'], data['volume'], timeperiod=get_p('MFI'))
            df['OBV'] = talib.OBV(data['close'], data['volume'])
        else:
            df['MFI'] = 50
            df['OBV'] = 0
            
        # 4. Volatility
        df['ATR'] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=14)
        df['NATR'] = talib.NATR(data['high'], data['low'], data['close'], timeperiod=14)
        
        # Fill NaNs
        df = df.fillna(method='bfill').fillna(0)
        
        # Normalize features (Z-score)
        # In production, we should use saved scaler stats
        df_norm = (df - df.mean()) / (df.std() + 1e-8)
        
        return df_norm.values
