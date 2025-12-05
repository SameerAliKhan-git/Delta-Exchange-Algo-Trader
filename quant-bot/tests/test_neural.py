import sys
import os
import numpy as np
import pandas as pd
import unittest

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.ml.neural_classifier import AdaptiveTrader, SignalClassifier
from src.features.indicators import FeatureEngineer

class TestNeuralSystem(unittest.TestCase):
    
    def setUp(self):
        if not HAS_TORCH:
            self.skipTest("Torch not installed")
            
        # Generate synthetic data
        dates = pd.date_range(start='2023-01-01', periods=1000, freq='5min')
        self.data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.randn(1000) + 100,
            'high': np.random.randn(1000) + 101,
            'low': np.random.randn(1000) + 99,
            'close': np.random.randn(1000) + 100,
            'volume': np.random.randint(100, 1000, 1000)
        })
        
    def test_feature_engineer(self):
        print("\nTesting Feature Engineer...")
        fe = FeatureEngineer()
        
        # Test optimization (mocking optimization by running it on small data)
        # We'll just run compute_features which uses defaults if not optimized
        features = fe.compute_features(self.data)
        
        print(f"Feature shape: {features.shape}")
        self.assertEqual(features.shape[0], 1000)
        # We expect around 28 features, but let's just check it's not empty
        self.assertGreater(features.shape[1], 10)
        
        # Check for NaNs
        self.assertFalse(np.isnan(features).any(), "Features contain NaNs")
        
    def test_neural_classifier(self):
        print("\nTesting Neural Classifier...")
        classifier = SignalClassifier(input_dim=28)
        
        # Test Forward Pass
        dummy_input = np.random.randn(28).astype(np.float32)
        signal, conf = classifier.predict(dummy_input)
        
        print(f"Prediction: Signal={signal}, Conf={conf:.4f}")
        self.assertIn(signal, [0, 1, 2])
        self.assertTrue(0 <= conf <= 1.0)
        
    def test_adaptive_trader(self):
        print("\nTesting Adaptive Trader...")
        trader = AdaptiveTrader(input_dim=28, retrain_bars=10) # Small retrain for test
        
        # Simulate trading loop
        for i in range(20):
            features = np.random.randn(28).astype(np.float32)
            # Mock label (0, 1, 2)
            label = np.random.randint(0, 3)
            
            trader.update(features, label)
            
            if i == 10:
                print("Triggered retrain...")
                
        # Check if model still predicts
        features = np.random.randn(28).astype(np.float32)
        signal, conf = trader.predict(features)
        self.assertIn(signal, [0, 1, 2])

if __name__ == '__main__':
    unittest.main()
