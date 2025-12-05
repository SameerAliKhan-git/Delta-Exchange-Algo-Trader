import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional

class SignalClassifier(nn.Module):
    """
    3-Class Neural Classifier (Wait/Buy/Sell) based on IEEE 2018 paper.
    Architecture: Dense -> Dropout(0.5) -> Dense -> Dropout(0.5) -> ...
    """
    def __init__(self, input_dim: int = 28):
        super(SignalClassifier, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),  # Aggressive dropout for noise
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            
            nn.Linear(64, 3)  # [Wait, Buy, Sell]
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, x):
        return self.network(x)
    
    def predict(self, indicators: np.ndarray) -> Tuple[int, float]:
        """
        Predict signal from indicators.
        
        Args:
            indicators: Numpy array of shape (n_features,)
            
        Returns:
            (signal, confidence)
            signal: 0=Wait, 1=Buy, 2=Sell
            confidence: Probability of the predicted class
        """
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(indicators).unsqueeze(0).to(self.device)
            logits = self(x)
            probs = torch.softmax(logits, dim=1)
            
            confidence, signal = torch.max(probs, dim=1)
            
            return signal.item(), confidence.item()

    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Single training step.
        X: (batch_size, n_features)
        y: (batch_size,) class indices
        """
        self.train()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        self.optimizer.zero_grad()
        outputs = self(X_tensor)
        loss = self.criterion(outputs, y_tensor)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

class AdaptiveTrader:
    """
    Wrapper for SignalClassifier that handles retraining.
    """
    def __init__(self, input_dim=28, retrain_bars=1008):
        self.model = SignalClassifier(input_dim)
        self.retrain_bars = retrain_bars
        self.bar_count = 0
        self.recent_data = []
        self.recent_labels = []
        
    def update(self, features: np.ndarray, label: int):
        """
        Update with new data point.
        label: 0=Wait, 1=Buy, 2=Sell
        """
        self.recent_data.append(features)
        self.recent_labels.append(label)
        self.bar_count += 1
        
        # Keep last 2000 bars
        if len(self.recent_data) > 2000:
            self.recent_data.pop(0)
            self.recent_labels.pop(0)
            
        # Retrain condition
        if self.bar_count >= self.retrain_bars:
            self._retrain()
            self.bar_count = 0
            
    def _retrain(self):
        if len(self.recent_data) < 100:
            return
            
        print(f"Retraining on {len(self.recent_data)} samples...")
        X = np.array(self.recent_data)
        y = np.array(self.recent_labels)
        
        # Simple training loop
        epochs = 20
        batch_size = 50
        
        for epoch in range(epochs):
            # Shuffle
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Mini-batch
            for i in range(0, len(X), batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                self.model.train_step(X_batch, y_batch)
                
    def predict(self, features: np.ndarray) -> Tuple[int, float]:
        return self.model.predict(features)
