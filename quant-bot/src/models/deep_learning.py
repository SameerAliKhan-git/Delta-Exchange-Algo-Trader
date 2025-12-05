"""
Deep Learning Models for Trading
================================
Neural network architectures for time series prediction:
- LSTM/GRU for sequential patterns
- Temporal Convolutional Networks (TCN)
- Transformer with attention
- Autoencoder for anomaly detection

No external deep learning framework required - pure NumPy implementation
for educational purposes. For production, use PyTorch/TensorFlow.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# ACTIVATION FUNCTIONS
# =============================================================================

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / (exp_x.sum(axis=-1, keepdims=True) + 1e-10)

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


# =============================================================================
# LSTM CELL
# =============================================================================

class LSTMCell:
    """
    Long Short-Term Memory cell.
    
    Gates:
    - Forget gate: What to discard from cell state
    - Input gate: What new information to store
    - Output gate: What to output from cell state
    """
    
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Xavier initialization
        scale = np.sqrt(2.0 / (input_size + hidden_size))
        
        # Weights for input
        self.Wf = np.random.randn(input_size, hidden_size) * scale  # Forget gate
        self.Wi = np.random.randn(input_size, hidden_size) * scale  # Input gate
        self.Wc = np.random.randn(input_size, hidden_size) * scale  # Cell candidate
        self.Wo = np.random.randn(input_size, hidden_size) * scale  # Output gate
        
        # Weights for hidden state
        self.Uf = np.random.randn(hidden_size, hidden_size) * scale
        self.Ui = np.random.randn(hidden_size, hidden_size) * scale
        self.Uc = np.random.randn(hidden_size, hidden_size) * scale
        self.Uo = np.random.randn(hidden_size, hidden_size) * scale
        
        # Biases
        self.bf = np.zeros(hidden_size)
        self.bi = np.zeros(hidden_size)
        self.bc = np.zeros(hidden_size)
        self.bo = np.zeros(hidden_size)
        
        # For gradient computation
        self.cache = {}
    
    def forward(self, x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through LSTM cell.
        
        Args:
            x: Input at current timestep (batch_size, input_size)
            h_prev: Previous hidden state (batch_size, hidden_size)
            c_prev: Previous cell state (batch_size, hidden_size)
        
        Returns:
            h_next: Next hidden state
            c_next: Next cell state
        """
        # Forget gate
        f = sigmoid(x @ self.Wf + h_prev @ self.Uf + self.bf)
        
        # Input gate
        i = sigmoid(x @ self.Wi + h_prev @ self.Ui + self.bi)
        
        # Cell candidate
        c_tilde = tanh(x @ self.Wc + h_prev @ self.Uc + self.bc)
        
        # Update cell state
        c_next = f * c_prev + i * c_tilde
        
        # Output gate
        o = sigmoid(x @ self.Wo + h_prev @ self.Uo + self.bo)
        
        # Hidden state
        h_next = o * tanh(c_next)
        
        # Cache for backprop
        self.cache = {
            'x': x, 'h_prev': h_prev, 'c_prev': c_prev,
            'f': f, 'i': i, 'c_tilde': c_tilde, 'o': o,
            'c_next': c_next, 'h_next': h_next
        }
        
        return h_next, c_next


class GRUCell:
    """
    Gated Recurrent Unit - simpler than LSTM.
    
    Gates:
    - Reset gate: How much past to forget
    - Update gate: How much to update state
    """
    
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        scale = np.sqrt(2.0 / (input_size + hidden_size))
        
        # Reset gate
        self.Wr = np.random.randn(input_size, hidden_size) * scale
        self.Ur = np.random.randn(hidden_size, hidden_size) * scale
        self.br = np.zeros(hidden_size)
        
        # Update gate
        self.Wz = np.random.randn(input_size, hidden_size) * scale
        self.Uz = np.random.randn(hidden_size, hidden_size) * scale
        self.bz = np.zeros(hidden_size)
        
        # Candidate
        self.Wh = np.random.randn(input_size, hidden_size) * scale
        self.Uh = np.random.randn(hidden_size, hidden_size) * scale
        self.bh = np.zeros(hidden_size)
    
    def forward(self, x: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        """Forward pass through GRU cell."""
        # Reset gate
        r = sigmoid(x @ self.Wr + h_prev @ self.Ur + self.br)
        
        # Update gate
        z = sigmoid(x @ self.Wz + h_prev @ self.Uz + self.bz)
        
        # Candidate hidden state
        h_tilde = tanh(x @ self.Wh + (r * h_prev) @ self.Uh + self.bh)
        
        # New hidden state
        h_next = (1 - z) * h_prev + z * h_tilde
        
        return h_next


# =============================================================================
# LSTM NETWORK
# =============================================================================

class LSTMNetwork:
    """
    Multi-layer LSTM network for sequence prediction.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 num_layers: int = 1, dropout: float = 0.0):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # LSTM layers
        self.layers = []
        for i in range(num_layers):
            layer_input = input_size if i == 0 else hidden_size
            self.layers.append(LSTMCell(layer_input, hidden_size))
        
        # Output layer
        self.Wy = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.by = np.zeros(output_size)
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, List]:
        """
        Forward pass through LSTM network.
        
        Args:
            X: Input sequence (batch_size, seq_len, input_size)
        
        Returns:
            output: Network output (batch_size, output_size)
            hidden_states: List of hidden states at each timestep
        """
        batch_size, seq_len, _ = X.shape
        
        # Initialize hidden states
        h = [np.zeros((batch_size, self.hidden_size)) for _ in range(self.num_layers)]
        c = [np.zeros((batch_size, self.hidden_size)) for _ in range(self.num_layers)]
        
        hidden_states = []
        
        # Process sequence
        for t in range(seq_len):
            x_t = X[:, t, :]
            
            for layer_idx, layer in enumerate(self.layers):
                h[layer_idx], c[layer_idx] = layer.forward(x_t, h[layer_idx], c[layer_idx])
                x_t = h[layer_idx]
                
                # Dropout (during training)
                if self.dropout > 0 and layer_idx < self.num_layers - 1:
                    mask = (np.random.rand(*x_t.shape) > self.dropout) / (1 - self.dropout)
                    x_t = x_t * mask
            
            hidden_states.append(h[-1].copy())
        
        # Output from last hidden state
        output = h[-1] @ self.Wy + self.by
        
        return output, hidden_states
    
    def predict_sequence(self, X: np.ndarray) -> np.ndarray:
        """
        Predict output for each timestep.
        
        Returns:
            outputs: (batch_size, seq_len, output_size)
        """
        batch_size, seq_len, _ = X.shape
        
        h = [np.zeros((batch_size, self.hidden_size)) for _ in range(self.num_layers)]
        c = [np.zeros((batch_size, self.hidden_size)) for _ in range(self.num_layers)]
        
        outputs = []
        
        for t in range(seq_len):
            x_t = X[:, t, :]
            
            for layer_idx, layer in enumerate(self.layers):
                h[layer_idx], c[layer_idx] = layer.forward(x_t, h[layer_idx], c[layer_idx])
                x_t = h[layer_idx]
            
            out_t = h[-1] @ self.Wy + self.by
            outputs.append(out_t)
        
        return np.stack(outputs, axis=1)


# =============================================================================
# ATTENTION MECHANISM
# =============================================================================

class Attention:
    """
    Scaled Dot-Product Attention.
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """
    
    def __init__(self, d_model: int):
        self.d_model = d_model
        self.scale = np.sqrt(d_model)
    
    def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute attention.
        
        Args:
            Q: Queries (batch, seq_len, d_model)
            K: Keys (batch, seq_len, d_model)
            V: Values (batch, seq_len, d_model)
            mask: Optional attention mask
        
        Returns:
            Attention output (batch, seq_len, d_model)
        """
        # Compute attention scores
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + mask * -1e9
        
        # Softmax
        attention_weights = softmax(scores)
        
        # Apply attention to values
        output = np.matmul(attention_weights, V)
        
        return output, attention_weights


class MultiHeadAttention:
    """
    Multi-Head Attention.
    
    Allows model to attend to different representation subspaces.
    """
    
    def __init__(self, d_model: int, num_heads: int):
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        scale = np.sqrt(2.0 / d_model)
        self.Wq = np.random.randn(d_model, d_model) * scale
        self.Wk = np.random.randn(d_model, d_model) * scale
        self.Wv = np.random.randn(d_model, d_model) * scale
        self.Wo = np.random.randn(d_model, d_model) * scale
        
        self.attention = Attention(self.d_k)
    
    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """Split into multiple heads."""
        batch, seq_len, _ = x.shape
        x = x.reshape(batch, seq_len, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)  # (batch, heads, seq, d_k)
    
    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        """Merge heads back."""
        batch, _, seq_len, _ = x.shape
        x = x.transpose(0, 2, 1, 3)  # (batch, seq, heads, d_k)
        return x.reshape(batch, seq_len, self.d_model)
    
    def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass."""
        # Linear projections
        Q = Q @ self.Wq
        K = K @ self.Wk
        V = V @ self.Wv
        
        # Split heads
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)
        
        # Apply attention per head
        batch, heads, seq_len, d_k = Q.shape
        output = np.zeros_like(Q)
        
        for h in range(heads):
            out_h, _ = self.attention.forward(Q[:, h], K[:, h], V[:, h], mask)
            output[:, h] = out_h
        
        # Merge and project
        output = self._merge_heads(output)
        output = output @ self.Wo
        
        return output


# =============================================================================
# TRANSFORMER ENCODER
# =============================================================================

class TransformerBlock:
    """
    Single Transformer encoder block.
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        self.d_model = d_model
        
        # Multi-head attention
        self.mha = MultiHeadAttention(d_model, num_heads)
        
        # Feed-forward network
        scale = np.sqrt(2.0 / d_model)
        self.W1 = np.random.randn(d_model, d_ff) * scale
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * scale
        self.b2 = np.zeros(d_model)
        
        # Layer norm parameters
        self.gamma1 = np.ones(d_model)
        self.beta1 = np.zeros(d_model)
        self.gamma2 = np.ones(d_model)
        self.beta2 = np.zeros(d_model)
        
        self.dropout = dropout
    
    def _layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Layer normalization."""
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True) + 1e-6
        return gamma * (x - mean) / std + beta
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass."""
        # Self-attention with residual
        attn_out = self.mha.forward(x, x, x, mask)
        x = self._layer_norm(x + attn_out, self.gamma1, self.beta1)
        
        # Feed-forward with residual
        ff_out = relu(x @ self.W1 + self.b1) @ self.W2 + self.b2
        x = self._layer_norm(x + ff_out, self.gamma2, self.beta2)
        
        return x


class TransformerEncoder:
    """
    Transformer Encoder for time series.
    """
    
    def __init__(self, input_size: int, d_model: int, num_heads: int,
                 num_layers: int, d_ff: int, output_size: int,
                 max_seq_len: int = 512):
        self.d_model = d_model
        
        # Input projection
        self.input_proj = np.random.randn(input_size, d_model) * np.sqrt(2.0 / input_size)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(max_seq_len, d_model)
        
        # Transformer blocks
        self.blocks = [
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ]
        
        # Output projection
        self.output_proj = np.random.randn(d_model, output_size) * np.sqrt(2.0 / d_model)
        self.output_bias = np.zeros(output_size)
    
    def _create_positional_encoding(self, max_len: int, d_model: int) -> np.ndarray:
        """Create sinusoidal positional encoding."""
        pos = np.arange(max_len)[:, np.newaxis]
        div = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div)
        
        return pe
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            x: Input (batch, seq_len, input_size)
        
        Returns:
            output: (batch, output_size)
        """
        batch, seq_len, _ = x.shape
        
        # Project input and add positional encoding
        x = x @ self.input_proj + self.pos_encoding[:seq_len]
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block.forward(x)
        
        # Global average pooling and output projection
        x = x.mean(axis=1)
        output = x @ self.output_proj + self.output_bias
        
        return output


# =============================================================================
# AUTOENCODER
# =============================================================================

class Autoencoder:
    """
    Autoencoder for anomaly detection.
    
    Learns compressed representation of normal data.
    High reconstruction error indicates anomaly.
    """
    
    def __init__(self, input_size: int, hidden_sizes: List[int],
                 latent_size: int, activation: str = 'relu'):
        self.input_size = input_size
        self.latent_size = latent_size
        
        # Build encoder
        self.encoder_weights = []
        self.encoder_biases = []
        
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.encoder_weights.append(
                np.random.randn(prev_size, hidden_size) * np.sqrt(2.0 / prev_size)
            )
            self.encoder_biases.append(np.zeros(hidden_size))
            prev_size = hidden_size
        
        # Latent layer
        self.encoder_weights.append(
            np.random.randn(prev_size, latent_size) * np.sqrt(2.0 / prev_size)
        )
        self.encoder_biases.append(np.zeros(latent_size))
        
        # Build decoder (mirror of encoder)
        self.decoder_weights = []
        self.decoder_biases = []
        
        prev_size = latent_size
        for hidden_size in reversed(hidden_sizes):
            self.decoder_weights.append(
                np.random.randn(prev_size, hidden_size) * np.sqrt(2.0 / prev_size)
            )
            self.decoder_biases.append(np.zeros(hidden_size))
            prev_size = hidden_size
        
        # Output layer
        self.decoder_weights.append(
            np.random.randn(prev_size, input_size) * np.sqrt(2.0 / prev_size)
        )
        self.decoder_biases.append(np.zeros(input_size))
        
        self.activation = relu if activation == 'relu' else tanh
    
    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode input to latent space."""
        for W, b in zip(self.encoder_weights[:-1], self.encoder_biases[:-1]):
            x = self.activation(x @ W + b)
        
        # Latent layer (no activation)
        x = x @ self.encoder_weights[-1] + self.encoder_biases[-1]
        return x
    
    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode from latent space."""
        x = z
        for W, b in zip(self.decoder_weights[:-1], self.decoder_biases[:-1]):
            x = self.activation(x @ W + b)
        
        # Output layer (no activation for reconstruction)
        x = x @ self.decoder_weights[-1] + self.decoder_biases[-1]
        return x
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass (encode then decode)."""
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z
    
    def reconstruction_error(self, x: np.ndarray) -> np.ndarray:
        """Compute reconstruction error (for anomaly detection)."""
        x_reconstructed, _ = self.forward(x)
        return np.mean((x - x_reconstructed) ** 2, axis=-1)
    
    def detect_anomalies(self, x: np.ndarray, threshold: float) -> np.ndarray:
        """
        Detect anomalies based on reconstruction error.
        
        Args:
            x: Input data
            threshold: Error threshold (e.g., mean + 2*std from training)
        
        Returns:
            Boolean array (True = anomaly)
        """
        errors = self.reconstruction_error(x)
        return errors > threshold


# =============================================================================
# MODEL WRAPPER FOR TRADING
# =============================================================================

class DeepTradingModel:
    """
    Unified interface for deep learning models in trading.
    """
    
    def __init__(self, model_type: str = 'lstm', **kwargs):
        """
        Initialize model.
        
        Args:
            model_type: 'lstm', 'gru', 'transformer', or 'autoencoder'
            **kwargs: Model-specific parameters
        """
        self.model_type = model_type
        
        if model_type == 'lstm':
            self.model = LSTMNetwork(
                input_size=kwargs.get('input_size', 10),
                hidden_size=kwargs.get('hidden_size', 64),
                output_size=kwargs.get('output_size', 2),
                num_layers=kwargs.get('num_layers', 2),
                dropout=kwargs.get('dropout', 0.1)
            )
        elif model_type == 'transformer':
            self.model = TransformerEncoder(
                input_size=kwargs.get('input_size', 10),
                d_model=kwargs.get('d_model', 64),
                num_heads=kwargs.get('num_heads', 4),
                num_layers=kwargs.get('num_layers', 2),
                d_ff=kwargs.get('d_ff', 128),
                output_size=kwargs.get('output_size', 2)
            )
        elif model_type == 'autoencoder':
            self.model = Autoencoder(
                input_size=kwargs.get('input_size', 10),
                hidden_sizes=kwargs.get('hidden_sizes', [32, 16]),
                latent_size=kwargs.get('latent_size', 8)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features (batch, seq_len, features) or (batch, features)
        
        Returns:
            Predictions
        """
        if self.model_type in ['lstm', 'transformer']:
            if X.ndim == 2:
                X = X.reshape(X.shape[0], 1, X.shape[1])
            output, _ = self.model.forward(X) if hasattr(self.model, 'forward') else (self.model.forward(X), None)
            return output if isinstance(output, np.ndarray) else output[0]
        else:
            return self.model.forward(X)[0]
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get class probabilities."""
        logits = self.predict(X)
        return softmax(logits)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("DEEP LEARNING MODELS FOR TRADING")
    print("="*70)
    
    np.random.seed(42)
    
    # Generate sample sequence data
    batch_size = 32
    seq_len = 20
    input_size = 10
    
    X = np.random.randn(batch_size, seq_len, input_size)
    
    # 1. LSTM
    print("\n1. LSTM Network")
    print("-" * 40)
    lstm = LSTMNetwork(input_size=input_size, hidden_size=64, output_size=2)
    output, hidden_states = lstm.forward(X)
    print(f"   Input shape: {X.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Hidden states: {len(hidden_states)} timesteps")
    
    # Sequence prediction
    seq_output = lstm.predict_sequence(X)
    print(f"   Sequence output shape: {seq_output.shape}")
    
    # 2. Transformer
    print("\n2. Transformer Encoder")
    print("-" * 40)
    transformer = TransformerEncoder(
        input_size=input_size,
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=128,
        output_size=2
    )
    output = transformer.forward(X)
    print(f"   Input shape: {X.shape}")
    print(f"   Output shape: {output.shape}")
    
    # 3. Autoencoder
    print("\n3. Autoencoder (Anomaly Detection)")
    print("-" * 40)
    ae = Autoencoder(
        input_size=input_size,
        hidden_sizes=[32, 16],
        latent_size=8
    )
    
    X_flat = X[:, -1, :]  # Use last timestep
    reconstructed, latent = ae.forward(X_flat)
    errors = ae.reconstruction_error(X_flat)
    
    print(f"   Input shape: {X_flat.shape}")
    print(f"   Latent shape: {latent.shape}")
    print(f"   Reconstruction shape: {reconstructed.shape}")
    print(f"   Mean reconstruction error: {errors.mean():.4f}")
    
    # Anomaly detection
    threshold = errors.mean() + 2 * errors.std()
    anomalies = ae.detect_anomalies(X_flat, threshold)
    print(f"   Anomaly threshold: {threshold:.4f}")
    print(f"   Anomalies detected: {anomalies.sum()} / {len(anomalies)}")
    
    # 4. Unified Wrapper
    print("\n4. DeepTradingModel Wrapper")
    print("-" * 40)
    model = DeepTradingModel(
        model_type='lstm',
        input_size=input_size,
        hidden_size=64,
        output_size=2
    )
    
    proba = model.predict_proba(X)
    print(f"   Prediction probabilities shape: {proba.shape}")
    print(f"   Sample prediction: {proba[0]}")
    
    print("\n" + "="*70)
    print("PRODUCTION NOTES")
    print("="*70)
    print("""
This is a NumPy-only educational implementation.
For production trading, use:

1. PyTorch or TensorFlow for:
   - GPU acceleration (100x+ speedup)
   - Automatic differentiation (backprop)
   - Optimizers (Adam, SGD)
   - Pre-built layers and models

2. Key architectures for trading:
   - LSTM/GRU: Sequential patterns, regime persistence
   - Transformer: Long-range dependencies, attention to key events
   - Autoencoder: Anomaly detection, regime change detection
   - TCN: Causal convolutions, faster than LSTM

3. Common pitfalls:
   - Overfitting (use dropout, early stopping)
   - Look-ahead bias (causal architecture)
   - Non-stationarity (normalize features)
   - Sample weighting (recent data more relevant)

4. Installation for production:
   pip install torch  # or tensorflow
   pip install pytorch-lightning  # easier training
""")
