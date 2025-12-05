import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
import json
import os

class ErrorMemory:
    """
    Stores 'Bad Contexts' (feature vectors of failed trades) and checks 
    if current market conditions resemble past failures.
    """
    
    def __init__(self, memory_file: str = "data/error_memory.json"):
        self.memory_file = memory_file
        self.memory: List[Dict] = []
        self.similarity_threshold = 0.95  # High similarity required to block
        self._load_memory()
        
    def _load_memory(self):
        """Load memory from disk."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    self.memory = json.load(f)
            except Exception as e:
                print(f"Failed to load error memory: {e}")
                self.memory = []
                
    def _save_memory(self):
        """Save memory to disk."""
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self.memory, f, indent=2)
        except Exception as e:
            print(f"Failed to save error memory: {e}")

    def log_failure(self, context_vector: List[float], reason: str, symbol: str):
        """
        Log a failed trade context.
        
        Args:
            context_vector: List of normalized feature values [volatility, rsi, spread, etc.]
            reason: Why it failed (e.g., 'stop_loss', 'liquidation')
            symbol: Trading pair
        """
        # Normalize vector to unit length for cosine similarity
        vec = np.array(context_vector)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
            
        entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "vector": vec.tolist(),
            "reason": reason
        }
        self.memory.append(entry)
        self._save_memory()
        
    def is_risky(self, current_context: List[float]) -> bool:
        """
        Check if current context is similar to past failures.
        
        Returns: True if risky (similar to failure), False otherwise.
        """
        if not self.memory:
            return False
            
        # Normalize current vector
        curr_vec = np.array(current_context)
        norm = np.linalg.norm(curr_vec)
        if norm == 0:
            return False
        curr_vec = curr_vec / norm
        
        # Check similarity with recent failures (last 100)
        # Using simple cosine similarity
        recent_memories = self.memory[-100:]
        
        for mem in recent_memories:
            past_vec = np.array(mem['vector'])
            similarity = np.dot(curr_vec, past_vec)
            
            if similarity > self.similarity_threshold:
                return True
                
        return False
