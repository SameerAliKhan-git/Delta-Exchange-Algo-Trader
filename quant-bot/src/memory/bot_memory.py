import logging
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
import os

try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

class BotMemory:
    """
    Vector Memory System for the Trading Bot.
    Stores and recalls trading experiences (patterns, errors, regimes).
    """
    
    def __init__(self, persist_dir: str = "data/memory_db"):
        self.logger = logging.getLogger("BotMemory")
        self.persist_dir = persist_dir
        
        if not HAS_CHROMA:
            self.logger.warning("ChromaDB not found. Memory will be disabled.")
            self.client = None
            return
            
        # Initialize ChromaDB
        try:
            self.client = chromadb.PersistentClient(path=persist_dir)
            
            # Create collections
            self.patterns = self.client.get_or_create_collection(
                name="trading_patterns",
                metadata={"hnsw:space": "cosine"}
            )
            self.errors = self.client.get_or_create_collection(
                name="trading_errors",
                metadata={"hnsw:space": "cosine"}
            )
            self.regimes = self.client.get_or_create_collection(
                name="market_regimes",
                metadata={"hnsw:space": "cosine"}
            )
            self.logger.info(f"BotMemory initialized at {persist_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB: {e}")
            self.client = None

    def store_experience(self, 
                        features: List[float], 
                        outcome: float, 
                        context: Dict[str, Any], 
                        memory_type: str = "pattern"):
        """
        Store a trading experience.
        
        Args:
            features: Vector representation of market state
            outcome: PnL or success metric
            context: Metadata (symbol, strategy, timestamp)
            memory_type: 'pattern', 'error', or 'regime'
        """
        if not self.client:
            return
            
        try:
            collection = self._get_collection(memory_type)
            if not collection:
                return

            # Generate ID
            timestamp = datetime.now().isoformat()
            doc_id = f"{context.get('symbol', 'UNK')}_{timestamp}"
            
            # Prepare metadata
            metadata = context.copy()
            metadata['outcome'] = float(outcome)
            metadata['timestamp'] = timestamp
            
            # Add to DB
            collection.add(
                embeddings=[features],
                metadatas=[metadata],
                documents=[json.dumps(metadata)], # Store full context as doc
                ids=[doc_id]
            )
            self.logger.debug(f"Stored {memory_type}: {doc_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to store experience: {e}")

    def query_similar(self, 
                     features: List[float], 
                     memory_type: str = "pattern", 
                     n_results: int = 5) -> List[Dict]:
        """
        Find similar past experiences.
        """
        if not self.client:
            return []
            
        try:
            collection = self._get_collection(memory_type)
            if not collection:
                return []
                
            results = collection.query(
                query_embeddings=[features],
                n_results=n_results
            )
            
            # Parse results
            parsed_results = []
            if results['ids']:
                for i in range(len(results['ids'][0])):
                    parsed_results.append({
                        'id': results['ids'][0][i],
                        'distance': results['distances'][0][i] if 'distances' in results else 0,
                        'metadata': results['metadatas'][0][i],
                        'document': results['documents'][0][i]
                    })
            
            return parsed_results
            
        except Exception as e:
            self.logger.error(f"Failed to query memory: {e}")
            return []

    def _get_collection(self, memory_type: str):
        if memory_type == "pattern":
            return self.patterns
        elif memory_type == "error":
            return self.errors
        elif memory_type == "regime":
            return self.regimes
        else:
            self.logger.warning(f"Unknown memory type: {memory_type}")
            return None
