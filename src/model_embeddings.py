"""
Enhanced Embedding Models for Document QA

This module provides advanced embedding models for improved document retrieval and QA.
It replaces the previous model_training.py which was focused on churn prediction.
"""

import os
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

# For saving/loading embeddings
import pickle
import json

class EmbeddingModel:
    """
    Enhanced embedding model for document representation and retrieval
    """
    
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2", use_gpu=False):
        """
        Initialize the embedding model
        
        Args:
            model_name (str): Name of the embedding model to use
            use_gpu (bool): Whether to use GPU acceleration if available
        """
        self.model_name = model_name
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.model = None
        self.embeddings_cache = {}
        self.last_update = None
        
        # Load model only when needed to save memory
    
    def load_model(self):
        """
        Load the embedding model (lazy loading)
        """
        if self.model is not None:
            return
            
        try:
            from sentence_transformers import SentenceTransformer
            
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            if self.use_gpu:
                self.model = self.model.to(torch.device("cuda"))
                
            print(f"Embedding model loaded successfully")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            self.model = None
    
    def get_embeddings(self, texts: List[str], batch_size=32, show_progress=True) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts (List[str]): List of texts to embed
            batch_size (int): Batch size for processing
            show_progress (bool): Whether to show progress bar
            
        Returns:
            np.ndarray: Array of embeddings
        """
        # Load model if not already loaded
        self.load_model()
        
        if self.model is None:
            print("Warning: Embedding model not available, returning random embeddings")
            # Return random embeddings as fallback
            return np.random.rand(len(texts), 384)  # 384 is a common embedding size
        
        try:
            # Generate embeddings
            embeddings = self.model.encode(
                texts, 
                batch_size=batch_size,
                show_progress_bar=show_progress
            )
            
            # Update cache with new embeddings
            for i, text in enumerate(texts):
                # Use a hash of the text as key to save memory
                text_hash = hash(text)
                self.embeddings_cache[text_hash] = embeddings[i]
            
            self.last_update = datetime.now()
            
            return embeddings
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            # Return random embeddings as fallback
            return np.random.rand(len(texts), 384)
    
    def save_embeddings(self, cache_dir: str = "data/embeddings"):
        """
        Save embedded vectors to disk
        
        Args:
            cache_dir (str): Directory to save embeddings
        """
        if not self.embeddings_cache:
            print("No embeddings to save")
            return
        
        os.makedirs(cache_dir, exist_ok=True)
        
        # Save embeddings
        cache_path = Path(cache_dir) / f"embeddings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(self.embeddings_cache, f)
                
            # Save metadata
            metadata = {
                "model_name": self.model_name,
                "embedding_count": len(self.embeddings_cache),
                "created_at": datetime.now().isoformat(),
                "embedding_dim": next(iter(self.embeddings_cache.values())).shape[0] if self.embeddings_cache else None
            }
            
            with open(cache_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
                
            print(f"Saved {len(self.embeddings_cache)} embeddings to {cache_path}")
        except Exception as e:
            print(f"Error saving embeddings: {e}")
    
    def load_embeddings(self, cache_path: str):
        """
        Load embedded vectors from disk
        
        Args:
            cache_path (str): Path to the embeddings file
        """
        try:
            with open(cache_path, 'rb') as f:
                loaded_cache = pickle.load(f)
                
            # Update cache with loaded embeddings
            self.embeddings_cache.update(loaded_cache)
            
            print(f"Loaded {len(loaded_cache)} embeddings from {cache_path}")
        except Exception as e:
            print(f"Error loading embeddings: {e}")


class CrossEncoderModel:
    """
    Cross-encoder model for reranking retrieved passages
    """
    
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", use_gpu=False):
        """
        Initialize the cross-encoder model
        
        Args:
            model_name (str): Name of the cross-encoder model to use
            use_gpu (bool): Whether to use GPU acceleration if available
        """
        self.model_name = model_name
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.model = None
    
    def load_model(self):
        """
        Load the cross-encoder model (lazy loading)
        """
        if self.model is not None:
            return
            
        try:
            from sentence_transformers import CrossEncoder
            
            print(f"Loading cross-encoder model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            
            # CrossEncoder handles GPU internally
                
            print(f"Cross-encoder model loaded successfully")
        except Exception as e:
            print(f"Error loading cross-encoder model: {e}")
            self.model = None
    
    def rerank(self, query: str, passages: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Rerank passages based on relevance to query
        
        Args:
            query (str): Query string
            passages (List[str]): List of passages to rerank
            top_k (int): Number of top passages to return
            
        Returns:
            List[Tuple[str, float]]: List of (passage, score) tuples
        """
        # Load model if not already loaded
        self.load_model()
        
        if self.model is None or not passages:
            print("Warning: Cross-encoder model not available or no passages, skipping reranking")
            return [(passage, 0.5) for passage in passages[:top_k]]
        
        try:
            # Prepare passage pairs for reranking
            passage_pairs = [[query, passage] for passage in passages]
            
            # Score passages
            scores = self.model.predict(passage_pairs)
            
            # Create list of (passage, score) pairs
            passage_score_pairs = list(zip(passages, scores))
            
            # Sort by score descending and take top_k
            passage_score_pairs.sort(key=lambda x: x[1], reverse=True)
            
            return passage_score_pairs[:top_k]
        except Exception as e:
            print(f"Error reranking passages: {e}")
            return [(passage, 0.5) for passage in passages[:top_k]]
