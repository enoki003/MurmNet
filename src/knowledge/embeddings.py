"""
Embedding generation for text documents.
Handles conversion of text to vector embeddings.
"""

from typing import List

import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.config import config


class EmbeddingModel:
    """
    Wrapper for embedding model.
    Converts text to dense vector representations.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
    ):
        """
        Initialize embedding model.
        
        Args:
            model_name: Name of the embedding model
            device: Device to run model on ('cuda', 'cpu', 'mps')
            batch_size: Batch size for encoding
        """
        self.model_name = model_name or config.model.embedding_model
        self.device = device or config.model.embedding_device.value
        self.batch_size = batch_size or config.model.embedding_batch_size
        
        # Check device availability
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"
        elif self.device == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS not available, falling back to CPU")
            self.device = "cpu"
        
        # Load model
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
        )
        
        # Get embedding dimension
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        logger.info(
            f"Loaded embedding model on {self.device} "
            f"(dimension: {self.dimension})"
        )
    
    def encode(
        self,
        texts: List[str],
        show_progress: bool = False,
    ) -> List[List[float]]:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of text strings to encode
            show_progress: Whether to show progress bar
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Encode in batches
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        
        # Convert to list of lists
        return embeddings.tolist()
    
    def encode_single(self, text: str) -> List[float]:
        """
        Encode a single text to embedding.
        
        Args:
            text: Text string to encode
            
        Returns:
            Embedding vector
        """
        embeddings = self.encode([text])
        return embeddings[0] if embeddings else []
    
    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.dimension


# Import Optional
from typing import Optional
