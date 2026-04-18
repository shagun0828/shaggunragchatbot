"""
Sentence Transformer embedder
Generates embeddings using sentence transformer models
"""

import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

from models.chunk import Chunk


class SentenceTransformerEmbedder:
    """Embedder using sentence transformer models"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.batch_size = 32
    
    def generate_embeddings(self, chunks: List[Chunk]) -> np.ndarray:
        """Generate embeddings for chunks using sentence transformers"""
        texts = [chunk.text for chunk in chunks]
        
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self.model.encode(
                batch_texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        embedding = self.model.encode(
            text,
            normalize_embeddings=True
        )
        return embedding
