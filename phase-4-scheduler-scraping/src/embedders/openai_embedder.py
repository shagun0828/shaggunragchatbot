"""
OpenAI embedder
Generates embeddings using OpenAI's API
"""

import os
import numpy as np
from typing import List
import openai

from models.chunk import Chunk


class OpenAIEmbedder:
    """Embedder using OpenAI's API"""
    
    def __init__(self, model: str = "text-embedding-3-small"):
        self.client = openai.OpenAI()
        self.model = model
        self.dimension = 1536 if "small" in model else 3072
        self.max_tokens = 8191
        self.batch_size = 100
    
    def generate_embeddings(self, chunks: List[Chunk]) -> np.ndarray:
        """Generate embeddings using OpenAI API"""
        texts = [self._truncate_text(chunk.text) for chunk in chunks]
        
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch_texts
            )
            batch_embeddings = [data.embedding for data in response.data]
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    def _truncate_text(self, text: str) -> str:
        """Truncate text to fit within token limits"""
        # Simple token approximation (can be enhanced with tiktoken)
        tokens = text.split()
        if len(tokens) <= self.max_tokens:
            return text
        return " ".join(tokens[:self.max_tokens])
    
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        truncated_text = self._truncate_text(text)
        response = self.client.embeddings.create(
            model=self.model,
            input=[truncated_text]
        )
        return np.array(response.data[0].embedding)
