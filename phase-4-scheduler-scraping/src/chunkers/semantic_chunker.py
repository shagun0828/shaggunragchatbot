"""
Semantic chunker for text processing
Splits text based on semantic similarity
"""

import re
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from models.chunk import Chunk


class SemanticChunker:
    """Chunks text based on semantic similarity"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = 0.7
        self.max_chunk_size = 1000
        self.min_chunk_size = 100
    
    def chunk_semantic(self, text: str) -> List[Chunk]:
        """Split text based on semantic similarity"""
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [self._create_chunk(text)]
        
        embeddings = self.model.encode(sentences)
        
        chunks = []
        current_chunk_sentences = []
        current_chunk_embedding = None
        
        for i, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
            if not current_chunk_sentences:
                current_chunk_sentences.append(sentence)
                current_chunk_embedding = embedding
            else:
                similarity = cosine_similarity(
                    current_chunk_embedding.reshape(1, -1),
                    embedding.reshape(1, -1)
                )[0][0]
                
                combined_text = " ".join(current_chunk_sentences + [sentence])
                
                if similarity >= self.similarity_threshold and len(combined_text) <= self.max_chunk_size:
                    current_chunk_sentences.append(sentence)
                    # Update average embedding
                    current_chunk_embedding = self._average_embeddings(
                        current_chunk_sentences,
                        embeddings[i-len(current_chunk_sentences)+1:i+1]
                    )
                else:
                    # Create chunk from current sentences
                    chunk_text = " ".join(current_chunk_sentences)
                    if len(chunk_text) >= self.min_chunk_size:
                        chunks.append(self._create_chunk(chunk_text))
                    
                    # Start new chunk
                    current_chunk_sentences = [sentence]
                    current_chunk_embedding = embedding
        
        # Add the last chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(self._create_chunk(chunk_text))
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting - can be enhanced with NLTK
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _average_embeddings(self, sentences: List[str], embeddings: np.ndarray) -> np.ndarray:
        """Calculate average embedding for a group of sentences"""
        return np.mean(embeddings, axis=0)
    
    def _create_chunk(self, text: str, **metadata) -> Chunk:
        """Create a chunk with text and metadata"""
        chunk_metadata = {
            'chunk_type': 'semantic',
            'text_length': len(text),
            **metadata
        }
        return Chunk(text=text, metadata=chunk_metadata)
