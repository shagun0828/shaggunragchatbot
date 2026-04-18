"""
Fixed-size chunker for text processing
Splits text into fixed-size chunks with overlap
"""

from typing import List

from models.chunk import Chunk


class FixedSizeChunker:
    """Chunks text into fixed-size segments with overlap"""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_fixed(self, text: str) -> List[Chunk]:
        """Split text into fixed-size chunks with overlap"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            if chunk_words:
                chunk_text = " ".join(chunk_words)
                chunk = self._create_chunk(chunk_text, start_pos=i)
                chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(self, text: str, start_pos: int = 0) -> Chunk:
        """Create a chunk with metadata"""
        metadata = {
            'chunk_type': 'fixed_size',
            'text_length': len(text),
            'start_position': start_pos,
            'chunk_size': self.chunk_size,
            'overlap': self.overlap
        }
        return Chunk(text=text, metadata=metadata)
