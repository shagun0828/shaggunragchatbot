"""
BGE-small Embedder for Phase 4.3
Handles embedding generation for up to 5 URLs with fast processing
"""

import asyncio
import logging
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from collections import deque
import json

from models.chunk import Chunk


@dataclass
class BGESmallMetrics:
    """Metrics for BGE-small processing"""
    urls_processed: int = 0
    chunks_generated: int = 0
    embeddings_created: int = 0
    processing_time: float = 0.0
    avg_quality_score: float = 0.0
    throughput: float = 0.0
    memory_usage: float = 0.0
    batch_count: int = 0


class BGESmallEmbedder:
    """BGE-small embedder for fast processing of up to 5 URLs"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        
        # Initialize BGE-small model
        self.model_name = "bge-small-en-v1.5"
        self.model = SentenceTransformer(self.model_name)
        self.dimension = 384  # BGE-small dimension
        
        # Processing parameters
        self.max_urls = self.config['max_urls']
        self.batch_size = self.config['batch_size']
        self.quality_threshold = self.config['quality_threshold']
        
        # Metrics tracking
        self.metrics = BGESmallMetrics()
        self.processing_history = deque(maxlen=100)
        
        # Content enhancement (lightweight for speed)
        self.financial_keywords = self._load_financial_keywords()
        
        self.logger.info(f"BGE-small embedder initialized with {self.dimension} dimensions")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for BGE-small"""
        return {
            'max_urls': 5,
            'batch_size': 32,  # Larger batch size for faster processing
            'quality_threshold': 0.7,
            'enhancement_enabled': True,
            'normalization_enabled': True,
            'cache_embeddings': True,
            'memory_limit_mb': 1024,
            'fast_mode': True
        }
    
    def _load_financial_keywords(self) -> List[str]:
        """Load essential financial keywords for lightweight enhancement"""
        return [
            'nav', 'return', 'fund', 'aum', 'cr', 'lakh', 'crore', '%', 'rs', 'â¹â¹',
            'equity', 'debt', 'hybrid', 'risk', 'performance', 'holding', 'allocation'
        ]
    
    async def process_urls(self, url_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process multiple URLs with BGE-small embedding"""
        start_time = time.time()
        
        if len(url_data) > self.max_urls:
            self.logger.warning(f"URL count ({len(url_data)}) exceeds BGE-small limit ({self.max_urls})")
            url_data = url_data[:self.max_urls]
        
        self.logger.info(f"Processing {len(url_data)} URLs with BGE-small")
        
        # Extract and process content
        all_chunks = []
        for url_info in url_data:
            chunks = self._extract_chunks_from_url_data(url_info)
            all_chunks.extend(chunks)
        
        # Generate embeddings
        embeddings, enhancement_metadata = await self._generate_fast_embeddings(all_chunks)
        
        # Quality assessment
        quality_scores = self._assess_embedding_quality(embeddings, all_chunks)
        
        # Update metrics
        processing_time = time.time() - start_time
        self._update_metrics(len(url_data), len(all_chunks), len(embeddings), processing_time, quality_scores)
        
        # Create result
        result = {
            'model_used': self.model_name,
            'urls_processed': len(url_data),
            'chunks_generated': len(all_chunks),
            'embeddings_created': len(embeddings),
            'embedding_dimension': self.dimension,
            'processing_time': processing_time,
            'avg_quality_score': np.mean(quality_scores),
            'throughput': len(all_chunks) / processing_time,
            'enhancement_metadata': enhancement_metadata,
            'quality_scores': quality_scores.tolist(),
            'embeddings': embeddings.tolist(),
            'chunks': [{'id': chunk.id, 'text': chunk.text, 'metadata': chunk.metadata} for chunk in all_chunks],
            'timestamp': time.time()
        }
        
        self.logger.info(f"BGE-small processing completed: {len(url_data)} URLs, {len(all_chunks)} chunks, "
                        f"avg quality: {result['avg_quality_score']:.3f}")
        
        return result
    
    def _extract_chunks_from_url_data(self, url_info: Dict[str, Any]) -> List[Chunk]:
        """Extract chunks from URL data with fast processing"""
        url = url_info.get('url', '')
        content = url_info.get('content', '')
        metadata = url_info.get('metadata', {})
        
        # Fast chunking - larger chunks for speed
        chunks = []
        chunk_size = 1200  # Larger chunks for BGE-small
        
        words = content.split()
        current_chunk_words = []
        
        for i, word in enumerate(words):
            current_chunk_words.append(word)
            
            # Check chunk size
            current_text = ' '.join(current_chunk_words)
            if len(current_text) >= chunk_size or i == len(words) - 1:
                if current_text.strip():
                    chunk = Chunk(
                        id=f"{url}_{len(chunks)}",
                        text=current_text.strip(),
                        metadata={
                            **metadata,
                            'url': url,
                            'chunk_index': len(chunks),
                            'model': 'bge-small',
                            'chunk_length': len(current_text.strip()),
                            'word_count': len(current_chunk_words)
                        }
                    )
                    chunks.append(chunk)
                current_chunk_words = []
        
        return chunks
    
    async def _generate_fast_embeddings(self, chunks: List[Chunk]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate embeddings with fast processing"""
        texts = [chunk.text for chunk in chunks]
        
        # Fast batch processing
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.config['normalization_enabled'],
            show_progress_bar=False
        )
        
        # Apply lightweight enhancement if enabled
        if self.config['enhancement_enabled']:
            enhanced_embeddings = self._apply_lightweight_enhancement(embeddings, chunks)
        else:
            enhanced_embeddings = embeddings
        
        # Enhancement metadata
        enhancement_metadata = {
            'base_embeddings_shape': embeddings.shape,
            'enhancement_enabled': self.config['enhancement_enabled'],
            'batch_size': self.batch_size,
            'normalization_enabled': self.config['normalization_enabled'],
            'fast_mode': self.config['fast_mode'],
            'financial_keywords_detected': self._count_financial_keywords(chunks),
            'lightweight_enhancement_applied': self.config['enhancement_enabled']
        }
        
        return enhanced_embeddings, enhancement_metadata
    
    def _apply_lightweight_enhancement(self, embeddings: np.ndarray, chunks: List[Chunk]) -> np.ndarray:
        """Apply lightweight financial enhancement for speed"""
        enhanced_embeddings = embeddings.copy()
        
        for i, chunk in enumerate(chunks):
            # Quick financial relevance check
            financial_score = self._quick_financial_relevance(chunk.text)
            
            if financial_score > 0:
                # Lightweight enhancement
                enhancement_factor = 1 + (financial_score * 0.1)  # Up to 10% enhancement
                enhanced_embeddings[i] *= enhancement_factor
                
                # Re-normalize
                norm = np.linalg.norm(enhanced_embeddings[i])
                if norm > 0:
                    enhanced_embeddings[i] /= norm
        
        return enhanced_embeddings
    
    def _quick_financial_relevance(self, text: str) -> float:
        """Quick financial relevance calculation for speed"""
        text_lower = text.lower()
        
        # Count financial keywords
        keyword_count = sum(1 for keyword in self.financial_keywords if keyword in text_lower)
        
        # Quick relevance calculation
        relevance = min(keyword_count * 0.1, 0.5)  # Max 0.5 for speed
        
        return relevance
    
    def _count_financial_keywords(self, chunks: List[Chunk]) -> int:
        """Count financial keywords across all chunks"""
        total_keywords = 0
        for chunk in chunks:
            text_lower = chunk.text.lower()
            total_keywords += sum(1 for keyword in self.financial_keywords if keyword in text_lower)
        return total_keywords
    
    def _assess_embedding_quality(self, embeddings: np.ndarray, chunks: List[Chunk]) -> np.ndarray:
        """Fast quality assessment for embeddings"""
        quality_scores = []
        
        for i, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
            score = 0.0
            
            # Length score (relaxed for speed)
            if 30 <= len(chunk.text) <= 1500:
                score += 0.4
            
            # Financial content score (quick)
            financial_score = self._quick_financial_relevance(chunk.text)
            score += financial_score * 0.4
            
            # Embedding norm score
            norm = np.linalg.norm(embedding)
            if 0.8 <= norm <= 1.2:  # Relaxed range for speed
                score += 0.2
            
            quality_scores.append(score)
        
        return np.array(quality_scores)
    
    def _update_metrics(self, url_count: int, chunk_count: int, embedding_count: int, 
                       processing_time: float, quality_scores: np.ndarray) -> None:
        """Update processing metrics"""
        self.metrics.urls_processed += url_count
        self.metrics.chunks_generated += chunk_count
        self.metrics.embeddings_created += embedding_count
        self.metrics.processing_time += processing_time
        self.metrics.avg_quality_score = np.mean(quality_scores)
        self.metrics.throughput = chunk_count / processing_time if processing_time > 0 else 0
        self.metrics.batch_count += 1
        
        # Add to history
        self.processing_history.append({
            'timestamp': time.time(),
            'url_count': url_count,
            'chunk_count': chunk_count,
            'processing_time': processing_time,
            'avg_quality': np.mean(quality_scores)
        })
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return {
            'model_name': self.model_name,
            'dimension': self.dimension,
            'max_urls': self.max_urls,
            'current_metrics': {
                'urls_processed': self.metrics.urls_processed,
                'chunks_generated': self.metrics.chunks_generated,
                'embeddings_created': self.metrics.embeddings_created,
                'processing_time': self.metrics.processing_time,
                'avg_quality_score': self.metrics.avg_quality_score,
                'throughput': self.metrics.throughput,
                'batch_count': self.metrics.batch_count
            },
            'configuration': self.config,
            'processing_history_size': len(self.processing_history)
        }
    
    def reset_metrics(self) -> None:
        """Reset all metrics"""
        self.metrics = BGESmallMetrics()
        self.processing_history.clear()
        self.logger.info("BGE-small metrics reset")
    
    async def generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        embedding = self.model.encode(
            [text],
            normalize_embeddings=self.config['normalization_enabled'],
            show_progress_bar=False
        )[0]
        
        # Apply lightweight enhancement if enabled
        if self.config['enhancement_enabled']:
            financial_score = self._quick_financial_relevance(text)
            
            if financial_score > 0:
                enhancement_factor = 1 + (financial_score * 0.1)
                embedding *= enhancement_factor
                
                # Re-normalize
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding /= norm
        
        return embedding
    
    def compare_with_base(self, base_result: Dict[str, Any]) -> Dict[str, Any]:
        """Compare BGE-small results with BGE-base results"""
        small_metrics = self.get_metrics()['current_metrics']
        
        comparison = {
            'model_comparison': {
                'bge_small': {
                    'dimension': self.dimension,
                    'avg_quality': small_metrics['avg_quality_score'],
                    'throughput': small_metrics['throughput'],
                    'processing_time': small_metrics['processing_time']
                },
                'bge_base': {
                    'dimension': base_result['embedding_dimension'],
                    'avg_quality': base_result['avg_quality_score'],
                    'throughput': base_result['throughput'],
                    'processing_time': base_result['processing_time']
                }
            },
            'advantages': {
                'speed': small_metrics['throughput'] > base_result['throughput'],
                'memory_efficiency': self.dimension < base_result['embedding_dimension'],
                'cost_effectiveness': True  # Always true for local models
            },
            'tradeoffs': {
                'dimension_reduction': (base_result['embedding_dimension'] - self.dimension) / base_result['embedding_dimension'],
                'quality_difference': small_metrics['avg_quality_score'] - base_result['avg_quality_score']
            }
        }
        
        return comparison
