"""
BGE-base Embedder for Phase 4.3
Handles embedding generation for up to 20 URLs with high-quality processing
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
class BGEBaseMetrics:
    """Metrics for BGE-base processing"""
    urls_processed: int = 0
    chunks_generated: int = 0
    embeddings_created: int = 0
    processing_time: float = 0.0
    avg_quality_score: float = 0.0
    throughput: float = 0.0
    memory_usage: float = 0.0
    batch_count: int = 0


class BGEBaseEmbedder:
    """BGE-base embedder for high-quality processing of up to 20 URLs"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        
        # Initialize BGE-base model
        self.model_name = "bge-base-en-v1.5"
        self.model = SentenceTransformer(self.model_name)
        self.dimension = 768  # BGE-base dimension
        
        # Processing parameters
        self.max_urls = self.config['max_urls']
        self.batch_size = self.config['batch_size']
        self.quality_threshold = self.config['quality_threshold']
        
        # Metrics tracking
        self.metrics = BGEBaseMetrics()
        self.processing_history = deque(maxlen=100)
        
        # Content enhancement
        self.financial_vocabulary = self._load_financial_vocabulary()
        self.domain_weights = self._calculate_domain_weights()
        
        self.logger.info(f"BGE-base embedder initialized with {self.dimension} dimensions")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for BGE-base"""
        return {
            'max_urls': 20,
            'batch_size': 16,  # Smaller batch size for larger embeddings
            'quality_threshold': 0.8,
            'enhancement_enabled': True,
            'normalization_enabled': True,
            'cache_embeddings': True,
            'memory_limit_mb': 2048
        }
    
    def _load_financial_vocabulary(self) -> List[str]:
        """Load comprehensive financial vocabulary for enhancement"""
        return [
            # Mutual fund terms
            'mutual fund', 'nav', 'net asset value', 'aum', 'assets under management',
            'expense ratio', 'fund manager', 'inception date', 'category', 'risk level',
            'returns', 'absolute return', 'xirr', 'cagr', 'compounded annual growth rate',
            
            # Performance metrics
            'benchmark', 'outperformance', 'underperformance', 'alpha', 'beta', 'sharpe ratio',
            'standard deviation', 'volatility', 'drawdown', 'maximum drawdown', 'risk-adjusted returns',
            
            # Portfolio terms
            'portfolio', 'holdings', 'top holdings', 'allocation', 'sector allocation',
            'asset allocation', 'diversification', 'concentration', 'rebalancing', 'turnover',
            
            # Investment types
            'equity', 'debt', 'hybrid', 'balanced', 'arbitrage', 'elss', 'tax saver',
            'mid cap', 'large cap', 'small cap', 'multi cap', 'focused', 'sector', 'thematic',
            
            # Market terms
            'nifty', 'sensex', 'bse', 'nse', 'index', 'market cap', 'cr', 'lakh', 'crore',
            'billion', 'million', 'percentage', 'bps', 'basis points', 'yield',
            
            # Company analysis
            'balance sheet', 'income statement', 'cash flow', 'profit', 'loss',
            'revenue', 'earnings', 'dividend', 'payout ratio', 'book value', 'face value',
            
            # Economic indicators
            'inflation', 'interest rate', 'repo rate', 'gdp', 'inflation rate',
            'monetary policy', 'fiscal policy', 'economic growth', 'recession', 'expansion',
            
            # Risk terminology
            'systematic risk', 'unsystematic risk', 'market risk', 'credit risk',
            'liquidity risk', 'concentration risk', 'currency risk', 'inflation risk',
            'interest rate risk', 'default risk'
        ]
    
    def _calculate_domain_weights(self) -> Dict[str, float]:
        """Calculate domain weights for enhancement"""
        return {
            'performance': 0.25,
            'risk': 0.20,
            'portfolio': 0.20,
            'fundamentals': 0.15,
            'investment': 0.10,
            'market': 0.10
        }
    
    async def process_urls(self, url_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process multiple URLs with BGE-base embedding"""
        start_time = time.time()
        
        if len(url_data) > self.max_urls:
            self.logger.warning(f"URL count ({len(url_data)}) exceeds BGE-base limit ({self.max_urls})")
            url_data = url_data[:self.max_urls]
        
        self.logger.info(f"Processing {len(url_data)} URLs with BGE-base")
        
        # Extract and process content
        all_chunks = []
        for url_info in url_data:
            chunks = self._extract_chunks_from_url_data(url_info)
            all_chunks.extend(chunks)
        
        # Generate embeddings
        embeddings, enhancement_metadata = await self._generate_enhanced_embeddings(all_chunks)
        
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
        
        self.logger.info(f"BGE-base processing completed: {len(url_data)} URLs, {len(all_chunks)} chunks, "
                        f"avg quality: {result['avg_quality_score']:.3f}")
        
        return result
    
    def _extract_chunks_from_url_data(self, url_info: Dict[str, Any]) -> List[Chunk]:
        """Extract chunks from URL data"""
        url = url_info.get('url', '')
        content = url_info.get('content', '')
        metadata = url_info.get('metadata', {})
        
        # Basic chunking
        sentences = content.split('. ')
        chunks = []
        
        current_chunk = ""
        for i, sentence in enumerate(sentences):
            if len(current_chunk + sentence) <= 800:  # Smaller chunks for BGE-base
                current_chunk += sentence + ". "
            else:
                if current_chunk.strip():
                    chunk = Chunk(
                        id=f"{url}_{i}",
                        text=current_chunk.strip(),
                        metadata={
                            **metadata,
                            'url': url,
                            'chunk_index': len(chunks),
                            'model': 'bge-base',
                            'chunk_length': len(current_chunk.strip())
                        }
                    )
                    chunks.append(chunk)
                current_chunk = sentence + ". "
        
        if current_chunk.strip():
            chunk = Chunk(
                id=f"{url}_{len(chunks)}",
                text=current_chunk.strip(),
                metadata={
                    **metadata,
                    'url': url,
                    'chunk_index': len(chunks),
                    'model': 'bge-base',
                    'chunk_length': len(current_chunk.strip())
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    async def _generate_enhanced_embeddings(self, chunks: List[Chunk]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate enhanced embeddings using BGE-base"""
        texts = [chunk.text for chunk in chunks]
        
        # Generate base embeddings
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self.model.encode(
                batch_texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.config['normalization_enabled'],
                show_progress_bar=False
            )
            embeddings.append(batch_embeddings)
        
        base_embeddings = np.vstack(embeddings)
        
        # Apply financial domain enhancement if enabled
        if self.config['enhancement_enabled']:
            enhanced_embeddings = self._apply_financial_enhancement(base_embeddings, chunks)
        else:
            enhanced_embeddings = base_embeddings
        
        # Enhancement metadata
        enhancement_metadata = {
            'base_embeddings_shape': base_embeddings.shape,
            'enhancement_enabled': self.config['enhancement_enabled'],
            'batch_size': self.batch_size,
            'normalization_enabled': self.config['normalization_enabled'],
            'financial_terms_detected': self._count_financial_terms(chunks),
            'domain_enhancement_applied': self.config['enhancement_enabled']
        }
        
        return enhanced_embeddings, enhancement_metadata
    
    def _apply_financial_enhancement(self, embeddings: np.ndarray, chunks: List[Chunk]) -> np.ndarray:
        """Apply financial domain enhancement to embeddings"""
        enhanced_embeddings = embeddings.copy()
        
        for i, chunk in enumerate(chunks):
            # Calculate financial relevance score
            financial_score = self._calculate_financial_relevance(chunk.text)
            
            if financial_score > 0:
                # Apply enhancement
                enhancement_factor = 1 + (financial_score * 0.2)  # Up to 20% enhancement
                enhanced_embeddings[i] *= enhancement_factor
                
                # Re-normalize
                norm = np.linalg.norm(enhanced_embeddings[i])
                if norm > 0:
                    enhanced_embeddings[i] /= norm
        
        return enhanced_embeddings
    
    def _calculate_financial_relevance(self, text: str) -> float:
        """Calculate financial domain relevance score"""
        text_lower = text.lower()
        
        # Count financial terms
        term_count = sum(1 for term in self.financial_vocabulary if term in text_lower)
        
        # Calculate relevance
        max_possible = len(self.financial_vocabulary)
        relevance = term_count / max_possible if max_possible > 0 else 0
        
        # Boost for key financial indicators
        key_indicators = ['%', 'nav', 'return', 'cr', 'lakh', 'crore', 'rs', 'â¹â¹']
        key_count = sum(1 for indicator in key_indicators if indicator in text_lower)
        relevance += min(key_count * 0.1, 0.3)
        
        return min(relevance, 1.0)
    
    def _count_financial_terms(self, chunks: List[Chunk]) -> int:
        """Count total financial terms across all chunks"""
        total_terms = 0
        for chunk in chunks:
            total_terms += self._calculate_financial_relevance(chunk.text) * len(self.financial_vocabulary)
        return int(total_terms)
    
    def _assess_embedding_quality(self, embeddings: np.ndarray, chunks: List[Chunk]) -> np.ndarray:
        """Assess quality of generated embeddings"""
        quality_scores = []
        
        for i, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
            score = 0.0
            
            # Length score
            if 50 <= len(chunk.text) <= 1000:
                score += 0.3
            
            # Financial content score
            financial_score = self._calculate_financial_relevance(chunk.text)
            score += financial_score * 0.4
            
            # Embedding norm score
            norm = np.linalg.norm(embedding)
            if 0.9 <= norm <= 1.1:  # Well-normalized
                score += 0.2
            
            # Text complexity score
            word_count = len(chunk.text.split())
            if word_count > 10:
                score += 0.1
            
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
        self.metrics = BGEBaseMetrics()
        self.processing_history.clear()
        self.logger.info("BGE-base metrics reset")
    
    async def generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        embedding = self.model.encode(
            [text],
            normalize_embeddings=self.config['normalization_enabled'],
            show_progress_bar=False
        )[0]
        
        # Apply enhancement if enabled
        if self.config['enhancement_enabled']:
            chunk = Chunk(id="single", text=text, metadata={'model': 'bge-base'})
            financial_score = self._calculate_financial_relevance(text)
            
            if financial_score > 0:
                enhancement_factor = 1 + (financial_score * 0.2)
                embedding *= enhancement_factor
                
                # Re-normalize
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding /= norm
        
        return embedding
