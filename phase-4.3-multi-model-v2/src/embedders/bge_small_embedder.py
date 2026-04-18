"""
BGE-small Embedder for Phase 4.3
Fast embedding generation for up to 5 URLs
Uses bge-small-en-v1.5 with 384 dimensions for efficient processing
"""

import asyncio
import logging
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from sentence_transformers import SentenceTransformer

from models.chunk import Chunk


class ProcessingMode(Enum):
    """Processing modes for BGE-small"""
    FAST = "fast"
    BALANCED = "balanced"
    QUALITY = "quality"


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
    quality_distribution: Dict[str, int] = field(default_factory=dict)
    enhancement_applied: bool = False
    processing_mode: str = "fast"


class BGESmallEmbedder:
    """Fast BGE-small embedder for efficient processing"""
    
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
        self.processing_mode = ProcessingMode(self.config['processing_mode'])
        
        # Metrics tracking
        self.metrics = BGESmallMetrics()
        
        # Fast financial vocabulary (lightweight)
        self.financial_keywords = self._load_financial_keywords()
        
        # Quality assessment (simplified)
        self.quality_checker = self._initialize_quality_checker()
        
        self.logger.info(f"BGE-small embedder initialized: {self.dimension} dimensions, max_urls: {self.max_urls}, mode: {self.processing_mode.value}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'max_urls': 5,
            'batch_size': 32,
            'quality_threshold': 0.7,
            'enhancement_enabled': True,
            'normalization_enabled': True,
            'cache_embeddings': True,
            'memory_limit_mb': 1024,
            'processing_mode': 'fast',  # fast, balanced, quality
            'lightweight_enhancement': True,
            'chunk_size_target': 1200,  # Larger chunks for speed
            'quality_assessment': True
        }
    
    def _load_financial_keywords(self) -> List[str]:
        """Load essential financial keywords for lightweight enhancement"""
        return [
            # Core financial terms
            'nav', 'return', 'fund', 'aum', '%', 'cr', 'lakh', 'crore', 'rs', 'â¹â¹',
            'equity', 'debt', 'hybrid', 'risk', 'performance', 'holding', 'allocation',
            
            # Performance indicators
            'growth', 'gain', 'loss', 'profit', 'benchmark', 'alpha', 'beta',
            'volatility', 'drawdown', 'xirr', 'cagr',
            
            # Investment terms
            'sip', 'lumpsum', 'investment', 'portfolio', 'diversification',
            'expense', 'ratio', 'category', 'cap', 'mid', 'large', 'small',
            
            # Market terms
            'nifty', 'sensex', 'index', 'market', 'stock', 'share', 'price',
            'sector', 'industry', 'company', 'financial'
        ]
    
    def _initialize_quality_checker(self) -> Dict[str, Any]:
        """Initialize simplified quality assessment parameters"""
        return {
            'length_thresholds': {
                'min': 30,
                'optimal_min': 80,
                'optimal_max': 1500,
                'max': 3000
            },
            'keyword_thresholds': {
                'min_keywords': 1,
                'good_keywords': 3,
                'excellent_keywords': 5
            },
            'coherence_thresholds': {
                'min_coherence': 0.2,
                'good_coherence': 0.5,
                'excellent_coherence': 0.7
            }
        }
    
    async def process_urls(self, url_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process multiple URLs with BGE-small embedding"""
        start_time = time.time()
        
        if len(url_data) > self.max_urls:
            self.logger.warning(f"URL count ({len(url_data)}) exceeds BGE-small limit ({self.max_urls})")
            url_data = url_data[:self.max_urls]
        
        self.logger.info(f"Processing {len(url_data)} URLs with BGE-small embedder (mode: {self.processing_mode.value})")
        
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
            'processing_mode': self.processing_mode.value,
            'urls_processed': len(url_data),
            'chunks_generated': len(all_chunks),
            'embeddings_created': len(embeddings),
            'embedding_dimension': self.dimension,
            'processing_time': processing_time,
            'avg_quality_score': np.mean(quality_scores),
            'throughput': len(all_chunks) / processing_time,
            'enhancement_metadata': enhancement_metadata,
            'quality_scores': quality_scores.tolist(),
            'quality_distribution': self._calculate_quality_distribution(quality_scores),
            'embeddings': embeddings.tolist(),
            'chunks': [{'id': chunk.id, 'text': chunk.text, 'metadata': chunk.metadata} for chunk in all_chunks],
            'timestamp': time.time()
        }
        
        self.logger.info(f"BGE-small processing completed: {len(url_data)} URLs, {len(all_chunks)} chunks, "
                        f"avg quality: {result['avg_quality_score']:.3f}")
        
        return result
    
    def _extract_chunks_from_url_data(self, url_info: Dict[str, Any]) -> List[Chunk]:
        """Extract fast chunks from URL data"""
        url = url_info.get('url', '')
        content = url_info.get('content', '')
        metadata = url_info.get('metadata', {})
        
        chunks = []
        
        # Fast chunking strategy based on processing mode
        if self.processing_mode == ProcessingMode.FAST:
            chunks = self._fast_chunking(content, url, metadata)
        elif self.processing_mode == ProcessingMode.BALANCED:
            chunks = self._balanced_chunking(content, url, metadata)
        else:  # QUALITY
            chunks = self._quality_chunking(content, url, metadata)
        
        return chunks
    
    def _fast_chunking(self, content: str, url: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Fast chunking for speed optimization"""
        chunks = []
        chunk_size_target = self.config['chunk_size_target']
        
        # Simple word-based chunking
        words = content.split()
        current_chunk_words = []
        
        for i, word in enumerate(words):
            current_chunk_words.append(word)
            
            # Check chunk size
            current_text = ' '.join(current_chunk_words)
            if len(current_text) >= chunk_size_target or i == len(words) - 1:
                if current_text.strip():
                    chunk = Chunk(
                        id=f"{url}_fast_chunk_{len(chunks)}",
                        text=current_text.strip(),
                        metadata={
                            **metadata,
                            'url': url,
                            'chunk_index': len(chunks),
                            'model': 'bge-small',
                            'processing_mode': 'fast',
                            'chunk_length': len(current_text.strip()),
                            'word_count': len(current_chunk_words)
                        }
                    )
                    chunks.append(chunk)
                current_chunk_words = []
        
        return chunks
    
    def _balanced_chunking(self, content: str, url: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Balanced chunking for quality-speed balance"""
        chunks = []
        
        # Split by paragraphs first
        paragraphs = content.split('\n\n')
        
        for para_idx, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                continue
            
            # If paragraph is too long, split by sentences
            if len(paragraph) > self.config['chunk_size_target']:
                sentences = paragraph.split('. ')
                current_sentences = []
                
                for sentence in sentences:
                    current_sentences.append(sentence)
                    current_text = '. '.join(current_sentences) + '.'
                    
                    if len(current_text) >= self.config['chunk_size_target'] * 0.8:
                        if current_text.strip():
                            chunk = Chunk(
                                id=f"{url}_balanced_chunk_{len(chunks)}",
                                text=current_text.strip(),
                                metadata={
                                    **metadata,
                                    'url': url,
                                    'paragraph_index': para_idx,
                                    'chunk_index': len(chunks),
                                    'model': 'bge-small',
                                    'processing_mode': 'balanced',
                                    'chunk_length': len(current_text.strip())
                                }
                            )
                            chunks.append(chunk)
                        current_sentences = []
                
                # Add remaining sentences
                if current_sentences:
                    current_text = '. '.join(current_sentences) + '.'
                    if current_text.strip():
                        chunk = Chunk(
                            id=f"{url}_balanced_chunk_{len(chunks)}",
                            text=current_text.strip(),
                            metadata={
                                **metadata,
                                'url': url,
                                'paragraph_index': para_idx,
                                'chunk_index': len(chunks),
                                'model': 'bge-small',
                                'processing_mode': 'balanced',
                                'chunk_length': len(current_text.strip())
                            }
                        )
                        chunks.append(chunk)
            else:
                # Use paragraph as is
                chunk = Chunk(
                    id=f"{url}_balanced_chunk_{len(chunks)}",
                    text=paragraph.strip(),
                    metadata={
                        **metadata,
                        'url': url,
                        'paragraph_index': para_idx,
                        'chunk_index': len(chunks),
                        'model': 'bge-small',
                        'processing_mode': 'balanced',
                        'chunk_length': len(paragraph.strip())
                    }
                )
                chunks.append(chunk)
        
        return chunks
    
    def _quality_chunking(self, content: str, url: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Quality chunking for better semantic coherence"""
        chunks = []
        
        # Split by sentences and group by semantic similarity
        sentences = content.split('. ')
        current_chunk_sentences = []
        current_chunk_length = 0
        
        for sent_idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if adding this sentence would exceed optimal size
            potential_length = current_chunk_length + len(sentence) + 2
            
            if potential_length > self.config['chunk_size_target'] and current_chunk_sentences:
                # Create chunk
                chunk_text = '. '.join(current_chunk_sentences) + '.'
                
                chunk = Chunk(
                    id=f"{url}_quality_chunk_{len(chunks)}",
                    text=chunk_text.strip(),
                    metadata={
                        **metadata,
                        'url': url,
                        'chunk_index': len(chunks),
                        'model': 'bge-small',
                        'processing_mode': 'quality',
                        'chunk_length': len(chunk_text.strip()),
                        'sentence_count': len(current_chunk_sentences)
                    }
                )
                chunks.append(chunk)
                
                # Start new chunk
                current_chunk_sentences = [sentence]
                current_chunk_length = len(sentence)
            else:
                current_chunk_sentences.append(sentence)
                current_chunk_length = potential_length
        
        # Add remaining sentences
        if current_chunk_sentences:
            chunk_text = '. '.join(current_chunk_sentences) + '.'
            
            chunk = Chunk(
                id=f"{url}_quality_chunk_{len(chunks)}",
                text=chunk_text.strip(),
                metadata={
                    **metadata,
                    'url': url,
                    'chunk_index': len(chunks),
                    'model': 'bge-small',
                    'processing_mode': 'quality',
                    'chunk_length': len(chunk_text.strip()),
                    'sentence_count': len(current_chunk_sentences)
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    async def _generate_fast_embeddings(self, chunks: List[Chunk]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate embeddings with fast processing"""
        texts = [chunk.text for chunk in chunks]
        
        # Fast batch processing
        start_time = time.time()
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.config['normalization_enabled'],
            show_progress_bar=False
        )
        
        encoding_time = time.time() - start_time
        
        # Apply lightweight enhancement if enabled
        if self.config['lightweight_enhancement']:
            enhanced_embeddings = self._apply_lightweight_enhancement(embeddings, chunks)
            enhancement_applied = True
        else:
            enhanced_embeddings = embeddings
            enhancement_applied = False
        
        # Enhancement metadata
        enhancement_metadata = {
            'base_embeddings_shape': embeddings.shape,
            'enhancement_enabled': self.config['lightweight_enhancement'],
            'enhancement_applied': enhancement_applied,
            'batch_size': self.batch_size,
            'normalization_enabled': self.config['normalization_enabled'],
            'processing_mode': self.processing_mode.value,
            'encoding_time': encoding_time,
            'financial_keywords_detected': self._count_financial_keywords(chunks),
            'lightweight_enhancement_applied': enhancement_applied,
            'vocabulary_size': len(self.financial_keywords)
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
        relevance = min(keyword_count * 0.15, 0.6)  # Max 0.6 for speed
        
        # Boost for key financial indicators
        key_indicators = ['%', 'nav', 'return', 'cr', 'lakh', 'crore', 'rs', 'â¹â¹']
        key_count = sum(1 for indicator in key_indicators if indicator in text_lower)
        relevance += min(key_count * 0.05, 0.2)
        
        return min(relevance, 1.0)
    
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
            length = len(chunk.text)
            length_thresholds = self.quality_checker['length_thresholds']
            if length_thresholds['optimal_min'] <= length <= length_thresholds['optimal_max']:
                score += 0.4
            elif length >= length_thresholds['min']:
                score += 0.3
            
            # Financial content score (quick)
            financial_score = self._quick_financial_relevance(chunk.text)
            score += financial_score * 0.4
            
            # Embedding norm score
            norm = np.linalg.norm(embedding)
            if 0.8 <= norm <= 1.2:  # Relaxed range for speed
                score += 0.2
            
            quality_scores.append(score)
        
        return np.array(quality_scores)
    
    def _calculate_quality_distribution(self, quality_scores: np.ndarray) -> Dict[str, int]:
        """Calculate distribution of quality scores"""
        distribution = {
            'excellent': 0,
            'good': 0,
            'acceptable': 0,
            'poor': 0
        }
        
        for score in quality_scores:
            if score >= 0.85:
                distribution['excellent'] += 1
            elif score >= 0.65:
                distribution['good'] += 1
            elif score >= 0.45:
                distribution['acceptable'] += 1
            else:
                distribution['poor'] += 1
        
        return distribution
    
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
        self.metrics.quality_distribution = self._calculate_quality_distribution(quality_scores)
        self.metrics.processing_mode = self.processing_mode.value
        
        # Check if enhancement was applied
        if hasattr(self, '_last_enhancement_applied'):
            self.metrics.enhancement_applied = self._last_enhancement_applied
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return {
            'model_name': self.model_name,
            'dimension': self.dimension,
            'max_urls': self.max_urls,
            'processing_mode': self.processing_mode.value,
            'current_metrics': {
                'urls_processed': self.metrics.urls_processed,
                'chunks_generated': self.metrics.chunks_generated,
                'embeddings_created': self.metrics.embeddings_created,
                'processing_time': self.metrics.processing_time,
                'avg_quality_score': self.metrics.avg_quality_score,
                'throughput': self.metrics.throughput,
                'batch_count': self.metrics.batch_count,
                'quality_distribution': self.metrics.quality_distribution,
                'enhancement_applied': self.metrics.enhancement_applied
            },
            'configuration': self.config,
            'vocabulary_size': len(self.financial_keywords)
        }
    
    def reset_metrics(self) -> None:
        """Reset all metrics"""
        self.metrics = BGESmallMetrics()
        self.logger.info("BGE-small metrics reset")
    
    def set_processing_mode(self, mode: ProcessingMode) -> None:
        """Set processing mode"""
        self.processing_mode = mode
        self.config['processing_mode'] = mode.value
        self.logger.info(f"BGE-small processing mode set to: {mode.value}")
    
    async def generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        embedding = self.model.encode(
            [text],
            normalize_embeddings=self.config['normalization_enabled'],
            show_progress_bar=False
        )[0]
        
        # Apply lightweight enhancement if enabled
        if self.config['lightweight_enhancement']:
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
                    'processing_time': small_metrics['processing_time'],
                    'processing_mode': small_metrics['processing_mode']
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
                'cost_effectiveness': True,  # Always true for local models
                'processing_mode_flexibility': True
            },
            'tradeoffs': {
                'dimension_reduction': (base_result['embedding_dimension'] - self.dimension) / base_result['embedding_dimension'],
                'quality_difference': small_metrics['avg_quality_score'] - base_result['avg_quality_score'],
                'processing_mode_impact': self.processing_mode.value
            },
            'recommendations': self._generate_mode_recommendations(small_metrics, base_result)
        }
        
        return comparison
    
    def _generate_mode_recommendations(self, small_metrics: Dict, base_result: Dict) -> List[str]:
        """Generate processing mode recommendations"""
        recommendations = []
        
        throughput_ratio = small_metrics['throughput'] / base_result['throughput']
        quality_ratio = small_metrics['avg_quality_score'] / base_result['avg_quality_score']
        
        if self.processing_mode == ProcessingMode.FAST:
            recommendations.append("Fast mode provides highest throughput for time-sensitive applications")
            if quality_ratio < 0.9:
                recommendations.append("Consider switching to balanced mode if quality is more important")
        elif self.processing_mode == ProcessingMode.BALANCED:
            recommendations.append("Balanced mode provides good trade-off between speed and quality")
            if throughput_ratio > 1.2:
                recommendations.append("Good performance advantage over BGE-base")
        else:  # QUALITY
            recommendations.append("Quality mode provides best possible quality with BGE-small")
            if quality_ratio > 0.95:
                recommendations.append("Quality is comparable to BGE-base with better efficiency")
        
        return recommendations
