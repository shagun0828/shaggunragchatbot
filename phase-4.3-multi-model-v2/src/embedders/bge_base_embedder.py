"""
BGE-base Embedder for Phase 4.3
High-quality embedding generation for up to 20 URLs
Uses bge-base-en-v1.5 with 768 dimensions for superior performance
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


class EmbeddingQuality(Enum):
    """Embedding quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"


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
    quality_distribution: Dict[str, int] = field(default_factory=dict)
    enhancement_applied: bool = False


class BGEBaseEmbedder:
    """High-quality BGE-base embedder for complex financial data"""
    
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
        
        # Financial domain enhancement
        self.financial_vocabulary = self._load_financial_vocabulary()
        self.domain_weights = self._calculate_domain_weights()
        self.context_categories = self._load_context_categories()
        
        # Quality assessment
        self.quality_checker = self._initialize_quality_checker()
        
        self.logger.info(f"BGE-base embedder initialized: {self.dimension} dimensions, max_urls: {self.max_urls}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'max_urls': 20,
            'batch_size': 16,
            'quality_threshold': 0.8,
            'enhancement_enabled': True,
            'normalization_enabled': True,
            'cache_embeddings': True,
            'memory_limit_mb': 2048,
            'quality_assessment': True,
            'financial_enhancement': True
        }
    
    def _load_financial_vocabulary(self) -> List[str]:
        """Load comprehensive financial vocabulary"""
        return [
            # Mutual fund basics
            'mutual fund', 'nav', 'net asset value', 'aum', 'assets under management',
            'expense ratio', 'fund manager', 'inception date', 'category', 'risk level',
            'returns', 'absolute return', 'xirr', 'cagr', 'compounded annual growth return',
            
            # Performance metrics
            'benchmark', 'outperformance', 'underperformance', 'alpha', 'beta', 'sharpe ratio',
            'standard deviation', 'volatility', 'drawdown', 'maximum drawdown', 'risk-adjusted returns',
            'sortino ratio', 'information ratio', 'tracking error', 'upside capture ratio',
            
            # Portfolio management
            'portfolio', 'holdings', 'top holdings', 'allocation', 'sector allocation',
            'asset allocation', 'diversification', 'concentration', 'rebalancing', 'turnover',
            'portfolio turnover', 'style drift', 'active share', 'tracking error',
            
            # Investment types
            'equity', 'debt', 'hybrid', 'balanced', 'arbitrage', 'elss', 'tax saver',
            'mid cap', 'large cap', 'small cap', 'multi cap', 'focused', 'sector', 'thematic',
            'value', 'growth', 'blend', 'quality', 'momentum',
            
            # Market indicators
            'nifty', 'sensex', 'bse', 'nse', 'index', 'market cap', 'cr', 'lakh', 'crore',
            'billion', 'million', 'percentage', 'bps', 'basis points', 'yield', 'spread',
            
            # Company analysis
            'balance sheet', 'income statement', 'cash flow', 'profit', 'loss',
            'revenue', 'earnings', 'dividend', 'payout ratio', 'book value', 'face value',
            'earnings per share', 'price to earnings', 'price to book', 'return on equity',
            
            # Economic indicators
            'inflation', 'interest rate', 'repo rate', 'gdp', 'inflation rate',
            'monetary policy', 'fiscal policy', 'economic growth', 'recession', 'expansion',
            'consumer price index', 'wholesale price index', 'industrial production',
            
            # Risk terminology
            'systematic risk', 'unsystematic risk', 'market risk', 'credit risk',
            'liquidity risk', 'concentration risk', 'currency risk', 'inflation risk',
            'interest rate risk', 'default risk', 'counterparty risk', 'operational risk',
            
            # Regulatory terms
            'sebi', 'rbi', 'amfi', 'regulatory', 'compliance', 'kyc', 'aml',
            'regulation', 'guidelines', 'norms', 'circular', 'investor protection',
            
            # Financial statements
            'quarterly results', 'annual report', 'audited financials', 'consolidated',
            'standalone', 'profit and loss', 'balance sheet', 'cash flow statement',
            
            # Technical analysis
            'moving average', 'rsi', 'macd', 'bollinger bands', 'support', 'resistance',
            'trend', 'volume', 'volatility index', 'vix', 'technical indicators'
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
    
    def _load_context_categories(self) -> Dict[str, List[str]]:
        """Load context categories for financial content"""
        return {
            'performance': [
                'return', 'performance', 'growth', 'gain', 'profit', 'appreciation',
                'benchmark', 'outperform', 'underperform', 'alpha', 'beta', 'sharpe'
            ],
            'risk': [
                'risk', 'volatility', 'drawdown', 'loss', 'decline', 'downside',
                'beta', 'standard deviation', 'var', 'value at risk', 'uncertainty'
            ],
            'portfolio': [
                'portfolio', 'holding', 'allocation', 'diversification', 'rebalancing',
                'concentration', 'turnover', 'style', 'active', 'passive'
            ],
            'fundamentals': [
                'fundamental', 'intrinsic', 'value', 'earnings', 'revenue', 'profit',
                'margin', 'ratio', 'growth', 'book value', 'cash flow'
            ],
            'investment': [
                'invest', 'investment', 'sip', 'lumpsum', 'systematic', 'regular',
                'long term', 'short term', 'horizon', 'goal', 'objective'
            ],
            'market': [
                'market', 'index', 'nifty', 'sensex', 'stock', 'equity', 'debt',
                'commodity', 'currency', 'derivatives', 'futures', 'options'
            ]
        }
    
    def _initialize_quality_checker(self) -> Dict[str, Any]:
        """Initialize quality assessment parameters"""
        return {
            'length_thresholds': {
                'min': 50,
                'optimal_min': 100,
                'optimal_max': 800,
                'max': 2000
            },
            'financial_content_thresholds': {
                'min_terms': 2,
                'good_terms': 5,
                'excellent_terms': 8
            },
            'semantic_thresholds': {
                'min_coherence': 0.3,
                'good_coherence': 0.6,
                'excellent_coherence': 0.8
            }
        }
    
    async def process_urls(self, url_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process multiple URLs with BGE-base embedding"""
        start_time = time.time()
        
        if len(url_data) > self.max_urls:
            self.logger.warning(f"URL count ({len(url_data)}) exceeds BGE-base limit ({self.max_urls})")
            url_data = url_data[:self.max_urls]
        
        self.logger.info(f"Processing {len(url_data)} URLs with BGE-base embedder")
        
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
            'quality_distribution': self._calculate_quality_distribution(quality_scores),
            'embeddings': embeddings.tolist(),
            'chunks': [{'id': chunk.id, 'text': chunk.text, 'metadata': chunk.metadata} for chunk in all_chunks],
            'timestamp': time.time()
        }
        
        self.logger.info(f"BGE-base processing completed: {len(url_data)} URLs, {len(all_chunks)} chunks, "
                        f"avg quality: {result['avg_quality_score']:.3f}")
        
        return result
    
    def _extract_chunks_from_url_data(self, url_info: Dict[str, Any]) -> List[Chunk]:
        """Extract high-quality chunks from URL data"""
        url = url_info.get('url', '')
        content = url_info.get('content', '')
        metadata = url_info.get('metadata', {})
        
        # Advanced chunking for BGE-base
        chunks = []
        
        # Split by paragraphs first
        paragraphs = content.split('\n\n')
        
        for para_idx, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                continue
            
            # Further split by sentences within paragraphs
            sentences = paragraph.split('. ')
            
            current_chunk_sentences = []
            current_chunk_length = 0
            
            for sent_idx, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # Check if adding this sentence exceeds optimal size
                potential_length = current_chunk_length + len(sentence) + 2  # +2 for ". "
                
                if potential_length > self.config['quality_threshold']['length_thresholds']['optimal_max'] and current_chunk_sentences > 0:
                    # Create chunk from accumulated sentences
                    chunk_text = '. '.join(current_chunk_sentences) + '.'
                    
                    chunk = Chunk(
                        id=f"{url}_para{para_idx}_chunk{len(chunks)}",
                        text=chunk_text.strip(),
                        metadata={
                            **metadata,
                            'url': url,
                            'paragraph_index': para_idx,
                            'chunk_index': len(chunks),
                            'model': 'bge-base',
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
            
            # Add remaining sentences as last chunk
            if current_chunk_sentences:
                chunk_text = '. '.join(current_chunk_sentences) + '.'
                
                chunk = Chunk(
                    id=f"{url}_para{para_idx}_chunk{len(chunks)}",
                    text=chunk_text.strip(),
                    metadata={
                        **metadata,
                        'url': url,
                        'paragraph_index': para_idx,
                        'chunk_index': len(chunks),
                        'model': 'bge-base',
                        'chunk_length': len(chunk_text.strip()),
                        'sentence_count': len(current_chunk_sentences)
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
        if self.config['financial_enhancement']:
            enhanced_embeddings = self._apply_financial_enhancement(base_embeddings, chunks)
            enhancement_applied = True
        else:
            enhanced_embeddings = base_embeddings
            enhancement_applied = False
        
        # Enhancement metadata
        enhancement_metadata = {
            'base_embeddings_shape': base_embeddings.shape,
            'enhancement_enabled': self.config['financial_enhancement'],
            'enhancement_applied': enhancement_applied,
            'batch_size': self.batch_size,
            'normalization_enabled': self.config['normalization_enabled'],
            'financial_terms_detected': self._count_financial_terms(chunks),
            'domain_enhancement_applied': enhancement_applied,
            'vocabulary_size': len(self.financial_vocabulary),
            'context_categories_used': self._identify_context_categories(chunks)
        }
        
        return enhanced_embeddings, enhancement_metadata
    
    def _apply_financial_enhancement(self, embeddings: np.ndarray, chunks: List[Chunk]) -> np.ndarray:
        """Apply advanced financial domain enhancement to embeddings"""
        enhanced_embeddings = embeddings.copy()
        
        for i, chunk in enumerate(chunks):
            # Analyze chunk context
            context_analysis = self._analyze_chunk_context(chunk.text)
            
            # Calculate domain relevance score
            domain_relevance = self._calculate_domain_relevance(chunk.text, context_analysis)
            
            if domain_relevance > 0:
                # Apply multi-dimensional enhancement
                enhanced_embedding = self._apply_context_enhancement(
                    enhanced_embeddings[i], chunk.text, context_analysis, domain_relevance
                )
                enhanced_embeddings[i] = enhanced_embedding
            
            # Re-normalize
            norm = np.linalg.norm(enhanced_embeddings[i])
            if norm > 0:
                enhanced_embeddings[i] /= norm
        
        return enhanced_embeddings
    
    def _analyze_chunk_context(self, text: str) -> Dict[str, Any]:
        """Comprehensive context analysis for financial content"""
        text_lower = text.lower()
        
        context = {
            'primary_category': None,
            'category_scores': {},
            'financial_terms': [],
            'numeric_data': [],
            'sentiment': 'neutral',
            'complexity': 'medium',
            'context_density': 0.0,
            'semantic_coherence': 0.0
        }
        
        # Analyze category context
        category_scores = {}
        for category, keywords in self.context_categories.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            category_scores[category] = score
        
        context['category_scores'] = category_scores
        if category_scores:
            context['primary_category'] = max(category_scores, key=category_scores.get)
        
        # Extract financial terms
        context['financial_terms'] = [term for term in self.financial_vocabulary if term in text_lower]
        
        # Extract numeric data
        numeric_patterns = [
            r'\b\d+\.?\d*%\b',  # Percentages
            r'\b\d+\.?\d*\s*(?:cr|l|crore|lakh)\b',  # Indian currency
            r'\b(?:rs\.?|â¹â¹?)\s*\d+\.?\d*\b',  # Rupee amounts
            r'\b\d+\.?\d*\s*xirr\b',  # XIRR values
            r'\b\d+\.?\d*\s*years?\b',  # Time periods
            r'\b\d+\.\d+\s*pe\b',  # P/E ratios
            r'\b\d+\.\d*\s*pb\b',  # P/B ratios
        ]
        
        import re
        for pattern in numeric_patterns:
            matches = re.findall(pattern, text_lower)
            context['numeric_data'].extend(matches)
        
        # Calculate context density
        context['context_density'] = len(context['financial_terms']) / len(text.split()) if text.split() else 0
        
        # Analyze sentiment
        positive_words = ['growth', 'gain', 'profit', 'return', 'outperform', 'excellent', 'strong']
        negative_words = ['loss', 'decline', 'fall', 'risk', 'underperform', 'poor', 'weak']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            context['sentiment'] = 'positive'
        elif negative_count > positive_count:
            context['sentiment'] = 'negative'
        
        # Analyze complexity
        word_count = len(text.split())
        avg_word_length = sum(len(word) for word in text.split()) / word_count if word_count > 0 else 0
        
        if word_count < 10:
            context['complexity'] = 'low'
        elif word_count > 50 or avg_word_length > 8:
            context['complexity'] = 'high'
        else:
            context['complexity'] = 'medium'
        
        return context
    
    def _calculate_domain_relevance(self, text: str, context: Dict[str, Any]) -> float:
        """Calculate comprehensive domain relevance score"""
        relevance = 0.0
        
        # Financial terms relevance
        term_count = len(context['financial_terms'])
        relevance += min(term_count * 0.05, 0.4)
        
        # Category-specific relevance
        category_scores = context['category_scores']
        if category_scores:
            max_category_score = max(category_scores.values())
            relevance += min(max_category_score * 0.03, 0.3)
        
        # Numeric data relevance
        numeric_count = len(context['numeric_data'])
        relevance += min(numeric_count * 0.02, 0.2)
        
        # Context density relevance
        relevance += min(context['context_density'] * 10, 0.1)
        
        return min(relevance, 1.0)
    
    def _apply_context_enhancement(self, base_embedding: np.ndarray, text: str, 
                                context: Dict[str, Any], domain_relevance: float) -> np.ndarray:
        """Apply context-aware enhancement to base embedding"""
        enhanced_embedding = base_embedding.copy()
        
        # Category-specific enhancement
        primary_category = context['primary_category']
        if primary_category and primary_category in self.domain_weights:
            category_weight = self.domain_weights[primary_category]
            category_enhancement = self._create_category_vector(primary_category, text)
            enhanced_embedding += category_enhancement * category_weight * domain_relevance * 0.3
        
        # Financial terminology boost
        if context['financial_terms']:
            terminology_enhancement = self._create_terminology_vector(context['financial_terms'], text)
            enhanced_embedding += terminology_enhancement * domain_relevance * 0.2
        
        # Sentiment adjustment
        if context['sentiment'] != 'neutral':
            sentiment_adjustment = self._create_sentiment_vector(context['sentiment'], text)
            enhanced_embedding += sentiment_adjustment * domain_relevance * 0.1
        
        # Complexity adjustment
        if context['complexity'] == 'high':
            enhanced_embedding *= 1.05  # Slight boost for complex content
        elif context['complexity'] == 'low':
            enhanced_embedding *= 0.98  # Slight reduction for simple content
        
        return enhanced_embedding
    
    def _create_category_vector(self, category: str, text: str) -> np.ndarray:
        """Create category-specific enhancement vector"""
        category_keywords = self.context_categories.get(category, [])
        category_text = " ".join(category_keywords)
        
        try:
            category_embedding = self.model.encode([category_text])[0]
            return category_embedding
        except Exception:
            return np.zeros(self.dimension)
    
    def _create_terminology_vector(self, financial_terms: List[str], text: str) -> np.ndarray:
        """Create terminology-specific enhancement vector"""
        if not financial_terms:
            return np.zeros(self.dimension)
        
        term_text = " ".join(financial_terms)
        
        try:
            terminology_embedding = self.model.encode([term_text])[0]
            return terminology_embedding
        except Exception:
            return np.zeros(self.dimension)
    
    def _create_sentiment_vector(self, sentiment: str, text: str) -> np.ndarray:
        """Create sentiment-specific adjustment vector"""
        sentiment_words = {
            'positive': ['excellent', 'outstanding', 'superior', 'exceptional', 'strong', 'growth'],
            'negative': ['poor', 'weak', 'inadequate', 'subpar', 'disappointing', 'decline', 'loss']
        }
        
        if sentiment in sentiment_words:
            sentiment_text = " ".join(sentiment_words[sentiment])
            try:
                sentiment_embedding = self.model.encode([sentiment_text])[0]
                return sentiment_embedding * 0.05
            except Exception:
                pass
        
        return np.zeros(self.dimension)
    
    def _count_financial_terms(self, chunks: List[Chunk]) -> int:
        """Count total financial terms across all chunks"""
        total_terms = 0
        for chunk in chunks:
            text_lower = chunk.text.lower()
            total_terms += sum(1 for term in self.financial_vocabulary if term in text_lower)
        return total_terms
    
    def _identify_context_categories(self, chunks: List[Chunk]) -> List[str]:
        """Identify context categories used in chunks"""
        categories_used = set()
        
        for chunk in chunks:
            text_lower = chunk.text.lower()
            for category, keywords in self.context_categories.items():
                if any(keyword in text_lower for keyword in keywords):
                    categories_used.add(category)
        
        return list(categories_used)
    
    def _assess_embedding_quality(self, embeddings: np.ndarray, chunks: List[Chunk]) -> np.ndarray:
        """Comprehensive quality assessment for embeddings"""
        quality_scores = []
        
        for i, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
            score = 0.0
            
            # Length score
            length = len(chunk.text)
            length_thresholds = self.quality_checker['length_thresholds']
            if length_thresholds['optimal_min'] <= length <= length_thresholds['optimal_max']:
                score += 0.3
            elif length >= length_thresholds['min']:
                score += 0.2
            
            # Financial content score
            financial_score = self._calculate_financial_relevance(chunk.text, self._analyze_chunk_context(chunk.text))
            score += financial_score * 0.4
            
            # Embedding norm score
            norm = np.linalg.norm(embedding)
            if 0.9 <= norm <= 1.1:
                score += 0.2
            
            # Semantic coherence score
            coherence_score = self._calculate_semantic_coherence(chunk.text)
            score += coherence_score * 0.1
            
            quality_scores.append(score)
        
        return np.array(quality_scores)
    
    def _analyze_chunk_context(self, text: str) -> Dict[str, Any]:
        """Quick context analysis for quality assessment"""
        text_lower = text.lower()
        
        return {
            'financial_terms': [term for term in self.financial_vocabulary if term in text_lower],
            'numeric_data': len([word for word in text_lower if any(char.isdigit() for char in word)]),
            'complexity': 'medium'  # Simplified for quality assessment
        }
    
    def _calculate_semantic_coherence(self, text: str) -> float:
        """Calculate semantic coherence of text"""
        # Simplified coherence calculation based on sentence structure
        sentences = text.split('. ')
        if len(sentences) <= 1:
            return 0.5
        
        # Check for consistent financial terminology
        financial_words = 0
        total_words = 0
        
        for sentence in sentences:
            words = sentence.split()
            total_words += len(words)
            financial_words += sum(1 for word in words if any(term in word.lower() for term in self.financial_vocabulary[:20]))
        
        coherence = financial_words / total_words if total_words > 0 else 0
        return min(coherence * 2, 1.0)  # Normalize to 0-1 range
    
    def _calculate_quality_distribution(self, quality_scores: np.ndarray) -> Dict[str, int]:
        """Calculate distribution of quality scores"""
        distribution = {
            'excellent': 0,
            'good': 0,
            'acceptable': 0,
            'poor': 0
        }
        
        for score in quality_scores:
            if score >= 0.9:
                distribution['excellent'] += 1
            elif score >= 0.7:
                distribution['good'] += 1
            elif score >= 0.5:
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
        
        # Check if enhancement was applied
        if hasattr(self, '_last_enhancement_applied'):
            self.metrics.enhancement_applied = self._last_enhancement_applied
    
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
                'batch_count': self.metrics.batch_count,
                'quality_distribution': self.metrics.quality_distribution,
                'enhancement_applied': self.metrics.enhancement_applied
            },
            'configuration': self.config,
            'vocabulary_size': len(self.financial_vocabulary),
            'context_categories': list(self.context_categories.keys())
        }
    
    def reset_metrics(self) -> None:
        """Reset all metrics"""
        self.metrics = BGEBaseMetrics()
        self.logger.info("BGE-base metrics reset")
    
    async def generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        embedding = self.model.encode(
            [text],
            normalize_embeddings=self.config['normalization_enabled'],
            show_progress_bar=False
        )[0]
        
        # Apply enhancement if enabled
        if self.config['financial_enhancement']:
            chunk = Chunk(id="single", text=text, metadata={'model': 'bge-base'})
            context = self._analyze_chunk_context(text)
            domain_relevance = self._calculate_domain_relevance(text, context)
            
            if domain_relevance > 0:
                embedding = self._apply_context_enhancement(embedding, text, context, domain_relevance)
                
                # Re-normalize
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding /= norm
        
        return embedding
