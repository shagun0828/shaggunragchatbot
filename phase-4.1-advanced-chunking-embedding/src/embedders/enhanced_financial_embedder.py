"""
Enhanced Financial Domain Embedder
Advanced financial domain adaptation with context-aware enhancement
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

from models.chunk import Chunk


class EnhancedFinancialEmbedder:
    """Enhanced financial domain embedder with context-aware enhancement"""
    
    def __init__(self, base_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.base_model = SentenceTransformer(base_model)
        self.logger = logging.getLogger(__name__)
        
        # Financial vocabulary and terminology
        self.financial_vocabulary = self._load_comprehensive_financial_vocabulary()
        self.domain_weights = self._calculate_domain_weights()
        
        # Context categories
        self.context_categories = {
            'performance': ['return', 'nav', 'growth', 'performance', 'benchmark'],
            'risk': ['risk', 'volatility', 'beta', 'drawdown', 'loss'],
            'portfolio': ['holding', 'allocation', 'portfolio', 'sector', 'asset'],
            'fundamentals': ['aum', 'expense', 'manager', 'inception', 'category'],
            'investment': ['investment', 'sip', 'lumpsum', 'minimum', 'exit'],
            'tax': ['tax', 'elss', 'saving', 'deduction', 'benefit']
        }
        
        # Enhancement parameters
        self.enhancement_strength = 0.3
        self.context_awareness = True
        self.terminology_boost = 0.2
        
        # TF-IDF for lexical analysis
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words='english',
            ngram_range=(1, 3),
            vocabulary=self.financial_vocabulary
        )
    
    def enhance_financial_embeddings(self, chunks: List[Chunk], base_embeddings: np.ndarray) -> np.ndarray:
        """Enhance embeddings for financial domain with context awareness"""
        self.logger.info(f"Enhancing {len(chunks)} embeddings for financial domain")
        
        enhanced_embeddings = []
        
        for i, chunk in enumerate(chunks):
            base_embedding = base_embeddings[i]
            
            # Analyze chunk context
            context_analysis = self._analyze_chunk_context(chunk.text)
            
            # Calculate domain relevance score
            domain_relevance = self._calculate_domain_relevance(chunk.text, context_analysis)
            
            # Apply context-aware enhancement
            if domain_relevance > 0:
                enhanced_embedding = self._apply_context_enhancement(
                    base_embedding, chunk.text, context_analysis, domain_relevance
                )
            else:
                enhanced_embedding = base_embedding
            
            enhanced_embeddings.append(enhanced_embedding)
        
        return np.array(enhanced_embeddings)
    
    def _load_comprehensive_financial_vocabulary(self) -> List[str]:
        """Load comprehensive financial vocabulary"""
        vocabulary = [
            # Basic financial terms
            'nav', 'net asset value', 'aum', 'assets under management', 'expense ratio',
            'fund manager', 'inception date', 'category', 'risk level', 'returns',
            
            # Performance metrics
            'annual return', 'absolute return', 'xirr', 'cagr', 'compounded annual growth rate',
            'benchmark', 'outperformance', 'underperformance', 'alpha', 'beta', 'sharpe ratio',
            'standard deviation', 'volatility', 'drawdown', 'maximum drawdown',
            
            # Investment types
            'equity', 'debt', 'hybrid', 'balanced', 'arbitrage', 'elss', 'tax saver',
            'mid cap', 'large cap', 'small cap', 'multi cap', 'focused', 'sector',
            
            # Portfolio management
            'portfolio', 'holdings', 'top holdings', 'allocation', 'sector allocation',
            'asset allocation', 'diversification', 'concentration', 'rebalancing',
            
            # Investment methods
            'sip', 'systematic investment plan', 'lumpsum', 'swp', 'stp', 'switch',
            'redemption', 'purchase', 'entry load', 'exit load', 'lock-in period',
            
            # Tax and regulation
            'tax', 'tax saving', 'deduction', 'section 80c', 'long term capital gains',
            'short term capital gains', 'tds', 'pan', 'kyc', 'riskometer',
            
            # Market indicators
            'nifty', 'sensex', 'bse', 'nse', 'index', 'market cap', 'cr', 'lakh',
            'crore', 'billion', 'million', 'percentage', 'bps', 'basis points',
            
            # Fund analysis
            'rating', 'aum growth', 'investor count', 'average maturity', 'duration',
            'ytm', 'yield to maturity', 'credit rating', 'credit quality',
            
            # Economic indicators
            'inflation', 'interest rate', 'repo rate', 'gdp', 'inflation rate',
            'monetary policy', 'fiscal policy', 'economic growth',
            
            # Risk terminology
            'systematic risk', 'unsystematic risk', 'market risk', 'credit risk',
            'liquidity risk', 'concentration risk', 'currency risk', 'inflation risk',
            
            # Financial statements
            'balance sheet', 'income statement', 'cash flow', 'profit', 'loss',
            'revenue', 'earnings', 'dividend', 'yield', 'payout ratio',
            
            # Technical terms
            'correlation', 'covariance', 'variance', 'standard error', 'confidence interval',
            'regression', 'optimization', 'efficient frontier', 'modern portfolio theory'
        ]
        
        return vocabulary
    
    def _calculate_domain_weights(self) -> Dict[str, float]:
        """Calculate weights for different financial domains"""
        return {
            'performance': 0.25,
            'risk': 0.20,
            'portfolio': 0.20,
            'fundamentals': 0.15,
            'investment': 0.10,
            'tax': 0.10
        }
    
    def _analyze_chunk_context(self, text: str) -> Dict[str, Any]:
        """Analyze the context of a chunk"""
        context = {
            'primary_category': None,
            'category_scores': {},
            'financial_terms': [],
            'numeric_data': [],
            'sentiment': 'neutral',
            'complexity': 'medium'
        }
        
        text_lower = text.lower()
        
        # Analyze category context
        category_scores = {}
        for category, keywords in self.context_categories.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            category_scores[category] = score
        
        context['category_scores'] = category_scores
        context['primary_category'] = max(category_scores, key=category_scores.get) if category_scores else None
        
        # Extract financial terms
        context['financial_terms'] = [term for term in self.financial_vocabulary if term in text_lower]
        
        # Extract numeric data
        numeric_patterns = [
            r'\b\d+\.?\d*%\b',  # Percentages
            r'\b\d+\.?\d*\s*(?:cr|l|crore|lakh)\b',  # Indian currency
            r'\b(?:rs\.?|â¹â¹?)\s*\d+\.?\d*\b',  # Rupee amounts
            r'\b\d+\.\d+\s*xirr\b',  # XIRR values
            r'\b\d+\.\d+\s*years?\b',  # Time periods
        ]
        
        for pattern in numeric_patterns:
            matches = re.findall(pattern, text_lower)
            context['numeric_data'].extend(matches)
        
        # Analyze sentiment (simplified)
        positive_words = ['growth', 'gain', 'profit', 'return', 'outperform', 'excellent']
        negative_words = ['loss', 'decline', 'fall', 'risk', 'underperform', 'poor']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            context['sentiment'] = 'positive'
        elif negative_count > positive_count:
            context['sentiment'] = 'negative'
        
        # Analyze complexity
        complexity_indicators = [
            len(text.split()),  # Word count
            len(context['financial_terms']),  # Financial terms count
            len(context['numeric_data']),  # Numeric data count
        ]
        
        avg_complexity = sum(complexity_indicators) / len(complexity_indicators)
        
        if avg_complexity < 5:
            context['complexity'] = 'low'
        elif avg_complexity > 15:
            context['complexity'] = 'high'
        
        return context
    
    def _calculate_domain_relevance(self, text: str, context: Dict[str, Any]) -> float:
        """Calculate domain relevance score for a chunk"""
        relevance = 0.0
        
        # Base relevance from financial terms
        financial_term_count = len(context['financial_terms'])
        relevance += min(financial_term_count * 0.1, 0.5)
        
        # Category-specific relevance
        category_scores = context['category_scores']
        if category_scores:
            max_category_score = max(category_scores.values())
            relevance += min(max_category_score * 0.05, 0.3)
        
        # Numeric data relevance
        numeric_data_count = len(context['numeric_data'])
        relevance += min(numeric_data_count * 0.05, 0.2)
        
        return min(relevance, 1.0)
    
    def _apply_context_enhancement(self, base_embedding: np.ndarray, text: str, 
                                context: Dict[str, Any], domain_relevance: float) -> np.ndarray:
        """Apply context-aware enhancement to base embedding"""
        enhanced_embedding = base_embedding.copy()
        
        # Apply category-specific enhancement
        primary_category = context['primary_category']
        if primary_category and primary_category in self.domain_weights:
            category_weight = self.domain_weights[primary_category]
            category_enhancement = self._create_category_vector(primary_category, text)
            enhanced_embedding += category_enhancement * category_weight * self.enhancement_strength
        
        # Apply terminology boost
        if context['financial_terms']:
            terminology_enhancement = self._create_terminology_vector(context['financial_terms'], text)
            enhanced_embedding += terminology_enhancement * self.terminology_boost
        
        # Apply sentiment adjustment
        if context['sentiment'] != 'neutral':
            sentiment_adjustment = self._create_sentiment_vector(context['sentiment'], text)
            enhanced_embedding += sentiment_adjustment * 0.1
        
        # Apply complexity adjustment
        if context['complexity'] == 'high':
            enhanced_embedding *= 1.1  # Boost complex content
        elif context['complexity'] == 'low':
            enhanced_embedding *= 0.95  # Slightly reduce simple content
        
        # Normalize the enhanced embedding
        enhanced_embedding = enhanced_embedding / np.linalg.norm(enhanced_embedding)
        
        return enhanced_embedding
    
    def _create_category_vector(self, category: str, text: str) -> np.ndarray:
        """Create category-specific enhancement vector"""
        # Generate embedding for category-specific keywords
        category_keywords = self.context_categories[category]
        category_text = " ".join(category_keywords)
        
        try:
            category_embedding = self.base_model.encode([category_text])[0]
            return category_embedding
        except Exception:
            return np.zeros_like(self.base_model.encode(["test"])[0])
    
    def _create_terminology_vector(self, financial_terms: List[str], text: str) -> np.ndarray:
        """Create terminology-specific enhancement vector"""
        if not financial_terms:
            return np.zeros(self.base_model.get_sentence_embedding_dimension())
        
        # Create weighted text based on term frequency
        term_text = " ".join(financial_terms)
        
        try:
            terminology_embedding = self.base_model.encode([term_text])[0]
            return terminology_embedding
        except Exception:
            return np.zeros(self.base_model.get_sentence_embedding_dimension())
    
    def _create_sentiment_vector(self, sentiment: str, text: str) -> np.ndarray:
        """Create sentiment-specific adjustment vector"""
        sentiment_words = {
            'positive': ['excellent', 'outstanding', 'superior', 'exceptional', 'strong'],
            'negative': ['poor', 'weak', 'inadequate', 'subpar', 'disappointing']
        }
        
        if sentiment in sentiment_words:
            sentiment_text = " ".join(sentiment_words[sentiment])
            try:
                sentiment_embedding = self.base_model.encode([sentiment_text])[0]
                return sentiment_embedding * 0.1
            except Exception:
                pass
        
        return np.zeros(self.base_model.get_sentence_embedding_dimension())
    
    def batch_enhance_embeddings(self, chunks: List[Chunk], base_embeddings: np.ndarray, 
                               batch_size: int = 32) -> np.ndarray:
        """Batch enhance embeddings for better performance"""
        enhanced_embeddings = []
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_embeddings = base_embeddings[i:i + batch_size]
            
            batch_enhanced = self.enhance_financial_embeddings(batch_chunks, batch_embeddings)
            enhanced_embeddings.append(batch_enhanced)
        
        return np.vstack(enhanced_embeddings)
    
    def get_enhancement_statistics(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Get statistics about enhancement process"""
        stats = {
            'total_chunks': len(chunks),
            'context_distribution': {},
            'average_domain_relevance': 0.0,
            'financial_term_coverage': 0.0
        }
        
        context_counts = {}
        total_relevance = 0.0
        total_terms = 0
        
        for chunk in chunks:
            context = self._analyze_chunk_context(chunk.text)
            
            # Count contexts
            primary_category = context['primary_category']
            if primary_category:
                context_counts[primary_category] = context_counts.get(primary_category, 0) + 1
            
            # Calculate relevance
            relevance = self._calculate_domain_relevance(chunk.text, context)
            total_relevance += relevance
            
            # Count terms
            total_terms += len(context['financial_terms'])
        
        stats['context_distribution'] = context_counts
        stats['average_domain_relevance'] = total_relevance / len(chunks) if chunks else 0.0
        stats['financial_term_coverage'] = total_terms / len(chunks) if chunks else 0.0
        
        return stats
