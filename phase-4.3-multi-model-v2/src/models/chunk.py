"""
Chunk model for Phase 4.3
Represents a text chunk with comprehensive metadata
"""

import uuid
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class Chunk:
    """Represents a text chunk with comprehensive metadata"""
    id: str
    text: str
    metadata: Dict[str, Any]
    created_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Post-initialization processing"""
        if not self.id:
            self.id = str(uuid.uuid4())
        
        # Ensure created_at is set
        if self.created_at == 0:
            self.created_at = time.time()
        
        # Add computed metadata
        self._add_computed_metadata()
    
    def _add_computed_metadata(self):
        """Add computed metadata fields"""
        self.metadata.update({
            'text_length': len(self.text),
            'word_count': len(self.text.split()),
            'sentence_count': len(self.text.split('. ')),
            'has_financial_content': self.has_financial_content(),
            'domain_relevance': self.get_domain_relevance(),
            'complexity_score': self._calculate_complexity_score()
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary"""
        return {
            'id': self.id,
            'text': self.text,
            'metadata': self.metadata,
            'created_at': self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Chunk':
        """Create chunk from dictionary"""
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            text=data.get('text', ''),
            metadata=data.get('metadata', {}),
            created_at=data.get('created_at', time.time())
        )
    
    def get_text_length(self) -> int:
        """Get text length"""
        return len(self.text)
    
    def get_word_count(self) -> int:
        """Get word count"""
        return len(self.text.split())
    
    def get_sentence_count(self) -> int:
        """Get sentence count"""
        return len(self.text.split('. '))
    
    def has_financial_content(self) -> bool:
        """Check if chunk contains financial content"""
        financial_keywords = [
            'nav', 'return', 'fund', 'aum', '%', 'cr', 'lakh', 'crore', 'rs', 'â¹â¹',
            'equity', 'debt', 'hybrid', 'risk', 'performance', 'holding', 'allocation',
            'investment', 'portfolio', 'benchmark', 'volatility', 'expense', 'ratio'
        ]
        text_lower = self.text.lower()
        return any(keyword in text_lower for keyword in financial_keywords)
    
    def get_domain_relevance(self) -> float:
        """Calculate domain relevance score"""
        text_lower = self.text.lower()
        
        financial_terms = [
            'mutual fund', 'nav', 'net asset value', 'aum', 'assets under management',
            'return', 'performance', 'benchmark', 'outperformance', 'alpha', 'beta',
            'portfolio', 'holdings', 'top holdings', 'allocation', 'sector allocation',
            'risk', 'volatility', 'drawdown', 'standard deviation', 'sharpe ratio'
        ]
        
        term_count = sum(1 for term in financial_terms if term in text_lower)
        return min(term_count / len(financial_terms), 1.0) if financial_terms else 0.0
    
    def _calculate_complexity_score(self) -> float:
        """Calculate text complexity score"""
        score = 0.0
        
        # Length complexity
        length = len(self.text)
        if length > 1000:
            score += 0.3
        elif length > 500:
            score += 0.2
        elif length > 200:
            score += 0.1
        
        # Financial term complexity
        financial_terms = [
            'alpha', 'beta', 'sharpe ratio', 'sortino ratio', 'information ratio',
            'standard deviation', 'volatility', 'correlation', 'covariance',
            'compound annual growth rate', 'internal rate of return', 'net asset value'
        ]
        text_lower = self.text.lower()
        advanced_terms = sum(1 for term in financial_terms if term in text_lower)
        score += min(advanced_terms * 0.05, 0.3)
        
        # Numeric data complexity
        import re
        numeric_patterns = [
            r'\b\d+\.?\d*%\b',  # Percentages
            r'\b\d+\.?\d*\s*(?:cr|l|crore|lakh)\b',  # Indian currency
            r'\b\d+\.\d+\s*xirr\b',  # XIRR values
            r'\b\d+\.\d+\s*pe\b',  # P/E ratios
        ]
        
        numeric_count = 0
        for pattern in numeric_patterns:
            matches = re.findall(pattern, text_lower)
            numeric_count += len(matches)
        
        score += min(numeric_count * 0.02, 0.2)
        
        # Sentence structure complexity
        sentences = self.text.split('. ')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        if avg_sentence_length > 20:
            score += 0.2
        elif avg_sentence_length > 15:
            score += 0.1
        
        return min(score, 1.0)
    
    def get_quality_indicators(self) -> Dict[str, Any]:
        """Get quality indicators for the chunk"""
        return {
            'length_appropriate': 50 <= len(self.text) <= 2000,
            'has_financial_terms': self.has_financial_content(),
            'domain_relevance': self.get_domain_relevance(),
            'complexity_score': self._calculate_complexity_score(),
            'readability_score': self._calculate_readability_score(),
            'semantic_coherence': self._assess_semantic_coherence()
        }
    
    def _calculate_readability_score(self) -> float:
        """Calculate readability score"""
        words = self.text.split()
        sentences = self.text.split('. ')
        
        if not sentences:
            return 0.0
        
        avg_words_per_sentence = len(words) / len(sentences)
        
        # Simplified readability score
        if avg_words_per_sentence < 10:
            return 0.8  # Easy to read
        elif avg_words_per_sentence < 20:
            return 0.6  # Moderate
        else:
            return 0.4  # Difficult
    
    def _assess_semantic_coherence(self) -> float:
        """Assess semantic coherence of the chunk"""
        # Simplified coherence assessment based on financial term consistency
        sentences = self.text.split('. ')
        
        if len(sentences) <= 1:
            return 0.5
        
        financial_words = 0
        total_words = 0
        
        for sentence in sentences:
            words = sentence.split()
            total_words += len(words)
            financial_words += sum(1 for word in words if self.has_financial_content() and word.lower() in ['fund', 'nav', 'return', 'risk', 'portfolio'])
        
        coherence = financial_words / total_words if total_words > 0 else 0
        return min(coherence * 2, 1.0)  # Normalize to 0-1 range
    
    def update_metadata(self, additional_metadata: Dict[str, Any]):
        """Update chunk metadata"""
        self.metadata.update(additional_metadata)
        self._add_computed_metadata()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get chunk summary"""
        return {
            'id': self.id,
            'text_preview': self.text[:100] + '...' if len(self.text) > 100 else self.text,
            'length': len(self.text),
            'word_count': len(self.text.split()),
            'has_financial_content': self.has_financial_content(),
            'domain_relevance': self.get_domain_relevance(),
            'complexity_score': self._calculate_complexity_score(),
            'model': self.metadata.get('model', 'unknown'),
            'content_type': self.metadata.get('content_type', 'unknown')
        }
