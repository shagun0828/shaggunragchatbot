"""
Financial domain embedder
Enhances embeddings for financial domain terminology
"""

import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

from models.chunk import Chunk


class FinancialEmbedder:
    """Enhances embeddings for financial domain"""
    
    def __init__(self, base_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.base_model = SentenceTransformer(base_model)
        self.financial_terms = self._load_financial_vocabulary()
        self.domain_weights = 0.2  # Weight for domain enhancement
    
    def enhance_financial_embeddings(self, chunks: List[Chunk], base_embeddings: np.ndarray) -> np.ndarray:
        """Enhance embeddings for financial domain"""
        enhanced_embeddings = []
        
        for i, chunk in enumerate(chunks):
            financial_score = self._calculate_financial_relevance(chunk.text)
            base_embedding = base_embeddings[i]
            
            # Apply domain-specific enhancement
            if financial_score > 0:
                enhancement_factor = 1 + (financial_score * self.domain_weights)
                enhanced_embedding = base_embedding * enhancement_factor
            else:
                enhanced_embedding = base_embedding
            
            enhanced_embeddings.append(enhanced_embedding)
        
        return np.array(enhanced_embeddings)
    
    def _calculate_financial_relevance(self, text: str) -> float:
        """Calculate financial domain relevance score"""
        text_lower = text.lower()
        
        # Financial keywords with weights
        financial_keywords = {
            'nav': 0.3, 'returns': 0.3, 'aum': 0.2, 'expense ratio': 0.2,
            'fund': 0.1, 'investment': 0.1, 'equity': 0.2, 'debt': 0.2,
            'portfolio': 0.2, 'holdings': 0.2, 'allocation': 0.2,
            'risk': 0.2, 'performance': 0.2, 'growth': 0.1, 'dividend': 0.2,
            'benchmark': 0.2, 'volatility': 0.2, 'liquidity': 0.2
        }
        
        relevance = 0.0
        for keyword, weight in financial_keywords.items():
            if keyword in text_lower:
                relevance += weight
        
        # Cap relevance at 1.0
        return min(relevance, 1.0)
    
    def _load_financial_vocabulary(self) -> List[str]:
        """Load financial vocabulary for domain enhancement"""
        return [
            'nav', 'net asset value', 'returns', 'annual returns', 'aum', 'assets under management',
            'expense ratio', 'fund manager', 'equity', 'debt', 'portfolio', 'holdings',
            'allocation', 'sector allocation', 'risk level', 'performance', 'growth fund',
            'dividend', 'benchmark', 'volatility', 'liquidity', 'mutual fund', 'sip',
            'systematic investment plan', 'lumpsum', 'exit load', 'entry load', 'lock-in period',
            'tax saver', 'elss', 'mid cap', 'large cap', 'small cap', 'focused fund',
            'balanced fund', 'hybrid fund', 'debt fund', 'arbitrage fund'
        ]
