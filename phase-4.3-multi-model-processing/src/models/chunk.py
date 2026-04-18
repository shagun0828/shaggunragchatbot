"""
Chunk model for Phase 4.3
Represents a text chunk with metadata
"""

import uuid
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class Chunk:
    """Represents a text chunk with metadata"""
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary"""
        return {
            'id': self.id,
            'text': self.text,
            'metadata': self.metadata,
            'created_at': self.created_at,
            'text_length': len(self.text),
            'word_count': len(self.text.split())
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
    
    def has_financial_content(self) -> bool:
        """Check if chunk contains financial content"""
        financial_keywords = ['nav', 'return', 'fund', 'aum', '%', 'cr', 'lakh', 'crore']
        text_lower = self.text.lower()
        return any(keyword in text_lower for keyword in financial_keywords)
    
    def get_domain_relevance(self) -> float:
        """Calculate domain relevance score"""
        text_lower = self.text.lower()
        
        financial_terms = [
            'mutual fund', 'nav', 'net asset value', 'aum', 'assets under management',
            'return', 'performance', 'benchmark', 'risk', 'volatility', 'holding', 'allocation'
        ]
        
        term_count = sum(1 for term in financial_terms if term in text_lower)
        return min(term_count / len(financial_terms), 1.0) if financial_terms else 0.0
