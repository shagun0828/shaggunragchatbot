"""
Recursive Character Splitter
Advanced recursive text splitting with multiple separator strategies
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from models.chunk import Chunk


@dataclass
class SplittingRule:
    """Rule for text splitting"""
    separator: str
    priority: int
    description: str
    is_regex: bool = False


class RecursiveCharacterSplitter:
    """Advanced recursive character splitter with multiple strategies"""
    
    def __init__(self, custom_separators: Optional[List[SplittingRule]] = None):
        # Default splitting rules with priorities
        self.default_rules = [
            SplittingRule(r'\n\n\n', 1, "Triple newline - major section breaks", is_regex=True),
            SplittingRule(r'\n\n', 2, "Double newline - paragraph breaks", is_regex=True),
            SplittingRule('\n', 3, "Single newline - line breaks"),
            SplittingRule('. ', 4, "Period + space - sentence breaks"),
            SplittingRule('! ', 5, "Exclamation + space - emphatic breaks"),
            SplittingRule('? ', 6, "Question + space - question breaks"),
            SplittingRule('; ', 7, "Semicolon + space - clause breaks"),
            SplittingRule(', ', 8, "Comma + space - phrase breaks"),
            SplittingRule(' ', 9, "Space - word breaks"),
            SplittingRule('', 10, "Character-level - last resort")
        ]
        
        self.rules = custom_separators or self.default_rules
        self.max_chunk_size = 1000
        self.min_chunk_size = 50
        self.chunk_overlap = 0.1  # 10% overlap
        
        # Financial-specific separators
        self.financial_separators = [
            SplittingRule(r'(?<=\d+%)\s', 2.5, "After percentage values", is_regex=True),
            SplittingRule(r'(?<=\d+\.\d+)\s', 2.5, "After decimal numbers", is_regex=True),
            SplittingRule(r'(?<=Rs\.\s*\d+)\s', 2.5, "After rupee amounts", is_regex=True),
            SplittingRule(r'(?<=Cr)\s', 2.5, "After crore values", is_regex=True),
            SplittingRule(r'(?<=L)\s', 2.5, "After lakh values", is_regex=True)
        ]
    
    def chunk_recursive(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Recursively split text using multiple separator strategies"""
        chunks = []
        
        # Try different splitting strategies
        for strategy in ['semantic', 'financial', 'default']:
            strategy_chunks = self._apply_splitting_strategy(text, strategy, metadata)
            if strategy_chunks:
                chunks.extend(strategy_chunks)
                break
        
        # If no chunks created, fall back to character-level splitting
        if not chunks:
            chunks = self._character_level_splitting(text, metadata)
        
        # Post-process chunks
        chunks = self._post_process_chunks(chunks)
        
        return chunks
    
    def _apply_splitting_strategy(self, text: str, strategy: str, 
                                 metadata: Optional[Dict[str, Any]]) -> List[Chunk]:
        """Apply a specific splitting strategy"""
        if strategy == 'semantic':
            return self._semantic_splitting(text, metadata)
        elif strategy == 'financial':
            return self._financial_splitting(text, metadata)
        elif strategy == 'default':
            return self._default_splitting(text, metadata)
        return []
    
    def _semantic_splitting(self, text: str, metadata: Optional[Dict[str, Any]]) -> List[Chunk]:
        """Semantic splitting with content-aware separators"""
        # Identify semantic boundaries
        semantic_indicators = [
            r'(?<=\.)\s+(?=[A-Z])',  # After sentence, before capital
            r'(?<=\!)\s+(?=[A-Z])',  # After exclamation, before capital
            r'(?<=\?)\s+(?=[A-Z])',  # After question, before capital
            r'(?<=:)\s+(?=[A-Z])',   # After colon, before capital
        ]
        
        for pattern in semantic_indicators:
            chunks = self._split_with_pattern(text, pattern, metadata)
            if len(chunks) > 1:
                return chunks
        
        return []
    
    def _financial_splitting(self, text: str, metadata: Optional[Dict[str, Any]]) -> List[Chunk]:
        """Financial data-specific splitting"""
        # Combine default rules with financial separators
        enhanced_rules = self.financial_separators + self.default_rules
        
        for rule in sorted(enhanced_rules, key=lambda x: x.priority):
            chunks = self._split_with_rule(text, rule, metadata)
            if len(chunks) > 1:
                return chunks
        
        return []
    
    def _default_splitting(self, text: str, metadata: Optional[Dict[str, Any]]) -> List[Chunk]:
        """Default recursive splitting"""
        return self._recursive_split_with_rules(text, self.rules, 0, metadata)
    
    def _recursive_split_with_rules(self, text: str, rules: List[SplittingRule], 
                                   rule_index: int, metadata: Optional[Dict[str, Any]]) -> List[Chunk]:
        """Recursively split text using rules"""
        if rule_index >= len(rules):
            return [self._create_chunk(text, metadata, 'character_level')]
        
        rule = rules[rule_index]
        
        if len(text) <= self.max_chunk_size:
            return [self._create_chunk(text, metadata, f'rule_{rule.priority}')]
        
        chunks = []
        if rule.is_regex:
            parts = re.split(rule.separator, text)
        else:
            parts = text.split(rule.separator)
        
        # Filter out empty parts
        parts = [part.strip() for part in parts if part.strip()]
        
        if len(parts) == 1:
            # Try next rule
            return self._recursive_split_with_rules(text, rules, rule_index + 1, metadata)
        
        for part in parts:
            if len(part) <= self.max_chunk_size:
                chunks.append(self._create_chunk(part, metadata, f'rule_{rule.priority}'))
            else:
                # Recursively split large parts
                sub_chunks = self._recursive_split_with_rules(part, rules, rule_index + 1, metadata)
                chunks.extend(sub_chunks)
        
        return chunks
    
    def _split_with_pattern(self, text: str, pattern: str, 
                          metadata: Optional[Dict[str, Any]]) -> List[Chunk]:
        """Split text using regex pattern"""
        parts = re.split(pattern, text)
        parts = [part.strip() for part in parts if part.strip()]
        
        chunks = []
        for part in parts:
            if len(part) >= self.min_chunk_size and len(part) <= self.max_chunk_size:
                chunks.append(self._create_chunk(part, metadata, 'semantic_pattern'))
        
        return chunks
    
    def _split_with_rule(self, text: str, rule: SplittingRule, 
                        metadata: Optional[Dict[str, Any]]) -> List[Chunk]:
        """Split text using a specific rule"""
        if rule.is_regex:
            parts = re.split(rule.separator, text)
        else:
            parts = text.split(rule.separator)
        
        parts = [part.strip() for part in parts if part.strip()]
        
        chunks = []
        for part in parts:
            if len(part) >= self.min_chunk_size and len(part) <= self.max_chunk_size:
                chunk_metadata = {
                    'splitting_rule': rule.description,
                    'rule_priority': rule.priority,
                    **(metadata or {})
                }
                chunks.append(self._create_chunk(part, chunk_metadata, 'rule_based'))
        
        return chunks
    
    def _character_level_splitting(self, text: str, metadata: Optional[Dict[str, Any]]) -> List[Chunk]:
        """Last resort character-level splitting"""
        chunks = []
        chunk_size = self.max_chunk_size
        overlap = int(chunk_size * self.chunk_overlap)
        
        for i in range(0, len(text), chunk_size - overlap):
            chunk_text = text[i:i + chunk_size]
            if len(chunk_text) >= self.min_chunk_size:
                chunk_metadata = {
                    'splitting_method': 'character_level',
                    'start_position': i,
                    'overlap': overlap,
                    **(metadata or {})
                }
                chunks.append(self._create_chunk(chunk_text, chunk_metadata, 'character_level'))
        
        return chunks
    
    def _post_process_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Post-process chunks to improve quality"""
        processed_chunks = []
        
        for chunk in chunks:
            # Clean up text
            cleaned_text = self._clean_chunk_text(chunk.text)
            
            if self._is_valid_chunk(cleaned_text):
                chunk.text = cleaned_text
                chunk.metadata['post_processed'] = True
                chunk.metadata['final_length'] = len(cleaned_text)
                processed_chunks.append(chunk)
        
        return processed_chunks
    
    def _clean_chunk_text(self, text: str) -> str:
        """Clean chunk text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove leading/trailing punctuation
        text = text.strip('.,!?;:')
        
        # Ensure proper spacing around punctuation
        text = re.sub(r'([.,!?;:])(?!\s)', r'\1 ', text)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        
        return text.strip()
    
    def _is_valid_chunk(self, text: str) -> bool:
        """Check if chunk is valid"""
        if len(text) < self.min_chunk_size or len(text) > self.max_chunk_size:
            return False
        
        # Check for meaningful content
        if not re.search(r'[A-Za-z]', text):
            return False
        
        # Check for sentence structure
        if not re.search(r'[.!?]', text) and len(text) > 200:
            return False
        
        return True
    
    def _create_chunk(self, text: str, metadata: Optional[Dict[str, Any]], 
                     chunk_type: str) -> Chunk:
        """Create a chunk with metadata"""
        chunk_metadata = {
            'chunk_type': chunk_type,
            'text_length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(re.split(r'[.!?]+', text)),
            'max_chunk_size': self.max_chunk_size,
            'min_chunk_size': self.min_chunk_size,
            **(metadata or {})
        }
        return Chunk(text=text, metadata=chunk_metadata)
    
    def add_custom_rule(self, rule: SplittingRule) -> None:
        """Add a custom splitting rule"""
        self.rules.append(rule)
        # Sort rules by priority
        self.rules.sort(key=lambda x: x.priority)
    
    def set_chunk_parameters(self, max_size: int = None, min_size: int = None, 
                           overlap: float = None) -> None:
        """Update chunking parameters"""
        if max_size:
            self.max_chunk_size = max_size
        if min_size:
            self.min_chunk_size = min_size
        if overlap:
            self.chunk_overlap = overlap
