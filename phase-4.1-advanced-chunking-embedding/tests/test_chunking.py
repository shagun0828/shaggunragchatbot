"""
Test suite for advanced chunking functionality
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

# Import chunkers (adjust path as needed)
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from chunkers.enhanced_semantic_chunker import EnhancedSemanticChunker
from chunkers.recursive_character_splitter import RecursiveCharacterSplitter
from chunkers.mutual_fund_chunker_v2 import MutualFundChunkerV2
from models.chunk import Chunk


class TestEnhancedSemanticChunker:
    """Test cases for Enhanced Semantic Chunker"""
    
    @pytest.fixture
    def chunker(self):
        """Create chunker instance for testing"""
        return EnhancedSemanticChunker()
    
    @pytest.fixture
    def sample_text(self):
        """Sample text for testing"""
        return """
        HDFC Mid-Cap Fund Direct Growth is a mutual fund scheme that invests in mid-cap companies.
        The fund has generated returns of 24.5% in the last year. The current NAV is â¹175.43.
        The fund manager is Rashmi Joshi and the AUM is â¹28,432 Cr. The expense ratio is 1.25%.
        The top holdings include Reliance Industries (8.5%), TCS (7.2%), and HDFC Bank (6.8%).
        The sector allocation includes Financial Services (25%), Technology (20%), and Healthcare (15%).
        """
    
    def test_chunk_semantic_enhanced(self, chunker, sample_text):
        """Test enhanced semantic chunking"""
        chunks = chunker.chunk_semantic_enhanced(sample_text)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert len(chunk.text) >= chunker.min_chunk_size
            assert len(chunk.text) <= chunker.max_chunk_size
            assert 'chunk_type' in chunk.metadata
            assert chunk.metadata['chunk_type'] == 'enhanced_semantic'
    
    def test_adaptive_threshold_calculation(self, chunker):
        """Test adaptive threshold calculation"""
        # Create sample similarity matrix
        similarity_matrix = np.array([
            [1.0, 0.8, 0.3],
            [0.8, 1.0, 0.4],
            [0.3, 0.4, 1.0]
        ])
        
        threshold = chunker._calculate_adaptive_threshold(similarity_matrix)
        
        assert 0.3 <= threshold <= 0.9
        assert isinstance(threshold, float)
    
    def test_financial_sentence_splitting(self, chunker):
        """Test financial-aware sentence splitting"""
        financial_text = "The fund returned 24.5% last year. NAV is â¹175.43. AUM is â¹28,432 Cr."
        sentences = chunker._split_sentences(financial_text)
        
        assert len(sentences) >= 2
        for sentence in sentences:
            assert len(sentence.strip()) > 10


class TestRecursiveCharacterSplitter:
    """Test cases for Recursive Character Splitter"""
    
    @pytest.fixture
    def splitter(self):
        """Create splitter instance for testing"""
        return RecursiveCharacterSplitter()
    
    @pytest.fixture
    def sample_text(self):
        """Sample text with various separators"""
        return """
        Basic Information: HDFC Mid-Cap Fund Direct Growth
        
        Performance Metrics:
        Returns: 24.5% (1Y), 18.2% (3Y), 16.8% (5Y)
        NAV: â¹175.43
        Expense Ratio: 1.25%
        
        Portfolio Holdings:
        Reliance Industries (8.5%), TCS (7.2%), HDFC Bank (6.8%)
        """
    
    def test_chunk_recursive(self, splitter, sample_text):
        """Test recursive chunking"""
        chunks = splitter.chunk_recursive(sample_text)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert len(chunk.text) >= splitter.min_chunk_size
            assert len(chunk.text) <= splitter.max_chunk_size
    
    def test_financial_splitting_rules(self, splitter):
        """Test financial-specific splitting rules"""
        financial_text = "Returns: 24.5% | AUM: â¹28,432 Cr | NAV: â¹175.43"
        chunks = splitter.chunk_recursive(financial_text)
        
        assert len(chunks) >= 1
        # Should split at financial indicators
        financial_chunks = [c for c in chunks if any(indicator in c.text.lower() 
                           for indicator in ['%', 'cr', 'nav'])]
        assert len(financial_chunks) > 0


class TestMutualFundChunkerV2:
    """Test cases for Mutual Fund Chunker V2"""
    
    @pytest.fixture
    def chunker(self):
        """Create chunker instance for testing"""
        return MutualFundChunkerV2()
    
    @pytest.fixture
    def sample_fund_data(self):
        """Sample fund data for testing"""
        return {
            'fund_name': 'HDFC Mid-Cap Fund Direct Growth',
            'category': 'Mid Cap',
            'risk_level': 'Very High',
            'aum': 'â¹28,432 Cr',
            'nav': 'â¹175.43',
            'returns': {'1_year': '24.5%', '3_year': '18.2%', '5_year': '16.8%'},
            'expense_ratio': '1.25%',
            'fund_manager': 'Rashmi Joshi',
            'inception_date': '01-Jan-2010',
            'description': 'The scheme aims to generate long-term capital appreciation by investing in mid-cap companies.',
            'top_holdings': [
                {'name': 'Reliance Industries', 'percentage': '8.5%'},
                {'name': 'TCS', 'percentage': '7.2%'}
            ],
            'sector_allocation': [
                {'sector': 'Financial Services', 'allocation': '25%'},
                {'sector': 'Technology', 'allocation': '20%'}
            ]
        }
    
    def test_chunk_fund_data_advanced(self, chunker, sample_fund_data):
        """Test advanced fund data chunking"""
        chunks = chunker.chunk_fund_data_advanced(sample_fund_data)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.metadata.get('fund_name') == sample_fund_data['fund_name']
            assert 'section_type' in chunk.metadata
        
        # Check for different section types
        section_types = set(chunk.metadata.get('section_type') for chunk in chunks)
        assert len(section_types) > 1  # Should have multiple sections
    
    def test_section_extraction(self, chunker, sample_fund_data):
        """Test section data extraction"""
        config = chunker.section_configs[0]  # Basic info config
        section_data = chunker._extract_section_data(sample_fund_data, config)
        
        assert isinstance(section_data, dict)
        assert 'fund_name' in section_data
        assert section_data['fund_name'] == sample_fund_data['fund_name']
    
    def test_quality_score_calculation(self, chunker):
        """Test chunk quality scoring"""
        chunk = Chunk(
            text="This fund has 24.5% returns and NAV of â¹175.43 with AUM of â¹28,432 Cr.",
            metadata={'fund_name': 'Test Fund', 'section_type': 'performance'}
        )
        
        quality_score = chunker._calculate_chunk_quality(chunk)
        
        assert 0.0 <= quality_score <= 1.0
        assert isinstance(quality_score, float)


class TestChunkIntegration:
    """Integration tests for chunking components"""
    
    def test_end_to_end_chunking(self):
        """Test end-to-end chunking pipeline"""
        # Sample fund data
        fund_data = {
            'fund_name': 'Test Fund',
            'description': 'This is a test mutual fund with 24.5% returns and NAV of â¹175.43.',
            'returns': {'1_year': '24.5%'},
            'top_holdings': [{'name': 'Test Company', 'percentage': '10%'}]
        }
        
        # Process with different chunkers
        semantic_chunker = EnhancedSemanticChunker()
        fund_chunker = MutualFundChunkerV2()
        
        # Semantic chunking
        semantic_chunks = semantic_chunker.chunk_semantic_enhanced(fund_data['description'])
        
        # Fund-specific chunking
        fund_chunks = fund_chunker.chunk_fund_data_advanced(fund_data)
        
        # Verify results
        assert len(semantic_chunks) > 0
        assert len(fund_chunks) > 0
        
        # All chunks should be valid
        all_chunks = semantic_chunks + fund_chunks
        for chunk in all_chunks:
            assert isinstance(chunk, Chunk)
            assert len(chunk.text) > 10
            assert chunk.metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
