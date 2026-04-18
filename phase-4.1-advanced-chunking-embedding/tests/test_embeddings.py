"""
Test suite for advanced embedding functionality
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

# Import embedders (adjust path as needed)
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from embedders.enhanced_financial_embedder import EnhancedFinancialEmbedder
from embedders.embedding_quality_checker import EmbeddingQualityChecker, QualityReport, QualityIssue
from models.chunk import Chunk


class TestEnhancedFinancialEmbedder:
    """Test cases for Enhanced Financial Embedder"""
    
    @pytest.fixture
    def embedder(self):
        """Create embedder instance for testing"""
        return EnhancedFinancialEmbedder()
    
    @pytest.fixture
    def sample_chunks(self):
        """Sample chunks for testing"""
        return [
            Chunk(
                text="The fund generated 24.5% returns last year with NAV of â¹175.43.",
                metadata={'fund_name': 'Test Fund', 'section_type': 'performance'}
            ),
            Chunk(
                text="Top holdings include Reliance Industries (8.5%) and TCS (7.2%).",
                metadata={'fund_name': 'Test Fund', 'section_type': 'holdings'}
            ),
            Chunk(
                text="This is a low-risk investment option with stable returns.",
                metadata={'fund_name': 'Test Fund', 'section_type': 'description'}
            )
        ]
    
    @pytest.fixture
    def sample_embeddings(self):
        """Sample embeddings for testing"""
        return np.random.rand(3, 384)  # 3 chunks, 384 dimensions
    
    def test_enhance_financial_embeddings(self, embedder, sample_chunks, sample_embeddings):
        """Test financial embedding enhancement"""
        enhanced_embeddings = embedder.enhance_financial_embeddings(sample_chunks, sample_embeddings)
        
        assert enhanced_embeddings.shape == sample_embeddings.shape
        assert not np.array_equal(enhanced_embeddings, sample_embeddings)  # Should be different
        
        # Check that embeddings are normalized
        norms = np.linalg.norm(enhanced_embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)
    
    def test_context_analysis(self, embedder):
        """Test chunk context analysis"""
        text = "The fund returned 24.5% last year with NAV of â¹175.43 and AUM of â¹28,432 Cr."
        context = embedder._analyze_chunk_context(text)
        
        assert 'primary_category' in context
        assert 'category_scores' in context
        assert 'financial_terms' in context
        assert 'numeric_data' in context
        assert 'sentiment' in context
        assert 'complexity' in context
        
        # Should detect financial terms
        assert len(context['financial_terms']) > 0
        assert any('return' in term for term in context['financial_terms'])
    
    def test_domain_relevance_calculation(self, embedder):
        """Test domain relevance calculation"""
        high_relevance_text = "Fund returns 24.5% with NAV â¹175.43 and AUM â¹28,432 Cr"
        low_relevance_text = "This is some random text without financial content"
        
        high_context = embedder._analyze_chunk_context(high_relevance_text)
        low_context = embedder._analyze_chunk_context(low_relevance_text)
        
        high_relevance = embedder._calculate_domain_relevance(high_relevance_text, high_context)
        low_relevance = embedder._calculate_domain_relevance(low_relevance_text, low_context)
        
        assert high_relevance > low_relevance
        assert 0.0 <= high_relevance <= 1.0
        assert 0.0 <= low_relevance <= 1.0
    
    def test_context_enhancement(self, embedder, sample_chunks, sample_embeddings):
        """Test context-aware enhancement"""
        chunk = sample_chunks[0]  # Performance chunk
        embedding = sample_embeddings[0]
        
        context = embedder._analyze_chunk_context(chunk.text)
        domain_relevance = embedder._calculate_domain_relevance(chunk.text, context)
        
        enhanced_embedding = embedder._apply_context_enhancement(
            embedding, chunk.text, context, domain_relevance
        )
        
        assert enhanced_embedding.shape == embedding.shape
        assert not np.array_equal(enhanced_embedding, embedding)
    
    def test_enhancement_statistics(self, embedder, sample_chunks):
        """Test enhancement statistics calculation"""
        stats = embedder.get_enhancement_statistics(sample_chunks)
        
        assert 'total_chunks' in stats
        assert 'context_distribution' in stats
        assert 'average_domain_relevance' in stats
        assert 'financial_term_coverage' in stats
        
        assert stats['total_chunks'] == len(sample_chunks)
        assert 0.0 <= stats['average_domain_relevance'] <= 1.0


class TestEmbeddingQualityChecker:
    """Test cases for Embedding Quality Checker"""
    
    @pytest.fixture
    def quality_checker(self):
        """Create quality checker instance for testing"""
        return EmbeddingQualityChecker()
    
    @pytest.fixture
    def sample_chunks(self):
        """Sample chunks for testing"""
        return [
            Chunk(text="Fund performance: 24.5% returns", metadata={'section': 'performance'}),
            Chunk(text="Portfolio holdings: Reliance 8.5%", metadata={'section': 'holdings'}),
            Chunk(text="Risk analysis: Moderate volatility", metadata={'section': 'risk'})
        ]
    
    @pytest.fixture
    def good_embeddings(self):
        """Good quality embeddings for testing"""
        # Generate diverse, normalized embeddings
        embeddings = np.random.rand(3, 384)
        # Normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings
    
    @pytest.fixture
    def poor_embeddings(self):
        """Poor quality embeddings for testing"""
        embeddings = np.zeros((3, 384))  # Zero variance embeddings
        embeddings[0] = np.ones(384)  # Duplicate
        embeddings[1] = np.ones(384)  # Duplicate
        embeddings[2] = np.random.rand(384) * 0.001  # Very low variance
        return embeddings
    
    def test_check_embedding_quality_good(self, quality_checker, sample_chunks, good_embeddings):
        """Test quality checking with good embeddings"""
        quality_report = quality_checker.check_embedding_quality(good_embeddings, sample_chunks)
        
        assert isinstance(quality_report, QualityReport)
        assert quality_report.total_embeddings == len(sample_chunks)
        assert quality_report.dimension == good_embeddings.shape[1]
        assert quality_report.overall_score > 0.7  # Should be high quality
        assert len(quality_report.issues) == 0 or all(len(indices) == 0 for indices in quality_report.issues.values())
    
    def test_check_embedding_quality_poor(self, quality_checker, sample_chunks, poor_embeddings):
        """Test quality checking with poor embeddings"""
        quality_report = quality_checker.check_embedding_quality(poor_embeddings, sample_chunks)
        
        assert isinstance(quality_report, QualityReport)
        assert quality_report.overall_score < 0.7  # Should be low quality
        
        # Should detect issues
        assert len(quality_report.issues[QualityIssue.DUPLICATE_EMBEDDINGS]) > 0
        assert len(quality_report.issues[QualityIssue.LOW_VARIANCE]) > 0
    
    def test_duplicate_detection(self, quality_checker):
        """Test duplicate embedding detection"""
        # Create embeddings with duplicates
        embeddings = np.random.rand(4, 384)
        embeddings[1] = embeddings[0]  # Duplicate
        embeddings[3] = embeddings[2]  # Duplicate
        
        chunks = [Chunk(text=f"Chunk {i}", metadata={}) for i in range(4)]
        
        quality_report = quality_checker.check_embedding_quality(embeddings, chunks)
        
        # Should detect 2 pairs of duplicates
        assert len(quality_report.issues[QualityIssue.DUPLICATE_EMBEDDINGS]) >= 2
        assert quality_report.statistics['duplicate_pairs'] >= 2
    
    def test_variance_analysis(self, quality_checker):
        """Test embedding variance analysis"""
        # Create embeddings with varying variance
        embeddings = np.random.rand(3, 384)
        embeddings[0] *= 0.001  # Very low variance
        embeddings[1] *= 0.01   # Low variance
        embeddings[2] *= 1.0     # Normal variance
        
        chunks = [Chunk(text=f"Chunk {i}", metadata={}) for i in range(3)]
        
        quality_report = quality_checker.check_embedding_quality(embeddings, chunks)
        
        # Should detect low variance embeddings
        assert len(quality_report.issues[QualityIssue.LOW_VARIANCE]) >= 1
        assert quality_report.statistics['min_variance'] < quality_report.statistics['max_variance']
    
    def test_outlier_detection(self, quality_checker):
        """Test outlier embedding detection"""
        # Create embeddings with outliers
        embeddings = np.random.rand(3, 384)
        embeddings[0] *= 5.0  # Outlier (large norm)
        embeddings[1] *= 0.1  # Outlier (small norm)
        embeddings[2] *= 1.0  # Normal
        
        chunks = [Chunk(text=f"Chunk {i}", metadata={}) for i in range(3)]
        
        quality_report = quality_checker.check_embedding_quality(embeddings, chunks)
        
        # Should detect outliers
        assert len(quality_report.issues[QualityIssue.OUTLIER]) >= 1
        assert quality_report.statistics['outlier_count'] >= 1
    
    def test_nan_infinite_detection(self, quality_checker):
        """Test NaN and infinite value detection"""
        # Create embeddings with NaN and infinite values
        embeddings = np.random.rand(3, 384)
        embeddings[0, 0] = np.nan  # NaN value
        embeddings[1, 0] = np.inf  # Infinite value
        
        chunks = [Chunk(text=f"Chunk {i}", metadata={}) for i in range(3)]
        
        quality_report = quality_checker.check_embedding_quality(embeddings, chunks)
        
        # Should detect NaN and infinite values
        assert len(quality_report.issues[QualityIssue.NAN_VALUES]) >= 1
        assert len(quality_report.issues[QualityIssue.INFINITE_VALUES]) >= 1
    
    def test_overall_quality_score_calculation(self, quality_checker):
        """Test overall quality score calculation"""
        # Test with perfect embeddings
        perfect_embeddings = np.random.rand(5, 384)
        perfect_embeddings = perfect_embeddings / np.linalg.norm(perfect_embeddings, axis=1, keepdims=True)
        chunks = [Chunk(text=f"Chunk {i}", metadata={}) for i in range(5)]
        
        quality_report = quality_checker.check_embedding_quality(perfect_embeddings, chunks)
        
        assert 0.0 <= quality_report.overall_score <= 1.0
        assert quality_report.overall_score > 0.5  # Should be reasonably high
    
    def test_recommendation_generation(self, quality_checker):
        """Test recommendation generation"""
        # Create problematic embeddings
        embeddings = np.zeros((3, 384))  # All zero variance
        chunks = [Chunk(text=f"Chunk {i}", metadata={}) for i in range(3)]
        
        quality_report = quality_checker.check_embedding_quality(embeddings, chunks)
        
        assert len(quality_report.recommendations) > 0
        assert any("variance" in rec.lower() for rec in quality_report.recommendations)
    
    def test_quality_report_summary(self, quality_checker):
        """Test quality report summary generation"""
        embeddings = np.random.rand(3, 384)
        chunks = [Chunk(text=f"Chunk {i}", metadata={}) for i in range(3)]
        
        quality_report = quality_checker.check_embedding_quality(embeddings, chunks)
        summary = quality_checker.generate_quality_report_summary(quality_report)
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "Embedding Quality Report" in summary
        assert "Overall Quality Score" in summary


class TestEmbeddingIntegration:
    """Integration tests for embedding components"""
    
    def test_end_to_end_embedding_pipeline(self):
        """Test end-to-end embedding pipeline"""
        # Sample data
        chunks = [
            Chunk(
                text="Fund generated 24.5% returns with NAV â¹175.43.",
                metadata={'fund_name': 'Test Fund', 'section_type': 'performance'}
            )
        ]
        
        # Generate base embeddings
        embedder = EnhancedFinancialEmbedder()
        base_embeddings = embedder.base_model.encode([chunk.text for chunk in chunks])
        
        # Enhance embeddings
        enhanced_embeddings = embedder.enhance_financial_embeddings(chunks, base_embeddings)
        
        # Check quality
        quality_checker = EmbeddingQualityChecker()
        quality_report = quality_checker.check_embedding_quality(enhanced_embeddings, chunks)
        
        # Verify pipeline
        assert enhanced_embeddings.shape == base_embeddings.shape
        assert quality_report.overall_score > 0.0
        assert len(quality_report.recommendations) >= 0
    
    def test_batch_processing(self):
        """Test batch processing capabilities"""
        # Create larger dataset
        chunks = [
            Chunk(
                text=f"Fund chunk {i} with {i}% returns",
                metadata={'fund_name': f'Fund {i}', 'section_type': 'performance'}
            )
            for i in range(10)
        ]
        
        embedder = EnhancedFinancialEmbedder()
        base_embeddings = embedder.base_model.encode([chunk.text for chunk in chunks])
        
        # Test batch enhancement
        enhanced_embeddings = embedder.batch_enhance_embeddings(chunks, base_embeddings, batch_size=3)
        
        assert enhanced_embeddings.shape == base_embeddings.shape
        assert len(enhanced_embeddings) == len(chunks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
