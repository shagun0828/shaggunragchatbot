"""
Embedding Quality Assurance System
Comprehensive quality checking and validation for embeddings
"""

import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from dataclasses import dataclass
from enum import Enum

from models.chunk import Chunk


class QualityIssue(Enum):
    """Types of quality issues"""
    DUPLICATE_EMBEDDINGS = "duplicate_embeddings"
    LOW_VARIANCE = "low_variance"
    OUTLIER = "outlier"
    DIMENSION_MISMATCH = "dimension_mismatch"
    NAN_VALUES = "nan_values"
    INFINITE_VALUES = "infinite_values"
    POOR_COVERAGE = "poor_coverage"
    CLUSTER_IMBALANCE = "cluster_imbalance"


@dataclass
class QualityReport:
    """Quality report for embeddings"""
    total_embeddings: int
    dimension: int
    issues: Dict[QualityIssue, List[int]]
    statistics: Dict[str, float]
    recommendations: List[str]
    overall_score: float


class EmbeddingQualityChecker:
    """Comprehensive embedding quality assurance system"""
    
    def __init__(self, similarity_threshold: float = 0.95, 
                 variance_threshold: float = 0.01,
                 outlier_threshold: float = 3.0):
        self.logger = logging.getLogger(__name__)
        self.similarity_threshold = similarity_threshold
        self.variance_threshold = variance_threshold
        self.outlier_threshold = outlier_threshold
        
        # Quality weights for overall scoring
        self.quality_weights = {
            'duplicate_penalty': 0.3,
            'variance_penalty': 0.2,
            'outlier_penalty': 0.2,
            'coverage_bonus': 0.15,
            'balance_bonus': 0.15
        }
    
    def check_embedding_quality(self, embeddings: np.ndarray, chunks: List[Chunk]) -> QualityReport:
        """Comprehensive quality check for embeddings"""
        self.logger.info(f"Checking quality for {len(embeddings)} embeddings")
        
        # Initialize quality report
        issues = {issue_type: [] for issue_type in QualityIssue}
        statistics = {}
        recommendations = []
        
        # Basic validation
        self._validate_basic_properties(embeddings, issues, statistics)
        
        # Check for duplicates
        self._check_duplicate_embeddings(embeddings, issues, statistics)
        
        # Check variance
        self._check_embedding_variance(embeddings, issues, statistics)
        
        # Check for outliers
        self._check_outliers(embeddings, issues, statistics)
        
        # Check coverage
        self._check_embedding_coverage(embeddings, chunks, statistics)
        
        # Check cluster balance
        self._check_cluster_balance(embeddings, statistics)
        
        # Calculate overall quality score
        overall_score = self._calculate_overall_quality_score(issues, statistics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(issues, statistics)
        
        return QualityReport(
            total_embeddings=len(embeddings),
            dimension=embeddings.shape[1] if len(embeddings.shape) > 1 else 0,
            issues=issues,
            statistics=statistics,
            recommendations=recommendations,
            overall_score=overall_score
        )
    
    def _validate_basic_properties(self, embeddings: np.ndarray, issues: Dict[QualityIssue, List[int]], 
                                  statistics: Dict[str, float]) -> None:
        """Validate basic embedding properties"""
        # Check for NaN values
        nan_indices = np.where(np.isnan(embeddings))[0]
        if len(nan_indices) > 0:
            issues[QualityIssue.NAN_VALUES].extend(nan_indices.tolist())
        
        # Check for infinite values
        inf_indices = np.where(np.isinf(embeddings))[0]
        if len(inf_indices) > 0:
            issues[QualityIssue.INFINITE_VALUES].extend(inf_indices.tolist())
        
        # Check dimension consistency
        if len(embeddings.shape) == 2:
            statistics['dimension'] = embeddings.shape[1]
        else:
            issues[QualityIssue.DIMENSION_MISMATCH].extend(list(range(len(embeddings))))
        
        # Basic statistics
        statistics['mean_norm'] = np.mean(np.linalg.norm(embeddings, axis=1))
        statistics['std_norm'] = np.std(np.linalg.norm(embeddings, axis=1))
        statistics['mean_value'] = np.mean(embeddings)
        statistics['std_value'] = np.std(embeddings)
    
    def _check_duplicate_embeddings(self, embeddings: np.ndarray, issues: Dict[QualityIssue, List[int]], 
                                   statistics: Dict[str, float]) -> None:
        """Check for duplicate or highly similar embeddings"""
        if len(embeddings) < 2:
            return
        
        similarity_matrix = cosine_similarity(embeddings)
        
        # Find highly similar pairs (excluding diagonal)
        duplicate_pairs = []
        for i in range(len(similarity_matrix)):
            for j in range(i + 1, len(similarity_matrix)):
                if similarity_matrix[i][j] > self.similarity_threshold:
                    duplicate_pairs.append((i, j))
        
        # Record indices of duplicates
        duplicate_indices = set()
        for i, j in duplicate_pairs:
            duplicate_indices.add(i)
            duplicate_indices.add(j)
        
        issues[QualityIssue.DUPLICATE_EMBEDDINGS].extend(list(duplicate_indices))
        statistics['duplicate_pairs'] = len(duplicate_pairs)
        statistics['duplicate_rate'] = len(duplicate_indices) / len(embeddings)
        statistics['avg_similarity'] = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
    
    def _check_embedding_variance(self, embeddings: np.ndarray, issues: Dict[QualityIssue, List[int]], 
                                 statistics: Dict[str, float]) -> None:
        """Check for low variance embeddings"""
        # Calculate variance for each embedding
        embedding_variances = np.var(embeddings, axis=1)
        
        # Find embeddings with low variance
        low_variance_indices = np.where(embedding_variances < self.variance_threshold)[0]
        issues[QualityIssue.LOW_VARIANCE].extend(low_variance_indices.tolist())
        
        statistics['min_variance'] = np.min(embedding_variances)
        statistics['max_variance'] = np.max(embedding_variances)
        statistics['mean_variance'] = np.mean(embedding_variances)
        statistics['low_variance_rate'] = len(low_variance_indices) / len(embeddings)
    
    def _check_outliers(self, embeddings: np.ndarray, issues: Dict[QualityIssue, List[int]], 
                       statistics: Dict[str, float]) -> None:
        """Check for outlier embeddings using z-score"""
        # Calculate norms
        norms = np.linalg.norm(embeddings, axis=1)
        
        # Calculate z-scores
        z_scores = np.abs((norms - np.mean(norms)) / np.std(norms))
        
        # Find outliers
        outlier_indices = np.where(z_scores > self.outlier_threshold)[0]
        issues[QualityIssue.OUTLIER].extend(outlier_indices.tolist())
        
        statistics['outlier_count'] = len(outlier_indices)
        statistics['outlier_rate'] = len(outlier_indices) / len(embeddings)
        statistics['norm_std'] = np.std(norms)
    
    def _check_embedding_coverage(self, embeddings: np.ndarray, chunks: List[Chunk], 
                                statistics: Dict[str, float]) -> None:
        """Check embedding space coverage"""
        if len(embeddings) < 10:
            statistics['coverage_score'] = 0.5  # Default for small datasets
            return
        
        # Use PCA to reduce to 2D for coverage analysis
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
        # Calculate convex hull area (simplified)
        # For simplicity, use bounding box area
        min_coords = np.min(embeddings_2d, axis=0)
        max_coords = np.max(embeddings_2d, axis=0)
        bounding_area = np.prod(max_coords - min_coords)
        
        # Calculate density
        density = len(embeddings) / (bounding_area + 1e-8)
        
        # Normalize coverage score (0-1)
        coverage_score = min(density / 1000, 1.0)  # Arbitrary normalization
        statistics['coverage_score'] = coverage_score
        statistics['pca_explained_variance'] = np.sum(pca.explained_variance_ratio_)
    
    def _check_cluster_balance(self, embeddings: np.ndarray, statistics: Dict[str, float]) -> None:
        """Check cluster balance in embedding space"""
        if len(embeddings) < 20:
            statistics['cluster_balance_score'] = 0.5
            return
        
        # Determine optimal number of clusters (simplified)
        n_clusters = min(8, max(2, len(embeddings) // 10))
        
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Calculate cluster sizes
            cluster_sizes = [np.sum(cluster_labels == i) for i in range(n_clusters)]
            
            # Calculate balance score (lower variance in cluster sizes = better balance)
            expected_size = len(embeddings) / n_clusters
            size_variance = np.var(cluster_sizes)
            balance_score = 1.0 / (1.0 + size_variance / (expected_size ** 2))
            
            statistics['cluster_balance_score'] = balance_score
            statistics['n_clusters'] = n_clusters
            statistics['cluster_sizes'] = cluster_sizes
            
        except Exception as e:
            self.logger.warning(f"Cluster analysis failed: {e}")
            statistics['cluster_balance_score'] = 0.5
    
    def _calculate_overall_quality_score(self, issues: Dict[QualityIssue, List[int]], 
                                       statistics: Dict[str, float]) -> float:
        """Calculate overall quality score (0-1)"""
        score = 1.0
        
        # Penalize duplicates
        duplicate_rate = statistics.get('duplicate_rate', 0)
        score -= duplicate_rate * self.quality_weights['duplicate_penalty']
        
        # Penalize low variance
        low_variance_rate = len(issues[QualityIssue.LOW_VARIANCE]) / max(1, sum(len(v) for v in issues.values()))
        score -= low_variance_rate * self.quality_weights['variance_penalty']
        
        # Penalize outliers
        outlier_rate = statistics.get('outlier_rate', 0)
        score -= outlier_rate * self.quality_weights['outlier_penalty']
        
        # Bonus for good coverage
        coverage_bonus = statistics.get('coverage_score', 0) * self.quality_weights['coverage_bonus']
        score += coverage_bonus
        
        # Bonus for cluster balance
        balance_bonus = statistics.get('cluster_balance_score', 0) * self.quality_weights['balance_bonus']
        score += balance_bonus
        
        return max(0.0, min(1.0, score))
    
    def _generate_recommendations(self, issues: QualityIssue, statistics: Dict[str, float]) -> List[str]:
        """Generate recommendations based on quality issues"""
        recommendations = []
        
        # Duplicate recommendations
        if issues[QualityIssue.DUPLICATE_EMBEDDINGS]:
            duplicate_rate = statistics.get('duplicate_rate', 0)
            recommendations.append(
                f"High duplicate rate ({duplicate_rate:.2%}). Consider improving chunking strategy "
                "or increasing similarity threshold."
            )
        
        # Low variance recommendations
        if issues[QualityIssue.LOW_VARIANCE]:
            recommendations.append(
                "Low variance embeddings detected. Consider using more diverse text "
                "or different embedding model."
            )
        
        # Outlier recommendations
        if issues[QualityIssue.OUTLIER]:
            recommendations.append(
                "Outlier embeddings detected. Review chunk quality and consider "
                "filtering or reprocessing outlier chunks."
            )
        
        # Coverage recommendations
        coverage_score = statistics.get('coverage_score', 0)
        if coverage_score < 0.5:
            recommendations.append(
                "Poor embedding space coverage. Consider increasing dataset diversity "
                "or using different embedding parameters."
            )
        
        # Cluster balance recommendations
        balance_score = statistics.get('cluster_balance_score', 0)
        if balance_score < 0.6:
            recommendations.append(
                "Imbalanced clustering detected. Consider rebalancing data or "
                "adjusting chunking strategy."
            )
        
        # General recommendations
        overall_score = statistics.get('overall_score', 0)
        if overall_score > 0.8:
            recommendations.append("Excellent embedding quality!")
        elif overall_score > 0.6:
            recommendations.append("Good embedding quality with minor improvements possible.")
        else:
            recommendations.append("Significant quality issues detected. Review entire pipeline.")
        
        return recommendations
    
    def visualize_embedding_quality(self, embeddings: np.ndarray, chunks: List[Chunk], 
                                  save_path: Optional[str] = None) -> None:
        """Visualize embedding quality metrics"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Similarity heatmap
            similarity_matrix = cosine_similarity(embeddings[:50])  # Limit for visualization
            axes[0, 0].imshow(similarity_matrix, cmap='viridis')
            axes[0, 0].set_title('Embedding Similarity Heatmap (Sample)')
            axes[0, 0].set_xlabel('Embedding Index')
            axes[0, 0].set_ylabel('Embedding Index')
            
            # 2. Norm distribution
            norms = np.linalg.norm(embeddings, axis=1)
            axes[0, 1].hist(norms, bins=30, alpha=0.7)
            axes[0, 1].set_title('Embedding Norm Distribution')
            axes[0, 1].set_xlabel('Norm')
            axes[0, 1].set_ylabel('Frequency')
            
            # 3. PCA visualization
            if len(embeddings) > 2:
                pca = PCA(n_components=2)
                embeddings_2d = pca.fit_transform(embeddings)
                axes[1, 0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6)
                axes[1, 0].set_title(f'PCA Visualization (Explained Variance: {np.sum(pca.explained_variance_ratio_):.2%})')
                axes[1, 0].set_xlabel('PCA Component 1')
                axes[1, 0].set_ylabel('PCA Component 2')
            
            # 4. Variance distribution
            variances = np.var(embeddings, axis=1)
            axes[1, 1].hist(variances, bins=30, alpha=0.7)
            axes[1, 1].set_title('Embedding Variance Distribution')
            axes[1, 1].set_xlabel('Variance')
            axes[1, 1].set_ylabel('Frequency')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Quality visualization saved to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            self.logger.error(f"Visualization failed: {e}")
    
    def generate_quality_report_summary(self, report: QualityReport) -> str:
        """Generate a human-readable quality report summary"""
        summary = [
            f"Embedding Quality Report",
            f"========================",
            f"",
            f"Total Embeddings: {report.total_embeddings}",
            f"Dimension: {report.dimension}",
            f"Overall Quality Score: {report.overall_score:.3f}",
            f"",
            f"Issues Found:",
            f"------------"
        ]
        
        for issue_type, indices in report.issues.items():
            if indices:
                summary.append(f"{issue_type.value}: {len(indices)} embeddings")
        
        summary.extend([
            f"",
            f"Statistics:",
            f"----------"
        ])
        
        for key, value in report.statistics.items():
            if isinstance(value, float):
                summary.append(f"{key}: {value:.4f}")
            else:
                summary.append(f"{key}: {value}")
        
        summary.extend([
            f"",
            f"Recommendations:",
            f"----------------"
        ])
        
        for i, recommendation in enumerate(report.recommendations, 1):
            summary.append(f"{i}. {recommendation}")
        
        return "\n".join(summary)
