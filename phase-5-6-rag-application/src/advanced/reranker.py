"""
Advanced Reranker for Phase 5-6 Application
Optimizes search results using multiple ranking strategies
"""

import asyncio
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class RerankingResult:
    """Result of reranking operation"""
    original_results: List[Dict[str, Any]]
    reranked_results: List[Dict[str, Any]]
    reranking_method: str
    improvement_score: float
    processing_time: float
    metadata: Dict[str, Any]


class Reranker:
    """Advanced reranker with multiple ranking strategies"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Reranking strategies
        self.strategies = {
            "cross_encoder": self._cross_encoder_rerank,
            "learning_to_rank": self._learning_to_rank_rerank,
            "maximal_marginal_relevance": self._mmr_rerank,
            "diversification": self._diversification_rerank,
            "temporal_bias": self._temporal_bias_rerank,
            "quality_weighted": self._quality_weighted_rerank
        }
        
        # Feature weights for learning to rank
        self.feature_weights = {
            "semantic_similarity": 0.3,
            "keyword_match": 0.2,
            "document_quality": 0.2,
            "recency": 0.15,
            "popularity": 0.15
        }
    
    async def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        method: str = "cross_encoder",
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Rerank search results using specified method
        """
        if method not in self.strategies:
            raise ValueError(f"Unknown reranking method: {method}")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            self.logger.info(f"Reranking {len(results)} results using {method}")
            
            # Apply reranking strategy
            reranking_function = self.strategies[method]
            reranked_results = await reranking_function(query, results, top_k)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            # Add reranking metadata
            for i, result in enumerate(reranked_results):
                result["rerank_score"] = result.get("score", 0.0)
                result["rerank_rank"] = i + 1
                result["reranking_method"] = method
                result["original_rank"] = results.index(result) + 1 if result in results else -1
            
            self.logger.info(f"Reranking completed in {processing_time:.3f}s")
            return reranked_results
            
        except Exception as e:
            self.logger.error(f"Reranking failed: {e}")
            return results
    
    async def cross_encoder_rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Rerank using cross-encoder model
        """
        try:
            # Mock cross-encoder implementation
            # In production, would use actual cross-encoder model
            
            reranked = []
            for result in results:
                # Calculate cross-encoder score (mock implementation)
                text = result.get("text", "")
                semantic_score = result.get("score", 0.0)
                
                # Mock cross-encoder calculation
                cross_encoder_score = self._calculate_cross_encoder_score(query, text, semantic_score)
                
                result["score"] = cross_encoder_score
                reranked.append(result)
            
            # Sort by cross-encoder score
            reranked.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            
            return reranked[:top_k]
            
        except Exception as e:
            self.logger.error(f"Cross-encoder reranking failed: {e}")
            return results[:top_k]
    
    async def learning_to_rank_rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int,
        user_context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank using learning-to-rank model
        """
        try:
            reranked = []
            
            for result in results:
                # Extract features
                features = self._extract_features(query, result, user_context)
                
                # Calculate LTR score (mock implementation)
                ltr_score = self._calculate_ltr_score(features)
                
                result["score"] = ltr_score
                result["features"] = features
                reranked.append(result)
            
            # Sort by LTR score
            reranked.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            
            return reranked[:top_k]
            
        except Exception as e:
            self.logger.error(f"Learning-to-rank reranking failed: {e}")
            return results[:top_k]
    
    async def _mmr_rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int,
        lambda_param: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Rerank using Maximal Marginal Relevance (MMR)
        """
        try:
            if not results:
                return []
            
            selected = [results[0]]  # Select first result
            remaining = results[1:]
            
            while len(selected) < top_k and remaining:
                best_idx = 0
                best_score = -1
                best_result = None
                
                for i, candidate in enumerate(remaining):
                    # Calculate MMR score
                    relevance_score = candidate.get("score", 0.0)
                    
                    # Calculate maximal marginal relevance
                    max_similarity = 0
                    for selected_result in selected:
                        similarity = self._calculate_similarity(candidate, selected_result)
                        max_similarity = max(max_similarity, similarity)
                    
                    mmr_score = lambda_param * relevance_score - (1 - lambda_param) * max_similarity
                    
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = i
                        best_result = candidate
                
                if best_result:
                    best_result["score"] = best_score
                    selected.append(best_result)
                    del remaining[best_idx]
            
            return selected
            
        except Exception as e:
            self.logger.error(f"MMR reranking failed: {e}")
            return results[:top_k]
    
    async def _diversification_rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Rerank for result diversity
        """
        try:
            if not results:
                return []
            
            # Cluster results by topic/content
            clusters = self._cluster_results(results)
            
            # Select diverse results from different clusters
            selected = []
            cluster_used = set()
            
            for result in results:
                cluster_id = result.get("cluster_id", 0)
                
                if cluster_id not in cluster_used and len(selected) < top_k:
                    selected.append(result)
                    cluster_used.add(cluster_id)
                elif len(selected) < top_k:
                    selected.append(result)
            
            return selected
            
        except Exception as e:
            self.logger.error(f"Diversification reranking failed: {e}")
            return results[:top_k]
    
    async def _temporal_bias_rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int,
        bias_type: str = "recent"
    ) -> List[Dict[str, Any]]:
        """
        Rerank with temporal bias
        """
        try:
            current_time = datetime.now()
            
            for result in results:
                metadata = result.get("metadata", {})
                timestamp = metadata.get("timestamp", "")
                
                if timestamp:
                    try:
                        doc_time = datetime.fromisoformat(timestamp)
                        age_days = (current_time - doc_time).days
                        
                        # Apply temporal bias
                        if bias_type == "recent":
                            if age_days < 30:
                                result["score"] *= 1.2  # Boost recent documents
                            elif age_days > 365:
                                result["score"] *= 0.8  # Penalize old documents
                        elif bias_type == "old":
                            if age_days > 365:
                                result["score"] *= 1.1  # Boost old documents
                            elif age_days < 30:
                                result["score"] *= 0.9  # Penalize recent documents
                    except:
                        pass  # Skip if timestamp parsing fails
            
            # Sort by boosted scores
            results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            
            return results[:top_k]
            
        except Exception as e:
            self.logger.error(f"Temporal bias reranking failed: {e}")
            return results[:top_k]
    
    async def _quality_weighted_rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Rerank based on document quality
        """
        try:
            for result in results:
                metadata = result.get("metadata", {})
                
                # Quality factors
                length_score = self._calculate_length_score(result.get("text", ""))
                source_score = self._calculate_source_score(metadata.get("source", ""))
                completeness_score = self._calculate_completeness_score(metadata)
                
                # Calculate quality score
                quality_score = (
                    length_score * 0.3 +
                    source_score * 0.4 +
                    completeness_score * 0.3
                )
                
                # Combine with original score
                original_score = result.get("score", 0.0)
                combined_score = 0.7 * original_score + 0.3 * quality_score
                
                result["score"] = combined_score
                result["quality_score"] = quality_score
            
            # Sort by combined score
            results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            
            return results[:top_k]
            
        except Exception as e:
            self.logger.error(f"Quality-weighted reranking failed: {e}")
            return results[:top_k]
    
    async def ensemble_rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        methods: List[str],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Ensemble multiple reranking methods
        """
        try:
            all_reranked = {}
            
            # Apply each reranking method
            for method in methods:
                if method in self.strategies:
                    reranked = await self.strategies[method](query, results, len(results))
                    all_reranked[method] = reranked
            
            # Combine scores from all methods
            combined_results = []
            
            for result in results:
                combined_score = 0
                method_count = 0
                
                for method, reranked in all_reranked.items():
                    for reranked_result in reranked:
                        if reranked_result.get("id") == result.get("id"):
                            combined_score += reranked_result.get("score", 0.0)
                            method_count += 1
                            break
                
                if method_count > 0:
                    result["score"] = combined_score / method_count
                    result["ensemble_methods"] = methods
                    combined_results.append(result)
            
            # Sort by combined score
            combined_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            
            return combined_results[:top_k]
            
        except Exception as e:
            self.logger.error(f"Ensemble reranking failed: {e}")
            return results[:top_k]
    
    # Helper methods
    def _calculate_cross_encoder_score(self, query: str, text: str, semantic_score: float) -> float:
        """Calculate cross-encoder score (mock implementation)"""
        # Mock calculation - in production, would use actual cross-encoder
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        
        # Word overlap
        overlap = len(query_words.intersection(text_words))
        total_words = len(query_words.union(text_words))
        
        if total_words == 0:
            overlap_score = 0.0
        else:
            overlap_score = overlap / total_words
        
        # Combine with semantic score
        cross_encoder_score = 0.6 * semantic_score + 0.4 * overlap_score
        
        return cross_encoder_score
    
    def _extract_features(
        self,
        query: str,
        result: Dict[str, Any],
        user_context: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Extract features for learning-to-rank"""
        features = {}
        
        # Semantic similarity
        features["semantic_similarity"] = result.get("score", 0.0)
        
        # Keyword match
        query_words = set(query.lower().split())
        text_words = set(result.get("text", "").lower().split())
        features["keyword_match"] = len(query_words.intersection(text_words)) / max(len(query_words), 1)
        
        # Document quality
        metadata = result.get("metadata", {})
        features["document_quality"] = self._calculate_quality_score(metadata)
        
        # Recency
        features["recency"] = self._calculate_recency_score(metadata.get("timestamp", ""))
        
        # Popularity
        features["popularity"] = self._calculate_popularity_score(metadata)
        
        # User context features
        if user_context:
            features["user_preference_match"] = self._calculate_preference_match(
                metadata, user_context
            )
        
        return features
    
    def _calculate_ltr_score(self, features: Dict[str, float]) -> float:
        """Calculate learning-to-rank score from features"""
        score = 0.0
        
        for feature, weight in self.feature_weights.items():
            if feature in features:
                score += features[feature] * weight
        
        return score
    
    def _calculate_similarity(self, doc1: Dict[str, Any], doc2: Dict[str, Any]) -> float:
        """Calculate similarity between two documents"""
        # Simple similarity calculation based on text overlap
        text1 = doc1.get("text", "").lower()
        text2 = doc2.get("text", "").lower()
        
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if len(union) == 0:
            return 0.0
        
        return len(intersection) / len(union)
    
    def _cluster_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Cluster results by topic/content"""
        # Mock clustering - in production, would use actual clustering algorithm
        clusters = {}
        
        for i, result in enumerate(results):
            # Simple clustering based on content type
            metadata = result.get("metadata", {})
            content_type = metadata.get("content_type", "general")
            
            if content_type not in clusters:
                clusters[content_type] = []
            
            clusters[content_type].append(result)
            result["cluster_id"] = hash(content_type) % 10  # Mock cluster ID
        
        return clusters
    
    def _calculate_length_score(self, text: str) -> float:
        """Calculate quality score based on text length"""
        length = len(text)
        
        # Optimal length is between 100 and 1000 characters
        if 100 <= length <= 1000:
            return 1.0
        elif length < 100:
            return length / 100
        else:
            return max(0.5, 1000 / length)
    
    def _calculate_source_score(self, source: str) -> float:
        """Calculate quality score based on source"""
        source_scores = {
            "official": 1.0,
            "reputable": 0.9,
            "user": 0.7,
            "unknown": 0.5
        }
        
        return source_scores.get(source.lower(), 0.5)
    
    def _calculate_completeness_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate completeness score based on metadata"""
        required_fields = ["title", "date", "author", "content"]
        present_fields = [field for field in required_fields if metadata.get(field)]
        
        return len(present_fields) / len(required_fields)
    
    def _calculate_quality_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate overall quality score"""
        return (
            self._calculate_length_score(metadata.get("text", "")) * 0.3 +
            self._calculate_source_score(metadata.get("source", "")) * 0.4 +
            self._calculate_completeness_score(metadata) * 0.3
        )
    
    def _calculate_recency_score(self, timestamp: str) -> float:
        """Calculate recency score"""
        if not timestamp:
            return 0.5
        
        try:
            doc_time = datetime.fromisoformat(timestamp)
            current_time = datetime.now()
            age_days = (current_time - doc_time).days
            
            # Recent documents get higher scores
            if age_days < 7:
                return 1.0
            elif age_days < 30:
                return 0.8
            elif age_days < 365:
                return 0.6
            else:
                return 0.4
        except:
            return 0.5
    
    def _calculate_popularity_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate popularity score"""
        # Mock popularity calculation
        views = metadata.get("views", 0)
        likes = metadata.get("likes", 0)
        
        if views == 0:
            return 0.5
        
        # Normalize popularity score
        popularity = (views + likes * 2) / 1000  # Normalize to 0-1 range
        return min(1.0, popularity)
    
    def _calculate_preference_match(
        self,
        metadata: Dict[str, Any],
        user_context: Dict[str, Any]
    ) -> float:
        """Calculate user preference match score"""
        preferences = user_context.get("preferences", {})
        match_score = 0.0
        
        # Check content type preference
        if "content_type" in preferences:
            if metadata.get("content_type") == preferences["content_type"]:
                match_score += 0.5
        
        # Check source preference
        if "source" in preferences:
            if metadata.get("source") == preferences["source"]:
                match_score += 0.3
        
        # Check topic preference
        if "topics" in preferences:
            topics = preferences["topics"]
            metadata_topics = metadata.get("topics", [])
            common_topics = set(topics).intersection(set(metadata_topics))
            if common_topics:
                match_score += len(common_topics) / len(topics) * 0.2
        
        return min(1.0, match_score)
    
    async def get_reranking_statistics(self) -> Dict[str, Any]:
        """Get reranking statistics"""
        return {
            "available_methods": list(self.strategies.keys()),
            "feature_weights": self.feature_weights,
            "default_method": "cross_encoder",
            "supported_strategies": [
                "cross_encoder",
                "learning_to_rank",
                "maximal_marginal_relevance",
                "diversification",
                "temporal_bias",
                "quality_weighted",
                "ensemble"
            ]
        }
