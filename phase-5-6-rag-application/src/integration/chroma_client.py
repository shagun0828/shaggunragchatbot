"""
Chroma Client Integration for Phase 5-6 Application
Handles vector database operations for RAG system
"""

import asyncio
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import aiohttp
import json
from datetime import datetime

# Import from Phase 4.3 for Chroma Cloud integration
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "phase-4.3-multi-model-v2" / "src"))
from simple_chroma_cloud import ChromaCloudManager


class ChromaClient:
    """Chroma client for vector database operations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.chroma_manager = None
        self.initialized = False
        self.session = None
        
    async def initialize(self):
        """Initialize Chroma client"""
        try:
            self.chroma_manager = ChromaCloudManager()
            self.session = aiohttp.ClientSession()
            self.initialized = True
            self.logger.info("Chroma client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Chroma client: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Chroma health status"""
        if not self.initialized:
            return {"status": "not_initialized", "error": "Client not initialized"}
        
        try:
            health = await self.chroma_manager.health_check()
            return health
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def search(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
        collection_name: str = "mutual_funds_v1"
    ) -> List[Dict[str, Any]]:
        """
        Search for documents using semantic similarity
        """
        if not self.initialized:
            raise RuntimeError("Chroma client not initialized")
        
        try:
            # Generate query embedding (mock implementation)
            query_embedding = await self._generate_query_embedding(query)
            
            # Search in Chroma
            results = await self.chroma_manager.search_embeddings(
                query_embedding,
                collection_name,
                top_k
            )
            
            # Filter by similarity threshold
            filtered_results = []
            for result in results:
                if result.get("distance", 1.0) <= (1.0 - similarity_threshold):
                    filtered_results.append(result)
            
            # Apply additional filters
            if filters:
                filtered_results = await self._apply_filters(filtered_results, filters)
            
            self.logger.info(f"Search completed: {len(filtered_results)} results for query: {query[:50]}...")
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            raise
    
    async def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search
        """
        return await self.search(query, top_k, similarity_threshold, filters, "mutual_funds_v1")
    
    async def keyword_search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform keyword-based search
        """
        if not self.initialized:
            raise RuntimeError("Chroma client not initialized")
        
        try:
            # For keyword search, we'll use semantic search as fallback
            # In production, this would use Elasticsearch or similar
            results = await self.search(query, top_k, 0.3, filters, "financial_news_v1")
            
            # Boost results that contain exact keyword matches
            query_lower = query.lower()
            for result in results:
                text = result.get("document", "").lower()
                if query_lower in text:
                    result["score"] *= 1.5  # Boost exact matches
            
            # Sort by boosted score
            results.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            return results[:top_k]
            
        except Exception as e:
            self.logger.error(f"Keyword search failed: {e}")
            raise
    
    async def find_similar_documents(
        self,
        document_id: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Find documents similar to a given document
        """
        if not self.initialized:
            raise RuntimeError("Chroma client not initialized")
        
        try:
            # Get the document by ID
            document = await self._get_document_by_id(document_id)
            if not document:
                return []
            
            # Use document text for similarity search
            text = document.get("text", "")
            similar_docs = await self.search(text, top_k, similarity_threshold)
            
            # Remove the original document from results
            similar_docs = [doc for doc in similar_docs if doc.get("id") != document_id]
            
            return similar_docs
            
        except Exception as e:
            self.logger.error(f"Find similar documents failed: {e}")
            raise
    
    async def find_similar_queries(
        self,
        query_id: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find queries similar to a given query
        """
        # This would search a query history collection
        # For now, return mock results
        return [
            {"query": f"Similar query 1 to {query_id}", "similarity": 0.85},
            {"query": f"Similar query 2 to {query_id}", "similarity": 0.78},
            {"query": f"Similar query 3 to {query_id}", "similarity": 0.72}
        ][:limit]
    
    async def get_autocomplete_suggestions(
        self,
        partial_query: str,
        limit: int = 5
    ) -> List[str]:
        """
        Get autocomplete suggestions for partial query
        """
        if not self.initialized:
            raise RuntimeError("Chroma client not initialized")
        
        try:
            # Search with low threshold to get related documents
            results = await self.search(partial_query, limit * 2, 0.2)
            
            # Extract key terms from results
            suggestions = set()
            for result in results:
                text = result.get("document", "")
                words = text.lower().split()
                for word in words:
                    if partial_query.lower() in word and len(word) > len(partial_query):
                        suggestions.add(word)
            
            # Add common completions
            common_completions = [
                f"{partial_query} mutual funds",
                f"{partial_query} investment",
                f"{partial_query} returns",
                f"{partial_query} performance"
            ]
            
            all_suggestions = list(suggestions) + common_completions
            return all_suggestions[:limit]
            
        except Exception as e:
            self.logger.error(f"Autocomplete failed: {e}")
            return []
    
    async def get_search_facets(
        self,
        query: str = ""
    ) -> Dict[str, Any]:
        """
        Get search facets for filtering
        """
        if not self.initialized:
            raise RuntimeError("Chroma client not initialized")
        
        try:
            # Mock facets - in production, this would aggregate metadata
            facets = {
                "fund_types": {
                    "equity": 45,
                    "debt": 23,
                    "hybrid": 18,
                    "elss": 12
                },
                "risk_levels": {
                    "low": 15,
                    "moderate": 38,
                    "high": 27,
                    "very_high": 20
                },
                "categories": {
                    "large_cap": 22,
                    "mid_cap": 18,
                    "small_cap": 15,
                    "multi_cap": 25
                },
                "time_periods": {
                    "1_year": 25,
                    "3_years": 20,
                    "5_years": 18,
                    "since_inception": 15
                }
            }
            
            return facets
            
        except Exception as e:
            self.logger.error(f"Get facets failed: {e}")
            return {}
    
    async def upsert_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: List[np.ndarray],
        collection_name: str = "mutual_funds_v1"
    ) -> bool:
        """
        Upsert documents with embeddings
        """
        if not self.initialized:
            raise RuntimeError("Chroma client not initialized")
        
        try:
            # Prepare documents for upload
            texts = [doc.get("text", "") for doc in documents]
            metadatas = [doc.get("metadata", {}) for doc in documents]
            ids = [doc.get("id", f"doc_{i}") for i, doc in enumerate(documents)]
            
            # Upload to Chroma
            collection_id = await self.chroma_manager.upload_embeddings(
                np.array(embeddings),
                texts,
                metadatas,
                collection_name
            )
            
            self.logger.info(f"Upserted {len(documents)} documents to {collection_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Upsert failed: {e}")
            return False
    
    async def delete_documents(
        self,
        document_ids: List[str],
        collection_name: str = "mutual_funds_v1"
    ) -> bool:
        """
        Delete documents by IDs
        """
        if not self.initialized:
            raise RuntimeError("Chroma client not initialized")
        
        try:
            # This would delete documents from Chroma
            # For now, return success
            self.logger.info(f"Deleted {len(document_ids)} documents from {collection_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Delete failed: {e}")
            return False
    
    async def get_collection_stats(
        self,
        collection_name: str
    ) -> Dict[str, Any]:
        """
        Get collection statistics
        """
        if not self.initialized:
            raise RuntimeError("Chroma client not initialized")
        
        try:
            stats = await self.chroma_manager.get_collection_stats(collection_name)
            return stats
            
        except Exception as e:
            self.logger.error(f"Get collection stats failed: {e}")
            return {}
    
    async def list_collections(self) -> List[str]:
        """
        List all available collections
        """
        if not self.initialized:
            raise RuntimeError("Chroma client not initialized")
        
        try:
            metrics = self.chroma_manager.get_metrics()
            return metrics.get("collections", [])
            
        except Exception as e:
            self.logger.error(f"List collections failed: {e}")
            return []
    
    async def close(self):
        """Close Chroma client"""
        if self.session:
            await self.session.close()
        
        if self.chroma_manager:
            # Close Chroma manager if needed
            pass
        
        self.initialized = False
        self.logger.info("Chroma client closed")
    
    # Helper methods
    async def _generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for query text"""
        # Mock implementation - in production, use actual embedding model
        # For now, generate random embedding based on query hash
        query_hash = hash(query)
        np.random.seed(query_hash % (2**32))
        embedding = np.random.rand(384)  # BGE-small dimension
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    async def _apply_filters(
        self,
        results: List[Dict[str, Any]],
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply filters to search results"""
        filtered_results = []
        
        for result in results:
            metadata = result.get("metadata", {})
            matches = True
            
            for key, value in filters.items():
                if key in metadata:
                    if isinstance(value, list):
                        if metadata[key] not in value:
                            matches = False
                            break
                    else:
                        if metadata[key] != value:
                            matches = False
                            break
                else:
                    matches = False
                    break
            
            if matches:
                filtered_results.append(result)
        
        return filtered_results
    
    async def _get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID"""
        # Mock implementation - in production, query Chroma by ID
        return {
            "id": document_id,
            "text": f"Sample document content for {document_id}",
            "metadata": {"source": "test", "type": "document"}
        }


class VectorSearchEngine:
    """Advanced vector search engine with multiple strategies"""
    
    def __init__(self, chroma_client: ChromaClient):
        self.chroma_client = chroma_client
        self.logger = logging.getLogger(__name__)
    
    async def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and keyword search
        """
        try:
            # Get semantic results
            semantic_results = await self.chroma_client.semantic_search(
                query, top_k * 2, 0.5, filters
            )
            
            # Get keyword results
            keyword_results = await self.chroma_client.keyword_search(
                query, top_k * 2, filters
            )
            
            # Combine and weight results
            combined_results = []
            
            # Add semantic results with weight
            for result in semantic_results:
                result["combined_score"] = result.get("score", 0) * semantic_weight
                result["source"] = "semantic"
                combined_results.append(result)
            
            # Add keyword results with weight
            for result in keyword_results:
                result["combined_score"] = result.get("score", 0) * keyword_weight
                result["source"] = "keyword"
                combined_results.append(result)
            
            # Remove duplicates and sort by combined score
            seen_ids = set()
            unique_results = []
            
            for result in combined_results:
                doc_id = result.get("id", "")
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    unique_results.append(result)
            
            # Sort by combined score and return top_k
            unique_results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
            
            return unique_results[:top_k]
            
        except Exception as e:
            self.logger.error(f"Hybrid search failed: {e}")
            raise
    
    async def multi_vector_search(
        self,
        query: str,
        collections: List[str],
        top_k: int = 10
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search across multiple vector collections
        """
        results = {}
        
        for collection in collections:
            try:
                collection_results = await self.chroma_client.search(
                    query, top_k, 0.5, None, collection
                )
                results[collection] = collection_results
            except Exception as e:
                self.logger.error(f"Search in {collection} failed: {e}")
                results[collection] = []
        
        return results
    
    async def temporal_search(
        self,
        query: str,
        time_range: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Perform time-aware search
        """
        # Calculate time filter
        current_time = datetime.now()
        
        if time_range == "recent":
            # Last 30 days
            filter_date = current_time - timedelta(days=30)
        elif time_range == "month":
            # Last month
            filter_date = current_time - timedelta(days=30)
        elif time_range == "year":
            # Last year
            filter_date = current_time - timedelta(days=365)
        else:
            filter_date = None
        
        filters = {}
        if filter_date:
            filters["created_after"] = filter_date.isoformat()
        
        return await self.chroma_client.search(query, top_k, 0.5, filters)
