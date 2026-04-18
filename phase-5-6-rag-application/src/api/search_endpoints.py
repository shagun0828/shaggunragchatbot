"""
Search API Endpoints for Phase 5-6 Application
AI-enhanced search with semantic capabilities
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import asyncio
import logging

from integration.chroma_client import ChromaClient
from integration.llm_client import LLMClient
from advanced.query_processor import QueryProcessor
from advanced.reranker import Reranker

router = APIRouter()

# Pydantic models
class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    search_type: str = Field("semantic", description="Search type: semantic, keyword, hybrid")
    top_k: int = Field(10, description="Number of results to return")
    similarity_threshold: float = Field(0.5, description="Similarity threshold for semantic search")
    filters: Optional[Dict[str, Any]] = Field(None, description="Search filters")
    include_metadata: bool = Field(True, description="Include document metadata")
    use_reranking: bool = Field(True, description="Enable result reranking")
    user_id: Optional[str] = Field(None, description="User ID for personalization")

class SearchResult(BaseModel):
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]
    highlights: List[str]

class SearchResponse(BaseModel):
    query: str
    search_type: str
    results: List[SearchResult]
    total_results: int
    processing_time: float
    suggestions: List[str]
    metadata: Dict[str, Any]

class AdvancedSearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    search_strategy: str = Field("hybrid", description="Search strategy: semantic, keyword, hybrid, multi_modal")
    filters: Optional[Dict[str, Any]] = Field(None, description="Advanced filters")
    ranking_model: str = Field("cross_encoder", description="Ranking model: cross_encoder, learning_to_rank")
    diversity_threshold: float = Field(0.8, description="Maximal marginal relevance threshold")
    temporal_bias: str = Field("none", description="Temporal bias: none, recent, old")
    user_context: Optional[Dict[str, Any]] = Field(None, description="User context for personalization")
    top_k: int = Field(20, description="Number of results")
    explain_results: bool = Field(False, description="Include explanations")

class SearchExplanation(BaseModel):
    query: str
    explanation: str
    reasoning: List[str]
    confidence: float

class AdvancedSearchResponse(BaseModel):
    query: str
    search_strategy: str
    results: List[SearchResult]
    explanations: List[SearchExplanation]
    total_results: int
    processing_time: float
    metadata: Dict[str, Any]

# Dependency injection
async def get_chroma_client() -> ChromaClient:
    """Get Chroma client instance"""
    return ChromaClient()

async def get_query_processor() -> QueryProcessor:
    """Get query processor instance"""
    return QueryProcessor()

async def get_reranker() -> Reranker:
    """Get reranker instance"""
    return Reranker()

@router.post("/search", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    chroma_client: ChromaClient = Depends(get_chroma_client),
    query_processor: QueryProcessor = Depends(get_query_processor),
    reranker: Reranker = Depends(get_reranker)
):
    """
    Search documents with AI-enhanced capabilities
    """
    start_time = asyncio.get_event_loop().time()
    
    try:
        logging.info(f"Searching documents: {request.query[:50]}...")
        
        # Step 1: Query preprocessing
        processed_query = await query_processor.process_query(
            request.query,
            user_id=request.user_id
        )
        
        # Step 2: Perform search based on type
        if request.search_type == "semantic":
            results = await chroma_client.semantic_search(
                processed_query["expanded_query"],
                top_k=request.top_k,
                similarity_threshold=request.similarity_threshold,
                filters=request.filters
            )
        elif request.search_type == "keyword":
            results = await chroma_client.keyword_search(
                processed_query["expanded_query"],
                top_k=request.top_k,
                filters=request.filters
            )
        else:  # hybrid
            semantic_results = await chroma_client.semantic_search(
                processed_query["expanded_query"],
                top_k=request.top_k * 2,
                similarity_threshold=request.similarity_threshold * 0.8,
                filters=request.filters
            )
            
            keyword_results = await chroma_client.keyword_search(
                processed_query["expanded_query"],
                top_k=request.top_k * 2,
                filters=request.filters
            )
            
            # Combine and deduplicate results
            results = await combine_search_results(semantic_results, keyword_results, request.top_k)
        
        # Step 3: Rerank results if enabled
        if request.use_reranking and results:
            results = await reranker.rerank(processed_query["original_query"], results)
        
        # Step 4: Format results
        formatted_results = []
        for result in results:
            formatted_result = SearchResult(
                id=result.get("id", ""),
                text=result.get("text", ""),
                score=result.get("score", 0.0),
                metadata=result.get("metadata", {}) if request.include_metadata else {},
                highlights=generate_highlights(result.get("text", ""), processed_query["original_query"])
            )
            formatted_results.append(formatted_result)
        
        # Step 5: Generate suggestions
        suggestions = await generate_search_suggestions(
            processed_query["original_query"],
            formatted_results,
            chroma_client
        )
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        response = SearchResponse(
            query=request.query,
            search_type=request.search_type,
            results=formatted_results,
            total_results=len(formatted_results),
            processing_time=processing_time,
            suggestions=suggestions,
            metadata={
                "expanded_query": processed_query.get("expanded_query"),
                "reranking_used": request.use_reranking,
                "filters_applied": request.filters is not None
            }
        )
        
        logging.info(f"Search completed: {len(formatted_results)} results in {processing_time:.2f}s")
        return response
        
    except Exception as e:
        logging.error(f"Error during search: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.post("/search/advanced", response_model=AdvancedSearchResponse)
async def advanced_search(
    request: AdvancedSearchRequest,
    chroma_client: ChromaClient = Depends(get_chroma_client),
    query_processor: QueryProcessor = Depends(get_query_processor),
    reranker: Reranker = Depends(get_reranker)
):
    """
    Advanced search with multiple strategies and explanations
    """
    start_time = asyncio.get_event_loop().time()
    
    try:
        logging.info(f"Advanced search: {request.query[:50]}...")
        
        # Step 1: Query analysis and expansion
        processed_query = await query_processor.process_query(
            request.query,
            user_id=request.user_context.get("user_id") if request.user_context else None
        )
        
        # Step 2: Multi-strategy search
        all_results = {}
        
        if "semantic" in request.search_strategy:
            all_results["semantic"] = await chroma_client.semantic_search(
                processed_query["expanded_query"],
                top_k=request.top_k,
                filters=request.filters
            )
        
        if "keyword" in request.search_strategy:
            all_results["keyword"] = await chroma_client.keyword_search(
                processed_query["expanded_query"],
                top_k=request.top_k,
                filters=request.filters
            )
        
        if "hybrid" in request.search_strategy:
            # Combine semantic and keyword results
            semantic = all_results.get("semantic", [])
            keyword = all_results.get("keyword", [])
            all_results["hybrid"] = await combine_search_results(semantic, keyword, request.top_k)
        
        # Step 3: Apply temporal bias
        if request.temporal_bias != "none":
            for strategy in all_results:
                all_results[strategy] = apply_temporal_bias(
                    all_results[strategy],
                    request.temporal_bias
                )
        
        # Step 4: Apply diversity filter (Maximal Marginal Relevance)
        if request.diversity_threshold < 1.0:
            for strategy in all_results:
                all_results[strategy] = apply_diversity_filter(
                    all_results[strategy],
                    request.diversity_threshold
                )
        
        # Step 5: Advanced reranking
        if request.ranking_model == "cross_encoder":
            for strategy in all_results:
                all_results[strategy] = await reranker.cross_encoder_rerank(
                    processed_query["original_query"],
                    all_results[strategy]
                )
        elif request.ranking_model == "learning_to_rank":
            for strategy in all_results:
                all_results[strategy] = await reranker.learning_to_rank(
                    processed_query["original_query"],
                    all_results[strategy],
                    request.user_context
                )
        
        # Step 6: Select best strategy results
        best_strategy = max(all_results.keys(), key=lambda k: len(all_results[k]))
        final_results = all_results[best_strategy][:request.top_k]
        
        # Step 7: Generate explanations
        explanations = []
        if request.explain_results:
            explanations = await generate_search_explanations(
                processed_query["original_query"],
                final_results,
                best_strategy,
                request.search_strategy
            )
        
        # Step 8: Format results
        formatted_results = []
        for result in final_results:
            formatted_result = SearchResult(
                id=result.get("id", ""),
                text=result.get("text", ""),
                score=result.get("score", 0.0),
                metadata=result.get("metadata", {}),
                highlights=generate_highlights(result.get("text", ""), processed_query["original_query"])
            )
            formatted_results.append(formatted_result)
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        response = AdvancedSearchResponse(
            query=request.query,
            search_strategy=best_strategy,
            results=formatted_results,
            explanations=explanations,
            total_results=len(formatted_results),
            processing_time=processing_time,
            metadata={
                "all_strategies": list(all_results.keys()),
                "strategy_results": {k: len(v) for k, v in all_results.items()},
                "temporal_bias": request.temporal_bias,
                "diversity_threshold": request.diversity_threshold,
                "ranking_model": request.ranking_model
            }
        )
        
        logging.info(f"Advanced search completed: {len(formatted_results)} results in {processing_time:.2f}s")
        return response
        
    except Exception as e:
        logging.error(f"Error during advanced search: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/search/similar/{document_id}")
async def find_similar_documents(
    document_id: str,
    top_k: int = Query(5, description="Number of similar documents"),
    similarity_threshold: float = Query(0.7, description="Similarity threshold"),
    chroma_client: ChromaClient = Depends(get_chroma_client)
):
    """
    Find documents similar to a given document
    """
    try:
        similar_docs = await chroma_client.find_similar_documents(
            document_id,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
        
        formatted_results = []
        for doc in similar_docs:
            formatted_result = SearchResult(
                id=doc.get("id", ""),
                text=doc.get("text", ""),
                score=doc.get("score", 0.0),
                metadata=doc.get("metadata", {}),
                highlights=[]
            )
            formatted_results.append(formatted_result)
        
        return {
            "document_id": document_id,
            "similar_documents": formatted_results,
            "total_results": len(formatted_results)
        }
        
    except Exception as e:
        logging.error(f"Error finding similar documents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/search/autocomplete")
async def autocomplete_search(
    query: str = Query(..., description="Partial query for autocomplete"),
    limit: int = Query(5, description="Number of suggestions"),
    chroma_client: ChromaClient = Depends(get_chroma_client)
):
    """
    Get autocomplete suggestions for search query
    """
    try:
        suggestions = await chroma_client.get_autocomplete_suggestions(query, limit)
        
        return {
            "query": query,
            "suggestions": suggestions,
            "limit": limit
        }
        
    except Exception as e:
        logging.error(f"Error generating autocomplete: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/search/facets")
async def get_search_facets(
    query: str = Query("", description="Query to filter facets"),
    chroma_client: ChromaClient = Depends(get_chroma_client)
):
    """
    Get search facets for filtering
    """
    try:
        facets = await chroma_client.get_search_facets(query)
        
        return {
            "query": query,
            "facets": facets
        }
        
    except Exception as e:
        logging.error(f"Error getting search facets: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

# Helper functions
async def combine_search_results(
    semantic_results: List[Dict],
    keyword_results: List[Dict],
    top_k: int
) -> List[Dict]:
    """Combine semantic and keyword search results"""
    # Simple combination - would use more sophisticated merging in production
    combined = {}
    
    # Add semantic results
    for result in semantic_results:
        doc_id = result.get("id", "")
        combined[doc_id] = result
        combined[doc_id]["semantic_score"] = result.get("score", 0.0)
    
    # Add keyword results
    for result in keyword_results:
        doc_id = result.get("id", "")
        if doc_id in combined:
            # Average scores
            combined[doc_id]["score"] = (
                combined[doc_id].get("score", 0.0) + result.get("score", 0.0)
            ) / 2
            combined[doc_id]["keyword_score"] = result.get("score", 0.0)
        else:
            combined[doc_id] = result
            combined[doc_id]["keyword_score"] = result.get("score", 0.0)
    
    # Sort by combined score and return top_k
    sorted_results = sorted(
        combined.values(),
        key=lambda x: x.get("score", 0.0),
        reverse=True
    )
    
    return sorted_results[:top_k]

def generate_highlights(text: str, query: str) -> List[str]:
    """Generate highlights for search results"""
    # Simple highlight generation - would use more sophisticated approach
    query_terms = query.lower().split()
    text_lower = text.lower()
    
    highlights = []
    for term in query_terms:
        if term in text_lower:
            start_idx = text_lower.find(term)
            if start_idx != -1:
                # Extract context around the match
                context_start = max(0, start_idx - 50)
                context_end = min(len(text), start_idx + len(term) + 50)
                highlight = text[context_start:context_end]
                highlights.append(highlight)
    
    return highlights[:3]  # Return top 3 highlights

async def generate_search_suggestions(
    query: str,
    results: List[SearchResult],
    chroma_client: ChromaClient
) -> List[str]:
    """Generate search suggestions based on query and results"""
    suggestions = []
    
    # Add related queries from results metadata
    for result in results[:3]:
        metadata = result.metadata
        if "related_queries" in metadata:
            suggestions.extend(metadata["related_queries"])
    
    # Add common query expansions
    if len(results) < 5:
        # Suggest broader terms if few results
        suggestions.append(f"{query} overview")
        suggestions.append(f"{query} guide")
    elif len(results) > 15:
        # Suggest more specific terms if many results
        suggestions.append(f"{query} best practices")
        suggestions.append(f"{query} examples")
    
    return list(set(suggestions))[:5]  # Return unique suggestions

def apply_temporal_bias(results: List[Dict], bias_type: str) -> List[Dict]:
    """Apply temporal bias to search results"""
    current_time = asyncio.get_event_loop().time()
    
    for result in results:
        metadata = result.get("metadata", {})
        timestamp = metadata.get("timestamp", 0)
        
        if bias_type == "recent":
            # Boost recent documents
            age_days = (current_time - timestamp) / (24 * 3600)
            if age_days < 30:
                result["score"] *= 1.2  # Boost recent docs
        elif bias_type == "old":
            # Boost older documents
            age_days = (current_time - timestamp) / (24 * 3600)
            if age_days > 365:
                result["score"] *= 1.1  # Boost old docs
    
    return results

def apply_diversity_filter(results: List[Dict], threshold: float) -> List[Dict]:
    """Apply maximal marginal relevance for diversity"""
    if not results:
        return results
    
    selected = [results[0]]
    
    for result in results[1:]:
        # Check similarity with already selected results
        max_similarity = 0
        for selected_result in selected:
            # Simple similarity check - would use actual similarity in production
            similarity = calculate_content_similarity(
                result.get("text", ""),
                selected_result.get("text", "")
            )
            max_similarity = max(max_similarity, similarity)
        
        # Include if diverse enough
        if max_similarity < threshold:
            selected.append(result)
    
    return selected

def calculate_content_similarity(text1: str, text2: str) -> float:
    """Calculate content similarity between two texts"""
    # Simple similarity calculation - would use actual embedding similarity
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0

async def generate_search_explanations(
    query: str,
    results: List[SearchResult],
    strategy: str,
    search_strategy: str
) -> List[SearchExplanation]:
    """Generate explanations for search results"""
    explanations = []
    
    for i, result in enumerate(results[:3]):  # Explain top 3 results
        explanation_text = f"This result was found using {strategy} search"
        reasoning = [
            f"Query '{query}' matched content with {result.score:.2f} similarity score",
            f"Search strategy: {search_strategy}",
            f"Content relevance: High"
        ]
        
        explanation = SearchExplanation(
            query=query,
            explanation=explanation_text,
            reasoning=reasoning,
            confidence=result.score
        )
        explanations.append(explanation)
    
    return explanations
