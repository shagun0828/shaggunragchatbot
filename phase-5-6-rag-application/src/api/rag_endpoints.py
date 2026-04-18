"""
RAG API Endpoints for Phase 5-6 Application
Core RAG functionality with advanced features
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import asyncio
import logging

from integration.chroma_client import ChromaClient
from integration.llm_client import LLMClient
from advanced.query_processor import QueryProcessor
from advanced.reranker import Reranker
from personalization.user_profiler import UserProfiler

router = APIRouter()

# Pydantic models
class QueryRequest(BaseModel):
    query: str = Field(..., description="User query text")
    user_id: Optional[str] = Field(None, description="User ID for personalization")
    session_id: Optional[str] = Field(None, description="Session ID for context")
    top_k: int = Field(5, description="Number of results to retrieve")
    similarity_threshold: float = Field(0.7, description="Similarity threshold")
    use_reranking: bool = Field(True, description="Enable result reranking")
    include_sources: bool = Field(True, description="Include source information")
    stream_response: bool = Field(False, description="Stream response")

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    processing_time: float
    user_id: Optional[str]
    session_id: Optional[str]
    metadata: Dict[str, Any]

class BatchQueryRequest(BaseModel):
    queries: List[str] = Field(..., description="List of queries to process")
    user_id: Optional[str] = Field(None, description="User ID for personalization")
    top_k: int = Field(5, description="Number of results per query")
    parallel: bool = Field(True, description="Process queries in parallel")

class BatchQueryResponse(BaseModel):
    results: List[QueryResponse]
    total_processing_time: float
    queries_processed: int
    metadata: Dict[str, Any]

class FeedbackRequest(BaseModel):
    query_id: str = Field(..., description="Query ID for feedback")
    rating: int = Field(..., ge=1, le=5, description="Rating (1-5)")
    feedback_text: Optional[str] = Field(None, description="Feedback text")
    user_id: Optional[str] = Field(None, description="User ID")

# Dependency injection
async def get_chroma_client() -> ChromaClient:
    """Get Chroma client instance"""
    # This would be injected from the main app
    return ChromaClient()

async def get_llm_client() -> LLMClient:
    """Get LLM client instance"""
    return LLMClient()

async def get_query_processor() -> QueryProcessor:
    """Get query processor instance"""
    return QueryProcessor()

async def get_reranker() -> Reranker:
    """Get reranker instance"""
    return Reranker()

async def get_user_profiler() -> UserProfiler:
    """Get user profiler instance"""
    return UserProfiler()

# Core RAG endpoint
@router.post("/query", response_model=QueryResponse)
async def query_rag(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    chroma_client: ChromaClient = Depends(get_chroma_client),
    llm_client: LLMClient = Depends(get_llm_client),
    query_processor: QueryProcessor = Depends(get_query_processor),
    reranker: Reranker = Depends(get_reranker),
    user_profiler: UserProfiler = Depends(get_user_profiler)
):
    """
    Process a RAG query with advanced features
    """
    start_time = asyncio.get_event_loop().time()
    
    try:
        logging.info(f"Processing RAG query: {request.query[:50]}...")
        
        # Step 1: Query preprocessing and expansion
        processed_query = await query_processor.process_query(
            request.query,
            user_id=request.user_id,
            session_id=request.session_id
        )
        
        # Step 2: Retrieve relevant documents
        retrieved_docs = await chroma_client.search(
            processed_query["expanded_query"],
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            filters=processed_query.get("filters", {})
        )
        
        if not retrieved_docs:
            raise HTTPException(
                status_code=404,
                detail="No relevant documents found"
            )
        
        # Step 3: Rerank results if enabled
        if request.use_reranking:
            retrieved_docs = await reranker.rerank(
                processed_query["original_query"],
                retrieved_docs
            )
        
        # Step 4: Generate response
        context = [doc["text"] for doc in retrieved_docs[:request.top_k]]
        
        # Add personalization if user_id is provided
        if request.user_id:
            user_profile = await user_profiler.get_profile(request.user_id)
            context = await user_profiler.personalize_context(context, user_profile)
        
        # Generate LLM response
        if request.stream_response:
            # Streaming response (would need to be handled differently)
            response_text = await llm_client.generate_response(
                processed_query["original_query"],
                context,
                stream=False
            )
        else:
            response_text = await llm_client.generate_response(
                processed_query["original_query"],
                context,
                stream=False
            )
        
        # Step 5: Prepare response
        processing_time = asyncio.get_event_loop().time() - start_time
        
        sources = []
        if request.include_sources:
            sources = [
                {
                    "id": doc.get("id", ""),
                    "text": doc.get("text", "")[:200] + "...",
                    "metadata": doc.get("metadata", {}),
                    "score": doc.get("score", 0.0)
                }
                for doc in retrieved_docs
            ]
        
        response = QueryResponse(
            query=request.query,
            answer=response_text,
            sources=sources,
            confidence=min(0.95, max(0.1, 1.0 - (len(retrieved_docs) / request.top_k))),
            processing_time=processing_time,
            user_id=request.user_id,
            session_id=request.session_id,
            metadata={
                "retrieved_docs": len(retrieved_docs),
                "query_expansion": processed_query.get("expanded", False),
                "reranking_used": request.use_reranking,
                "personalization": request.user_id is not None
            }
        )
        
        # Background task: Log query for analytics
        background_tasks.add_task(
            log_query_analytics,
            request.query,
            response,
            processing_time
        )
        
        # Background task: Update user profile if user_id provided
        if request.user_id:
            background_tasks.add_task(
                user_profiler.update_profile,
                request.user_id,
                request.query,
                retrieved_docs
            )
        
        logging.info(f"RAG query processed successfully in {processing_time:.2f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error processing RAG query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.post("/batch-query", response_model=BatchQueryResponse)
async def batch_query_rag(
    request: BatchQueryRequest,
    background_tasks: BackgroundTasks,
    chroma_client: ChromaClient = Depends(get_chroma_client),
    llm_client: LLMClient = Depends(get_llm_client),
    query_processor: QueryProcessor = Depends(get_query_processor),
    reranker: Reranker = Depends(get_reranker)
):
    """
    Process multiple RAG queries in batch
    """
    start_time = asyncio.get_event_loop().time()
    
    try:
        logging.info(f"Processing batch of {len(request.queries)} queries")
        
        if request.parallel:
            # Process queries in parallel
            tasks = []
            for query in request.queries:
                task = asyncio.create_task(
                    process_single_query(
                        query,
                        request.user_id,
                        request.top_k,
                        chroma_client,
                        llm_client,
                        query_processor,
                        reranker
                    )
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Process queries sequentially
            results = []
            for query in request.queries:
                try:
                    result = await process_single_query(
                        query,
                        request.user_id,
                        request.top_k,
                        chroma_client,
                        llm_client,
                        query_processor,
                        reranker
                    )
                    results.append(result)
                except Exception as e:
                    results.append({"error": str(e)})
        
        # Filter out errors and create response
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logging.error(f"Error processing query {i}: {result}")
                continue
            
            successful_results.append(result)
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        response = BatchQueryResponse(
            results=successful_results,
            total_processing_time=processing_time,
            queries_processed=len(successful_results),
            metadata={
                "total_queries": len(request.queries),
                "parallel_processing": request.parallel,
                "success_rate": len(successful_results) / len(request.queries)
            }
        )
        
        # Background task: Log batch query analytics
        background_tasks.add_task(
            log_batch_query_analytics,
            request.queries,
            response,
            processing_time
        )
        
        logging.info(f"Batch query processed: {len(successful_results)}/{len(request.queries)} successful")
        return response
        
    except Exception as e:
        logging.error(f"Error processing batch query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.post("/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    user_profiler: UserProfiler = Depends(get_user_profiler)
):
    """
    Submit feedback for query results
    """
    try:
        # Store feedback
        feedback_data = {
            "query_id": request.query_id,
            "rating": request.rating,
            "feedback_text": request.feedback_text,
            "user_id": request.user_id,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        # Update user profile with feedback
        if request.user_id:
            await user_profiler.add_feedback(request.user_id, feedback_data)
        
        logging.info(f"Feedback submitted for query {request.query_id}: rating {request.rating}")
        
        return {"status": "success", "message": "Feedback submitted successfully"}
        
    except Exception as e:
        logging.error(f"Error submitting feedback: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/similar/{query_id}")
async def get_similar_queries(
    query_id: str,
    limit: int = 5,
    chroma_client: ChromaClient = Depends(get_chroma_client)
):
    """
    Get similar queries based on query embedding
    """
    try:
        # Find similar queries
        similar_queries = await chroma_client.find_similar_queries(
            query_id,
            limit=limit
        )
        
        return {
            "query_id": query_id,
            "similar_queries": similar_queries,
            "limit": limit
        }
        
    except Exception as e:
        logging.error(f"Error finding similar queries: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/history/{user_id}")
async def get_query_history(
    user_id: str,
    limit: int = 10,
    offset: int = 0,
    user_profiler: UserProfiler = Depends(get_user_profiler)
):
    """
    Get query history for a user
    """
    try:
        history = await user_profiler.get_query_history(
            user_id,
            limit=limit,
            offset=offset
        )
        
        return {
            "user_id": user_id,
            "history": history,
            "limit": limit,
            "offset": offset,
            "total": len(history)
        }
        
    except Exception as e:
        logging.error(f"Error getting query history: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

# Helper functions
async def process_single_query(
    query: str,
    user_id: Optional[str],
    top_k: int,
    chroma_client: ChromaClient,
    llm_client: LLMClient,
    query_processor: QueryProcessor,
    reranker: Reranker
) -> Dict[str, Any]:
    """Process a single query for batch processing"""
    processed_query = await query_processor.process_query(query, user_id)
    
    retrieved_docs = await chroma_client.search(
        processed_query["expanded_query"],
        top_k=top_k
    )
    
    if retrieved_docs and reranker:
        retrieved_docs = await reranker.rerank(query, retrieved_docs)
    
    context = [doc["text"] for doc in retrieved_docs[:top_k]]
    
    response_text = await llm_client.generate_response(query, context)
    
    return QueryResponse(
        query=query,
        answer=response_text,
        sources=[{"id": doc.get("id", ""), "text": doc.get("text", "")[:200] + "..."} for doc in retrieved_docs],
        confidence=0.8,
        processing_time=0.0,  # Would be calculated in actual implementation
        user_id=user_id,
        session_id=None,
        metadata={}
    ).dict()

async def log_query_analytics(query: str, response: QueryResponse, processing_time: float):
    """Log query analytics in background"""
    # This would log to analytics database
    logging.info(f"Query analytics logged: {query[:50]}...")

async def log_batch_query_analytics(queries: List[str], response: BatchQueryResponse, processing_time: float):
    """Log batch query analytics in background"""
    # This would log to analytics database
    logging.info(f"Batch query analytics logged: {len(queries)} queries")
