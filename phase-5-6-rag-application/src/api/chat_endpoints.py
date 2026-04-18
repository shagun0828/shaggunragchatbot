"""
Chat API Endpoints for Phase 5-6 Application
Conversational AI interface with context management
"""

from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import asyncio
import json
import logging

from integration.chroma_client import ChromaClient
from integration.llm_client import LLMClient
from advanced.context_manager import ContextManager
from personalization.user_profiler import UserProfiler

router = APIRouter()

# Pydantic models
class ChatMessage(BaseModel):
    message: str = Field(..., description="Chat message text")
    user_id: Optional[str] = Field(None, description="User ID")
    session_id: Optional[str] = Field(None, description="Session ID")
    message_type: str = Field("user", description="Message type (user/system/assistant)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    user_id: Optional[str] = Field(None, description="User ID")
    session_id: Optional[str] = Field(None, description="Session ID")
    context_length: int = Field(5, description="Number of previous messages to include in context")
    include_sources: bool = Field(True, description="Include source information")
    stream: bool = Field(False, description="Stream response")
    temperature: float = Field(0.7, description="LLM temperature")
    max_tokens: int = Field(1000, description="Maximum tokens in response")

class ChatResponse(BaseModel):
    message: str
    session_id: str
    user_id: Optional[str]
    sources: List[Dict[str, Any]]
    context_used: bool
    processing_time: float
    message_id: str
    metadata: Dict[str, Any]

class ConversationHistory(BaseModel):
    session_id: str
    user_id: Optional[str]
    messages: List[ChatMessage]
    total_messages: int
    created_at: str
    last_updated: str

class FeedbackRequest(BaseModel):
    message_id: str = Field(..., description="Message ID for feedback")
    rating: int = Field(..., ge=1, le=5, description="Rating (1-5)")
    feedback_type: str = Field("helpful", description="Type of feedback")
    feedback_text: Optional[str] = Field(None, description="Feedback text")

# Dependency injection
async def get_context_manager() -> ContextManager:
    """Get context manager instance"""
    return ContextManager()

async def get_user_profiler() -> UserProfiler:
    """Get user profiler instance"""
    return UserProfiler()

# Chat endpoints
@router.post("/chat", response_model=ChatResponse)
async def chat_message(
    request: ChatRequest,
    context_manager: ContextManager = Depends(get_context_manager),
    user_profiler: UserProfiler = Depends(get_user_profiler)
):
    """
    Process a chat message with RAG-enhanced responses
    """
    start_time = asyncio.get_event_loop().time()
    
    try:
        logging.info(f"Processing chat message: {request.message[:50]}...")
        
        # Generate session ID if not provided
        session_id = request.session_id or f"session_{asyncio.get_event_loop().time()}"
        
        # Get conversation context
        context_messages = await context_manager.get_context(
            session_id,
            request.user_id,
            limit=request.context_length
        )
        
        # Add current message to context
        current_message = ChatMessage(
            message=request.message,
            user_id=request.user_id,
            session_id=session_id,
            message_type="user"
        )
        
        # Generate message ID
        message_id = f"msg_{asyncio.get_event_loop().time()}_{hash(request.message)}"
        
        # Process with RAG if context suggests information retrieval
        context_used = False
        sources = []
        
        # Check if message requires information retrieval
        if await context_manager.requires_rag_retrieval(request.message, context_messages):
            # Use RAG to get relevant information
            from .rag_endpoints import process_single_query, get_chroma_client, get_llm_client, get_query_processor, get_reranker
            
            chroma_client = await get_chroma_client()
            llm_client = await get_llm_client()
            query_processor = await get_query_processor()
            reranker = await get_reranker()
            
            try:
                rag_result = await process_single_query(
                    request.message,
                    request.user_id,
                    5,  # top_k
                    chroma_client,
                    llm_client,
                    query_processor,
                    reranker
                )
                
                sources = rag_result.get("sources", [])
                context_used = True
                
                # Add RAG context to conversation
                rag_context = f"Based on relevant information: {rag_result.get('answer', '')}"
                context_messages.append({
                    "role": "system",
                    "content": rag_context
                })
                
            except Exception as e:
                logging.warning(f"RAG processing failed: {e}")
                # Continue without RAG
        
        # Generate response using LLM with context
        full_context = context_messages + [{"role": "user", "content": request.message}]
        
        # Personalize response if user_id provided
        if request.user_id:
            user_profile = await user_profiler.get_profile(request.user_id)
            if user_profile:
                # Add user preferences to context
                full_context.append({
                    "role": "system",
                    "content": f"User preferences: {user_profile.get('preferences', {})}"
                })
        
        # Generate LLM response
        llm_client = LLMClient()
        response_text = await llm_client.generate_chat_response(
            full_context,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        # Create assistant message
        assistant_message = ChatMessage(
            message=response_text,
            user_id=request.user_id,
            session_id=session_id,
            message_type="assistant",
            metadata={
                "message_id": message_id,
                "context_used": context_used,
                "sources_count": len(sources),
                "processing_time": processing_time
            }
        )
        
        # Store messages in context
        await context_manager.add_message(current_message)
        await context_manager.add_message(assistant_message)
        
        # Update user profile
        if request.user_id:
            await user_profiler.update_chat_activity(
                request.user_id,
                request.message,
                response_text,
                context_used
            )
        
        response = ChatResponse(
            message=response_text,
            session_id=session_id,
            user_id=request.user_id,
            sources=sources if request.include_sources else [],
            context_used=context_used,
            processing_time=processing_time,
            message_id=message_id,
            metadata={
                "context_length": len(context_messages),
                "temperature": request.temperature,
                "max_tokens": request.max_tokens
            }
        )
        
        logging.info(f"Chat message processed in {processing_time:.2f}s")
        return response
        
    except Exception as e:
        logging.error(f"Error processing chat message: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    context_manager: ContextManager = Depends(get_context_manager)
):
    """
    Stream chat response for real-time interaction
    """
    async def generate_stream():
        try:
            session_id = request.session_id or f"session_{asyncio.get_event_loop().time()}"
            
            # Get context
            context_messages = await context_manager.get_context(
                session_id,
                request.user_id,
                limit=request.context_length
            )
            
            # Add current message
            current_message = ChatMessage(
                message=request.message,
                user_id=request.user_id,
                session_id=session_id,
                message_type="user"
            )
            
            # Generate streaming response
            llm_client = LLMClient()
            
            async for chunk in llm_client.generate_streaming_response(
                context_messages + [{"role": "user", "content": request.message}],
                temperature=request.temperature,
                max_tokens=request.max_tokens
            ):
                yield f"data: {json.dumps({'chunk': chunk, 'type': 'message'})}\n\n"
            
            # Send completion signal
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
            # Store message in context
            assistant_message = ChatMessage(
                message="",  # Would be accumulated from stream
                user_id=request.user_id,
                session_id=session_id,
                message_type="assistant"
            )
            
            await context_manager.add_message(current_message)
            await context_manager.add_message(assistant_message)
            
        except Exception as e:
            logging.error(f"Error in chat stream: {e}")
            yield f"data: {json.dumps({'error': str(e), 'type': 'error'})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

@router.get("/chat/history/{session_id}", response_model=ConversationHistory)
async def get_chat_history(
    session_id: str,
    user_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    context_manager: ContextManager = Depends(get_context_manager)
):
    """
    Get chat history for a session
    """
    try:
        messages = await context_manager.get_session_history(
            session_id,
            user_id,
            limit=limit,
            offset=offset
        )
        
        return ConversationHistory(
            session_id=session_id,
            user_id=user_id,
            messages=messages,
            total_messages=len(messages),
            created_at="",  # Would be stored in context manager
            last_updated=""   # Would be stored in context manager
        )
        
    except Exception as e:
        logging.error(f"Error getting chat history: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.delete("/chat/history/{session_id}")
async def clear_chat_history(
    session_id: str,
    user_id: Optional[str] = None,
    context_manager: ContextManager = Depends(get_context_manager)
):
    """
    Clear chat history for a session
    """
    try:
        await context_manager.clear_session_history(session_id, user_id)
        
        return {"status": "success", "message": "Chat history cleared"}
        
    except Exception as e:
        logging.error(f"Error clearing chat history: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.post("/chat/feedback")
async def submit_chat_feedback(
    request: FeedbackRequest,
    user_profiler: UserProfiler = Depends(get_user_profiler)
):
    """
    Submit feedback for chat message
    """
    try:
        feedback_data = {
            "message_id": request.message_id,
            "rating": request.rating,
            "feedback_type": request.feedback_type,
            "feedback_text": request.feedback_text,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        # Store feedback (would integrate with user profiler)
        logging.info(f"Chat feedback submitted for message {request.message_id}: rating {request.rating}")
        
        return {"status": "success", "message": "Feedback submitted successfully"}
        
    except Exception as e:
        logging.error(f"Error submitting chat feedback: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/chat/sessions/{user_id}")
async def get_user_sessions(
    user_id: str,
    limit: int = 10,
    context_manager: ContextManager = Depends(get_context_manager)
):
    """
    Get all chat sessions for a user
    """
    try:
        sessions = await context_manager.get_user_sessions(user_id, limit)
        
        return {
            "user_id": user_id,
            "sessions": sessions,
            "total_sessions": len(sessions)
        }
        
    except Exception as e:
        logging.error(f"Error getting user sessions: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.post("/chat/suggestions")
async def get_chat_suggestions(
    message: str,
    user_id: Optional[str] = None,
    context_manager: ContextManager = Depends(get_context_manager)
):
    """
    Get chat suggestions based on current message and context
    """
    try:
        # Generate suggestions based on context
        suggestions = await context_manager.generate_suggestions(message, user_id)
        
        return {
            "message": message,
            "suggestions": suggestions,
            "user_id": user_id
        }
        
    except Exception as e:
        logging.error(f"Error generating chat suggestions: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

# WebSocket endpoint for real-time chat
@router.websocket("/ws/chat/{session_id}")
async def websocket_chat(
    websocket: WebSocket,
    session_id: str,
    context_manager: ContextManager = Depends(get_context_manager)
):
    """
    WebSocket endpoint for real-time chat
    """
    await websocket.accept()
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Process message
            chat_request = ChatRequest(
                message=message_data.get("message", ""),
                user_id=message_data.get("user_id"),
                session_id=session_id,
                context_length=message_data.get("context_length", 5),
                include_sources=message_data.get("include_sources", True),
                stream=False,
                temperature=message_data.get("temperature", 0.7),
                max_tokens=message_data.get("max_tokens", 1000)
            )
            
            # Generate response
            response = await chat_message(chat_request, context_manager)
            
            # Send response
            await websocket.send_text(json.dumps(response.dict()))
            
    except WebSocketDisconnect:
        logging.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
        await websocket.close()
