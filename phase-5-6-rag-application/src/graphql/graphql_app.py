"""
GraphQL Application for Phase 5-6
Flexible query interface with GraphQL
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
import strawberry
from strawberry import Schema
from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import Response
import json

# Import components
from integration.chroma_client import ChromaClient
from integration.llm_client import LLMClient
from advanced.query_processor import QueryProcessor
from advanced.reranker import Reranker
from personalization.user_profiler import UserProfiler


@strawberry.type
class DocumentResult:
    """GraphQL type for document results"""
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]
    highlights: List[str]


@strawberry.type
class SearchResult:
    """GraphQL type for search results"""
    query: str
    search_type: str
    results: List[DocumentResult]
    total_results: int
    processing_time: float
    suggestions: List[str]


@strawberry.type
class ChatMessage:
    """GraphQL type for chat messages"""
    message_id: str
    user_id: Optional[str]
    session_id: str
    message_type: str
    content: str
    timestamp: float
    metadata: Dict[str, Any]


@strawberry.type
class ChatResponse:
    """GraphQL type for chat responses"""
    message: str
    session_id: str
    user_id: Optional[str]
    sources: List[DocumentResult]
    context_used: bool
    processing_time: float
    message_id: str


@strawberry.type
class UserProfile:
    """GraphQL type for user profiles"""
    user_id: str
    risk_tolerance: str
    investment_horizon: str
    preferred_fund_types: List[str]
    expertise_level: str
    total_queries: int
    last_activity: float


@strawberry.type
class SystemMetrics:
    """GraphQL type for system metrics"""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    active_connections: int
    request_count: int
    error_count: float


@strawberry.input
class SearchInput:
    """GraphQL input for search queries"""
    query: str
    search_type: str = "semantic"
    top_k: int = 10
    similarity_threshold: float = 0.7
    filters: Optional[Dict[str, Any]] = None
    use_reranking: bool = True
    user_id: Optional[str] = None


@strawberry.input
class ChatInput:
    """GraphQL input for chat messages"""
    message: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    context_length: int = 5
    include_sources: bool = True
    temperature: float = 0.7
    max_tokens: int = 1000


@strawberry.input
class FeedbackInput:
    """GraphQL input for feedback"""
    message_id: str
    rating: int
    feedback_type: str = "helpful"
    feedback_text: Optional[str] = None
    user_id: Optional[str] = None


class GraphQLResolver:
    """GraphQL resolvers for Phase 5-6"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.chroma_client = ChromaClient()
        self.llm_client = LLMClient()
        self.query_processor = QueryProcessor()
        self.reranker = Reranker()
        self.user_profiler = UserProfiler()
        self.initialized = False
    
    async def initialize(self):
        """Initialize GraphQL resolvers"""
        if not self.initialized:
            await self.chroma_client.initialize()
            await self.llm_client.initialize()
            self.initialized = True
            self.logger.info("GraphQL resolvers initialized")
    
    @strawberry.field
    async def search_documents(self, input: SearchInput) -> SearchResult:
        """Search documents with flexible parameters"""
        await self.initialize()
        
        try:
            # Process query
            processed_query = await self.query_processor.process_query(
                input.query, input.user_id
            )
            
            # Perform search
            if input.search_type == "semantic":
                results = await self.chroma_client.semantic_search(
                    processed_query["optimized_query"],
                    input.top_k,
                    input.similarity_threshold,
                    input.filters
                )
            elif input.search_type == "keyword":
                results = await self.chroma_client.keyword_search(
                    processed_query["optimized_query"],
                    input.top_k,
                    input.filters
                )
            else:  # hybrid
                results = await self.chroma_client.search(
                    processed_query["optimized_query"],
                    input.top_k,
                    input.similarity_threshold,
                    input.filters
                )
            
            # Rerank if enabled
            if input.use_reranking:
                results = await self.reranker.rerank(
                    input.query, results, "cross_encoder", input.top_k
                )
            
            # Convert to GraphQL types
            document_results = []
            for result in results:
                doc_result = DocumentResult(
                    id=result.get("id", ""),
                    text=result.get("text", ""),
                    score=result.get("score", 0.0),
                    metadata=result.get("metadata", {}),
                    highlights=result.get("highlights", [])
                )
                document_results.append(doc_result)
            
            return SearchResult(
                query=input.query,
                search_type=input.search_type,
                results=document_results,
                total_results=len(document_results),
                processing_time=0.0,  # Would calculate actual time
                suggestions=[]
            )
            
        except Exception as e:
            self.logger.error(f"GraphQL search error: {e}")
            raise
    
    @strawberry.field
    async def chat_query(self, input: ChatInput) -> ChatResponse:
        """Process chat query with GraphQL"""
        await self.initialize()
        
        try:
            # Generate session ID if not provided
            session_id = input.session_id or f"session_{asyncio.get_event_loop().time()}"
            
            # Get context
            from advanced.context_manager import ContextManager
            context_manager = ContextManager()
            context_messages = await context_manager.get_context(
                session_id, input.user_id, input.context_length
            )
            
            # Generate response
            full_context = context_messages + [{"role": "user", "content": input.message}]
            response_text = await self.llm_client.generate_chat_response(
                full_context, input.temperature, input.max_tokens
            )
            
            # Get sources if needed
            sources = []
            if input.include_sources:
                # Simple search for sources
                search_results = await self.chroma_client.search(
                    input.message, 3, 0.5
                )
                for result in search_results:
                    doc_result = DocumentResult(
                        id=result.get("id", ""),
                        text=result.get("text", ""),
                        score=result.get("score", 0.0),
                        metadata=result.get("metadata", {}),
                        highlights=[]
                    )
                    sources.append(doc_result)
            
            # Generate message ID
            message_id = f"msg_{asyncio.get_event_loop().time()}_{hash(input.message)}"
            
            return ChatResponse(
                message=response_text,
                session_id=session_id,
                user_id=input.user_id,
                sources=sources,
                context_used=len(context_messages) > 0,
                processing_time=0.0,
                message_id=message_id
            )
            
        except Exception as e:
            self.logger.error(f"GraphQL chat error: {e}")
            raise
    
    @strawberry.field
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile by ID"""
        await self.initialize()
        
        try:
            profile = await self.user_profiler.get_profile(user_id)
            if not profile:
                return None
            
            return UserProfile(
                user_id=profile.user_id,
                risk_tolerance=profile.risk_tolerance,
                investment_horizon=profile.investment_horizon,
                preferred_fund_types=profile.preferred_fund_types,
                expertise_level=profile.expertise_level,
                total_queries=profile.total_queries,
                last_activity=profile.last_activity
            )
            
        except Exception as e:
            self.logger.error(f"GraphQL profile error: {e}")
            raise
    
    @strawberry.field
    async def get_chat_history(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        limit: int = 10
    ) -> List[ChatMessage]:
        """Get chat history for session"""
        await self.initialize()
        
        try:
            from advanced.context_manager import ContextManager
            context_manager = ContextManager()
            
            history = await context_manager.get_session_history(
                session_id, user_id, limit, 0
            )
            
            chat_messages = []
            for msg in history:
                chat_msg = ChatMessage(
                    message_id=msg.message_id,
                    user_id=msg.user_id,
                    session_id=msg.session_id,
                    message_type=msg.message_type,
                    content=msg.content,
                    timestamp=msg.timestamp,
                    metadata=msg.metadata
                )
                chat_messages.append(chat_msg)
            
            return chat_messages
            
        except Exception as e:
            self.logger.error(f"GraphQL history error: {e}")
            raise
    
    @strawberry.field
    async def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        await self.initialize()
        
        try:
            from monitoring.metrics import MetricsCollector
            metrics_collector = MetricsCollector()
            
            system_metrics = await metrics_collector.get_system_metrics()
            
            return SystemMetrics(
                timestamp=system_metrics.get("timestamp", 0),
                cpu_usage=system_metrics.get("cpu_usage", 0),
                memory_usage=system_metrics.get("memory_usage", 0),
                active_connections=system_metrics.get("active_connections", 0),
                request_count=system_metrics.get("request_count", 0),
                error_count=system_metrics.get("error_count", 0)
            )
            
        except Exception as e:
            self.logger.error(f"GraphQL metrics error: {e}")
            raise
    
    @strawberry.field
    async def submit_feedback(self, input: FeedbackInput) -> bool:
        """Submit feedback for message"""
        await self.initialize()
        
        try:
            success = await self.user_profiler.add_feedback(
                input.user_id or "anonymous",
                {
                    "message_id": input.message_id,
                    "rating": input.rating,
                    "feedback_type": input.feedback_type,
                    "feedback_text": input.feedback_text
                }
            )
            
            return success
            
        except Exception as e:
            self.logger.error(f"GraphQL feedback error: {e}")
            raise
    
    @strawberry.field
    async def get_similar_documents(
        self,
        document_id: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[DocumentResult]:
        """Get similar documents"""
        await self.initialize()
        
        try:
            similar_docs = await self.chroma_client.find_similar_documents(
                document_id, top_k, similarity_threshold
            )
            
            document_results = []
            for doc in similar_docs:
                doc_result = DocumentResult(
                    id=doc.get("id", ""),
                    text=doc.get("text", ""),
                    score=doc.get("score", 0.0),
                    metadata=doc.get("metadata", {}),
                    highlights=[]
                )
                document_results.append(doc_result)
            
            return document_results
            
        except Exception as e:
            self.logger.error(f"GraphQL similar docs error: {e}")
            raise


# Create GraphQL schema
resolver = GraphQLResolver()

@strawberry.type
class Query:
    """GraphQL Query root"""
    
    @strawberry.field
    async def search_documents(self, input: SearchInput) -> SearchResult:
        """Search documents"""
        return await resolver.search_documents(input)
    
    @strawberry.field
    async def chat_query(self, input: ChatInput) -> ChatResponse:
        """Process chat query"""
        return await resolver.chat_query(input)
    
    @strawberry.field
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile"""
        return await resolver.get_user_profile(user_id)
    
    @strawberry.field
    async def get_chat_history(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        limit: int = 10
    ) -> List[ChatMessage]:
        """Get chat history"""
        return await resolver.get_chat_history(session_id, user_id, limit)
    
    @strawberry.field
    async def get_system_metrics(self) -> SystemMetrics:
        """Get system metrics"""
        return await resolver.get_system_metrics()
    
    @strawberry.field
    async def get_similar_documents(
        self,
        document_id: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[DocumentResult]:
        """Get similar documents"""
        return await resolver.get_similar_documents(document_id, top_k, similarity_threshold)


@strawberry.type
class Mutation:
    """GraphQL Mutation root"""
    
    @strawberry.field
    async def submit_feedback(self, input: FeedbackInput) -> bool:
        """Submit feedback"""
        return await resolver.submit_feedback(input)
    
    @strawberry.field
    async def clear_chat_history(
        self,
        session_id: str,
        user_id: Optional[str] = None
    ) -> bool:
        """Clear chat history"""
        try:
            from advanced.context_manager import ContextManager
            context_manager = ContextManager()
            return await context_manager.clear_session_history(session_id, user_id)
        except Exception as e:
            resolver.logger.error(f"GraphQL clear history error: {e}")
            return False
    
    @strawberry.field
    async def update_user_preferences(
        self,
        user_id: str,
        preferences: Dict[str, Any]
    ) -> bool:
        """Update user preferences"""
        try:
            profile = await resolver.user_profiler.get_profile(user_id)
            if profile:
                profile.preferences.update(preferences)
                return True
            return False
        except Exception as e:
            resolver.logger.error(f"GraphQL update preferences error: {e}")
            return False


# Create schema
schema = Schema(query=Query, mutation=Mutation)


# GraphQL App for FastAPI
class GraphQLApp:
    """GraphQL application for FastAPI integration"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.schema = schema
    
    async def __call__(self, request: Request) -> Response:
        """Handle GraphQL requests"""
        try:
            # Parse request
            if request.method == "POST":
                data = await request.json()
            else:
                # Handle GraphQL Playground introspection
                data = {"query": request.query_params.get("query", "")}
            
            # Execute query
            result = await self.schema.execute(
                data.get("query", ""),
                variable_values=data.get("variables"),
                operation_name=data.get("operationName")
            )
            
            # Return response
            return Response(
                content=json.dumps({
                    "data": result.data,
                    "errors": [str(error) for error in result.errors] if result.errors else None
                }),
                media_type="application/json",
                status_code=200
            )
            
        except Exception as e:
            self.logger.error(f"GraphQL execution error: {e}")
            return Response(
                content=json.dumps({"errors": [str(e)]}),
                media_type="application/json",
                status_code=500
            )


# Create GraphQL app instance
graphql_app = GraphQLApp()


# GraphQL Playground HTML
PLAYGROUND_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Phase 5-6 GraphQL Playground</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/graphql-playground-react/build/static/css/index.css" />
</head>
<body>
    <div id="root">
        <style>
            body {
                margin: 0;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
                    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
                    sans-serif;
            }
            .playground {
                height: 100vh;
                width: 100vw;
            }
        </style>
        <div class="playground">
            <div id="playground-root"></div>
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/graphql-playground-react/build/static/js/index.js"></script>
    <script>
        window.addEventListener('load', function(event) {
            GraphQLPlayground.init(document.getElementById('playground-root'), {
                endpoint: '/graphql',
                subscriptionEndpoint: '/graphql'
            });
        });
    </script>
</body>
</html>
"""


async def create_graphql_playground_response() -> Response:
    """Create GraphQL Playground response"""
    return Response(
        content=PLAYGROUND_HTML,
        media_type="text/html"
    )
