"""
Context Manager for Phase 5-6 Application
Manages conversation context and session state
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, asdict


@dataclass
class ChatMessage:
    """Chat message data structure"""
    message_id: str
    user_id: Optional[str]
    session_id: str
    message_type: str  # user, assistant, system
    content: str
    timestamp: float
    metadata: Dict[str, Any]
    context_relevance: float = 0.0


@dataclass
class ConversationContext:
    """Conversation context data structure"""
    session_id: str
    user_id: Optional[str]
    messages: List[ChatMessage]
    current_topic: Optional[str]
    entities: List[Dict[str, Any]]
    user_preferences: Dict[str, Any]
    created_at: float
    last_activity: float
    context_window: int = 10


class ContextManager:
    """Advanced context manager for conversations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.sessions: Dict[str, ConversationContext] = {}
        self.context_window = 10  # Number of messages to keep in context
        self.session_timeout = 3600  # 1 hour in seconds
        
    async def get_context(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get conversation context for a session
        """
        try:
            # Get or create session
            session = await self._get_or_create_session(session_id, user_id)
            
            # Update last activity
            session.last_activity = asyncio.get_event_loop().time()
            
            # Get recent messages
            recent_messages = session.messages[-limit:] if len(session.messages) > limit else session.messages
            
            # Convert to context format
            context_messages = []
            for msg in recent_messages:
                context_messages.append({
                    "role": msg.message_type,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                    "metadata": msg.metadata
                })
            
            self.logger.info(f"Retrieved context for session {session_id}: {len(context_messages)} messages")
            return context_messages
            
        except Exception as e:
            self.logger.error(f"Error getting context: {e}")
            return []
    
    async def add_message(
        self,
        message: ChatMessage
    ) -> bool:
        """
        Add message to conversation context
        """
        try:
            session = await self._get_or_create_session(message.session_id, message.user_id)
            
            # Add message to session
            session.messages.append(message)
            
            # Update context
            await self._update_context(session, message)
            
            # Limit context window
            if len(session.messages) > self.context_window:
                session.messages = session.messages[-self.context_window:]
            
            self.logger.info(f"Added message to session {message.session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding message: {e}")
            return False
    
    async def clear_session_history(
        self,
        session_id: str,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Clear session history
        """
        try:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                session.messages.clear()
                session.current_topic = None
                session.entities.clear()
                session.last_activity = asyncio.get_event_loop().time()
                
                self.logger.info(f"Cleared history for session {session_id}")
                return True
            else:
                self.logger.warning(f"Session {session_id} not found")
                return False
                
        except Exception as e:
            self.logger.error(f"Error clearing session history: {e}")
            return False
    
    async def get_session_history(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[ChatMessage]:
        """
        Get session message history
        """
        try:
            session = await self._get_or_create_session(session_id, user_id)
            
            # Get messages with pagination
            messages = session.messages[offset:offset + limit]
            
            return messages
            
        except Exception as e:
            self.logger.error(f"Error getting session history: {e}")
            return []
    
    async def get_user_sessions(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get all sessions for a user
        """
        try:
            user_sessions = []
            
            for session_id, session in self.sessions.items():
                if session.user_id == user_id:
                    user_sessions.append({
                        "session_id": session_id,
                        "created_at": session.created_at,
                        "last_activity": session.last_activity,
                        "message_count": len(session.messages),
                        "current_topic": session.current_topic
                    })
            
            # Sort by last activity and limit
            user_sessions.sort(key=lambda x: x["last_activity"], reverse=True)
            
            return user_sessions[:limit]
            
        except Exception as e:
            self.logger.error(f"Error getting user sessions: {e}")
            return []
    
    async def requires_rag_retrieval(
        self,
        message: str,
        context_messages: List[Dict[str, Any]]
    ) -> bool:
        """
        Determine if RAG retrieval is needed for the message
        """
        try:
            # Check if message contains information-seeking keywords
            info_keywords = [
                "what", "how", "why", "when", "where", "tell me", "explain",
                "describe", "show me", "find", "search", "look for", "information"
            ]
            
            message_lower = message.lower()
            has_info_keyword = any(keyword in message_lower for keyword in info_keywords)
            
            # Check if context already contains relevant information
            context_relevant = False
            for ctx_msg in context_messages:
                if ctx_msg.get("role") == "assistant":
                    ctx_content = ctx_msg.get("content", "").lower()
                    # Simple relevance check
                    if any(word in ctx_content for word in message_lower.split()[:3]):
                        context_relevant = True
                        break
            
            # RAG needed if seeking information and context doesn't have it
            return has_info_keyword and not context_relevant
            
        except Exception as e:
            self.logger.error(f"Error determining RAG need: {e}")
            return True  # Default to True
    
    async def generate_suggestions(
        self,
        message: str,
        user_id: Optional[str] = None
    ) -> List[str]:
        """
        Generate conversation suggestions based on context
        """
        try:
            suggestions = []
            
            # Generate suggestions based on message content
            message_lower = message.lower()
            
            if "fund" in message_lower:
                suggestions.extend([
                    "What are the top performing mutual funds?",
                    "How do I choose the right mutual fund?",
                    "What are the risks of mutual fund investments?"
                ])
            
            if "returns" in message_lower:
                suggestions.extend([
                    "What are the historical returns of mutual funds?",
                    "How are mutual fund returns calculated?",
                    "What factors affect mutual fund returns?"
                ])
            
            if "invest" in message_lower:
                suggestions.extend([
                    "How much should I invest in mutual funds?",
                    "What is the best way to start investing?",
                    "What are the different types of investments?"
                ])
            
            # Add personalized suggestions if user_id provided
            if user_id:
                user_suggestions = await self._generate_personalized_suggestions(user_id, message)
                suggestions.extend(user_suggestions)
            
            # Remove duplicates and limit
            unique_suggestions = list(set(suggestions))
            return unique_suggestions[:5]
            
        except Exception as e:
            self.logger.error(f"Error generating suggestions: {e}")
            return []
    
    async def _get_or_create_session(
        self,
        session_id: str,
        user_id: Optional[str] = None
    ) -> ConversationContext:
        """Get existing session or create new one"""
        if session_id not in self.sessions:
            # Create new session
            session = ConversationContext(
                session_id=session_id,
                user_id=user_id,
                messages=[],
                current_topic=None,
                entities=[],
                user_preferences={},
                created_at=asyncio.get_event_loop().time(),
                last_activity=asyncio.get_event_loop().time()
            )
            self.sessions[session_id] = session
            
            self.logger.info(f"Created new session: {session_id}")
        
        return self.sessions[session_id]
    
    async def _update_context(
        self,
        session: ConversationContext,
        message: ChatMessage
    ) -> None:
        """Update conversation context based on new message"""
        try:
            # Extract entities from message
            if message.message_type == "user":
                entities = await self._extract_entities(message.content)
                session.entities.extend(entities)
                
                # Update current topic based on message
                topic = await self._extract_topic(message.content)
                if topic:
                    session.current_topic = topic
            
            # Update user preferences
            if message.user_id:
                await self._update_user_preferences(session, message)
                
        except Exception as e:
            self.logger.error(f"Error updating context: {e}")
    
    async def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text"""
        entities = []
        
        # Simple entity extraction (mock implementation)
        financial_terms = ["nav", "returns", "fund", "investment", "sip", "aum"]
        
        for term in financial_terms:
            if term in text.lower():
                entities.append({
                    "type": "financial_term",
                    "value": term,
                    "confidence": 0.8
                })
        
        return entities
    
    async def _extract_topic(self, text: str) -> Optional[str]:
        """Extract topic from text"""
        text_lower = text.lower()
        
        # Simple topic extraction
        if "fund" in text_lower and "performance" in text_lower:
            return "fund_performance"
        elif "investment" in text_lower and "advice" in text_lower:
            return "investment_advice"
        elif "nav" in text_lower:
            return "nav_information"
        elif "returns" in text_lower:
            return "returns_analysis"
        
        return None
    
    async def _update_user_preferences(
        self,
        session: ConversationContext,
        message: ChatMessage
    ) -> None:
        """Update user preferences based on message"""
        try:
            # Simple preference learning
            if "risk" in message.content.lower():
                if "low" in message.content.lower():
                    session.user_preferences["risk_tolerance"] = "low"
                elif "high" in message.content.lower():
                    session.user_preferences["risk_tolerance"] = "high"
            
            if "fund type" in message.content.lower():
                if "equity" in message.content.lower():
                    session.user_preferences["preferred_fund_type"] = "equity"
                elif "debt" in message.content.lower():
                    session.user_preferences["preferred_fund_type"] = "debt"
                    
        except Exception as e:
            self.logger.error(f"Error updating user preferences: {e}")
    
    async def _generate_personalized_suggestions(
        self,
        user_id: str,
        message: str
    ) -> List[str]:
        """Generate personalized suggestions for user"""
        # Mock personalized suggestions
        return [
            "Based on your previous queries, you might be interested in...",
            "Would you like to know more about investment strategies?",
            "How about exploring different fund categories?"
        ]
    
    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions"""
        try:
            current_time = asyncio.get_event_loop().time()
            expired_sessions = []
            
            for session_id, session in self.sessions.items():
                if current_time - session.last_activity > self.session_timeout:
                    expired_sessions.append(session_id)
            
            # Remove expired sessions
            for session_id in expired_sessions:
                del self.sessions[session_id]
            
            self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
            return len(expired_sessions)
            
        except Exception as e:
            self.logger.error(f"Error cleaning up sessions: {e}")
            return 0
    
    async def get_context_statistics(self) -> Dict[str, Any]:
        """Get context manager statistics"""
        try:
            total_sessions = len(self.sessions)
            total_messages = sum(len(session.messages) for session in self.sessions.values())
            active_sessions = len([
                s for s in self.sessions.values()
                if asyncio.get_event_loop().time() - s.last_activity < 300  # Active in last 5 minutes
            ])
            
            return {
                "total_sessions": total_sessions,
                "total_messages": total_messages,
                "active_sessions": active_sessions,
                "context_window": self.context_window,
                "session_timeout": self.session_timeout
            }
            
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {}


class ContextCache:
    """Cache for frequently used context"""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached context"""
        return self.cache.get(key)
    
    def set(self, key: str, value: Any) -> None:
        """Set cached context"""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = value
    
    def clear(self) -> None:
        """Clear cache"""
        self.cache.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "utilization": len(self.cache) / self.max_size * 100
        }


class ContextAnalyzer:
    """Analyzes conversation context for insights"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def analyze_conversation_flow(
        self,
        session_id: str,
        context_manager: ContextManager
    ) -> Dict[str, Any]:
        """Analyze conversation flow patterns"""
        try:
            session = await context_manager._get_or_create_session(session_id)
            messages = session.messages
            
            if len(messages) < 2:
                return {"status": "insufficient_data"}
            
            # Analyze message patterns
            user_messages = [msg for msg in messages if msg.message_type == "user"]
            assistant_messages = [msg for msg in messages if msg.message_type == "assistant"]
            
            # Calculate metrics
            avg_user_message_length = sum(len(msg.content) for msg in user_messages) / len(user_messages)
            avg_response_time = self._calculate_avg_response_time(messages)
            topic_transitions = self._count_topic_transitions(messages)
            
            return {
                "session_id": session_id,
                "total_messages": len(messages),
                "user_messages": len(user_messages),
                "assistant_messages": len(assistant_messages),
                "avg_user_message_length": avg_user_message_length,
                "avg_response_time": avg_response_time,
                "topic_transitions": topic_transitions,
                "current_topic": session.current_topic,
                "entities_identified": len(session.entities)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing conversation flow: {e}")
            return {"status": "error", "error": str(e)}
    
    def _calculate_avg_response_time(self, messages: List[ChatMessage]) -> float:
        """Calculate average response time between messages"""
        if len(messages) < 2:
            return 0.0
        
        response_times = []
        for i in range(1, len(messages)):
            if messages[i].message_type == "assistant" and messages[i-1].message_type == "user":
                response_time = messages[i].timestamp - messages[i-1].timestamp
                response_times.append(response_time)
        
        return sum(response_times) / len(response_times) if response_times else 0.0
    
    def _count_topic_transitions(self, messages: List[ChatMessage]) -> int:
        """Count topic transitions in conversation"""
        transitions = 0
        current_topic = None
        
        for message in messages:
            if message.metadata.get("topic") and message.metadata["topic"] != current_topic:
                current_topic = message.metadata["topic"]
                transitions += 1
        
        return transitions
