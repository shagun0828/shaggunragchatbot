"""
LLM Client Integration for Phase 5-6 Application
Handles language model operations for RAG system
"""

import asyncio
import logging
import json
from typing import List, Dict, Any, Optional, AsyncGenerator
import aiohttp
from datetime import datetime
import numpy as np


class LLMClient:
    """LLM client for language model operations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.session = None
        self.initialized = False
        self.model_configs = {
            "gpt-4": {
                "endpoint": "https://api.openai.com/v1/chat/completions",
                "max_tokens": 4096,
                "temperature": 0.7
            },
            "claude-3": {
                "endpoint": "https://api.anthropic.com/v1/messages",
                "max_tokens": 4096,
                "temperature": 0.7
            },
            "llama-3": {
                "endpoint": "https://api.llama-api.com/v1/chat/completions",
                "max_tokens": 4096,
                "temperature": 0.7
            }
        }
        self.current_model = "gpt-4"
        
    async def initialize(self):
        """Initialize LLM client"""
        try:
            self.session = aiohttp.ClientSession()
            self.initialized = True
            self.logger.info("LLM client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM client: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Check LLM health status"""
        if not self.initialized:
            return {"status": "not_initialized", "error": "Client not initialized"}
        
        try:
            # Simple health check - would make actual API call in production
            return {
                "status": "healthy",
                "model": self.current_model,
                "max_tokens": self.model_configs[self.current_model]["max_tokens"],
                "temperature": self.model_configs[self.current_model]["temperature"]
            }
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def generate_response(
        self,
        query: str,
        context: List[str],
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None
    ) -> str:
        """
        Generate response using LLM with RAG context
        """
        if not self.initialized:
            raise RuntimeError("LLM client not initialized")
        
        try:
            # Use provided model or current model
            model_name = model or self.current_model
            config = self.model_configs[model_name]
            
            # Build prompt with context
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(query, context)
            
            # Prepare messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Generate response
            if stream:
                # For streaming, would return async generator
                response = await self._generate_streaming_response(
                    messages, temperature, max_tokens, model_name
                )
                # Collect full response for non-streaming return
                full_response = ""
                async for chunk in response:
                    full_response += chunk
                return full_response
            else:
                response = await self._generate_single_response(
                    messages, temperature, max_tokens, model_name
                )
                return response
                
        except Exception as e:
            self.logger.error(f"Response generation failed: {e}")
            raise
    
    async def generate_chat_response(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None
    ) -> str:
        """
        Generate chat response from message history
        """
        if not self.initialized:
            raise RuntimeError("LLM client not initialized")
        
        try:
            model_name = model or self.current_model
            config = self.model_configs[model_name]
            
            # Add system message if not present
            if not any(msg.get("role") == "system" for msg in messages):
                system_message = {"role": "system", "content": self._build_system_prompt()}
                messages.insert(0, system_message)
            
            return await self._generate_single_response(
                messages, temperature, max_tokens, model_name
            )
            
        except Exception as e:
            self.logger.error(f"Chat response generation failed: {e}")
            raise
    
    async def generate_streaming_response(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming response
        """
        if not self.initialized:
            raise RuntimeError("LLM client not initialized")
        
        try:
            model_name = model or self.current_model
            config = self.model_configs[model_name]
            
            # Mock streaming response - in production, would use actual API
            response_text = await self._generate_single_response(
                messages, temperature, max_tokens, model_name
            )
            
            # Split response into chunks for streaming
            words = response_text.split()
            current_chunk = ""
            
            for word in words:
                current_chunk += word + " "
                if len(current_chunk) > 10:  # Send chunks of ~10 characters
                    yield current_chunk
                    current_chunk = ""
                    await asyncio.sleep(0.05)  # Simulate streaming delay
            
            if current_chunk:
                yield current_chunk
                
        except Exception as e:
            self.logger.error(f"Streaming response generation failed: {e}")
            raise
    
    async def analyze_query_intent(
        self,
        query: str
    ) -> Dict[str, Any]:
        """
        Analyze user query intent and extract entities
        """
        if not self.initialized:
            raise RuntimeError("LLM client not initialized")
        
        try:
            # Mock intent analysis - in production, would use actual LLM
            intent_analysis = {
                "intent": "information_seeking",
                "entities": [],
                "confidence": 0.85,
                "query_type": "question",
                "keywords": query.lower().split()
            }
            
            # Extract financial entities
            financial_terms = ["nav", "returns", "aum", "expense ratio", "fund", "investment"]
            entities = []
            for term in financial_terms:
                if term in query.lower():
                    entities.append({"type": "financial_term", "value": term})
            
            intent_analysis["entities"] = entities
            
            return intent_analysis
            
        except Exception as e:
            self.logger.error(f"Query intent analysis failed: {e}")
            raise
    
    async def extract_key_information(
        self,
        text: str,
        information_types: List[str] = None
    ) -> Dict[str, Any]:
        """
        Extract key information from text
        """
        if not self.initialized:
            raise RuntimeError("LLM client not initialized")
        
        try:
            # Mock information extraction
            extracted_info = {
                "fund_name": None,
                "nav": None,
                "returns": None,
                "expense_ratio": None,
                "risk_level": None,
                "investment_objective": None
            }
            
            # Simple pattern matching for demonstration
            text_lower = text.lower()
            
            # Extract NAV
            if "nav" in text_lower:
                import re
                nav_match = re.search(r'nav[:\s]*\s*([0-9,]+\.?\d*)', text_lower)
                if nav_match:
                    extracted_info["nav"] = nav_match.group(1)
            
            # Extract returns
            if "returns" in text_lower:
                returns_match = re.search(r'returns[:\s]*\s*([0-9.]+)%', text_lower)
                if returns_match:
                    extracted_info["returns"] = returns_match.group(1) + "%"
            
            return extracted_info
            
        except Exception as e:
            self.logger.error(f"Information extraction failed: {e}")
            raise
    
    async def summarize_documents(
        self,
        documents: List[str],
        max_length: int = 200
    ) -> str:
        """
        Summarize multiple documents
        """
        if not self.initialized:
            raise RuntimeError("LLM client not initialized")
        
        try:
            # Mock summarization - in production, would use actual LLM
            combined_text = " ".join(documents[:3])  # Limit to first 3 docs
            
            # Simple extractive summarization
            sentences = combined_text.split(". ")
            summary_sentences = sentences[:3]  # Take first 3 sentences
            
            summary = ". ".join(summary_sentences)
            if len(summary) > max_length:
                summary = summary[:max_length] + "..."
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Document summarization failed: {e}")
            raise
    
    async def generate_suggestions(
        self,
        query: str,
        context: List[str] = None
    ) -> List[str]:
        """
        Generate follow-up suggestions based on query and context
        """
        if not self.initialized:
            raise RuntimeError("LLM client not initialized")
        
        try:
            # Mock suggestion generation
            suggestions = []
            
            # Generate suggestions based on query content
            query_lower = query.lower()
            
            if "fund" in query_lower:
                suggestions.extend([
                    "What are the top performing mutual funds?",
                    "How to compare different mutual funds?",
                    "What are the risks associated with mutual funds?"
                ])
            
            if "returns" in query_lower:
                suggestions.extend([
                    "What are the historical returns of this fund?",
                    "How do returns compare to benchmark?",
                    "What factors affect fund returns?"
                ])
            
            if "nav" in query_lower:
                suggestions.extend([
                    "What is NAV and how is it calculated?",
                    "How often is NAV updated?",
                    "What is the significance of NAV changes?"
                ])
            
            # Add generic suggestions if no specific ones
            if not suggestions:
                suggestions = [
                    "Tell me more about mutual fund investments",
                    "What are the different types of mutual funds?",
                    "How to start investing in mutual funds?"
                ]
            
            return suggestions[:5]  # Return top 5 suggestions
            
        except Exception as e:
            self.logger.error(f"Suggestion generation failed: {e}")
            raise
    
    async def evaluate_response_quality(
        self,
        query: str,
        response: str,
        context: List[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the quality of a response
        """
        if not self.initialized:
            raise RuntimeError("LLM client not initialized")
        
        try:
            # Mock quality evaluation
            quality_metrics = {
                "relevance": 0.85,
                "accuracy": 0.90,
                "completeness": 0.80,
                "clarity": 0.88,
                "overall_score": 0.86,
                "feedback": "Good response with relevant information"
            }
            
            # Simple heuristics for quality assessment
            if len(response) < 50:
                quality_metrics["completeness"] = 0.5
                quality_metrics["feedback"] = "Response is too short"
            
            if not any(word in response.lower() for word in query.lower().split()[:3]):
                quality_metrics["relevance"] = 0.6
                quality_metrics["feedback"] = "Response may not be relevant to query"
            
            # Calculate overall score
            quality_metrics["overall_score"] = (
                quality_metrics["relevance"] * 0.3 +
                quality_metrics["accuracy"] * 0.3 +
                quality_metrics["completeness"] * 0.2 +
                quality_metrics["clarity"] * 0.2
            )
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"Response quality evaluation failed: {e}")
            raise
    
    async def close(self):
        """Close LLM client"""
        if self.session:
            await self.session.close()
        
        self.initialized = False
        self.logger.info("LLM client closed")
    
    # Helper methods
    def _build_system_prompt(self) -> str:
        """Build system prompt for LLM"""
        return """You are a helpful financial assistant specializing in mutual funds and investment advice. 
        You provide accurate, helpful information based on the context provided.
        Always be clear, concise, and professional in your responses.
        If you're not sure about something, acknowledge it and suggest seeking professional advice."""
    
    def _build_user_prompt(self, query: str, context: List[str]) -> str:
        """Build user prompt with context"""
        context_text = "\n\n".join(context) if context else ""
        
        prompt = f"""Based on the following context, please answer the user's question:

Context:
{context_text}

Question: {query}

Please provide a helpful and accurate response based on the context provided. 
If the context doesn't contain relevant information, please say so and provide general guidance."""
        
        return prompt
    
    async def _generate_single_response(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        model_name: str
    ) -> str:
        """Generate single response from LLM"""
        # Mock implementation - in production, would make actual API call
        config = self.model_configs[model_name]
        
        # Simulate API call delay
        await asyncio.sleep(0.5)
        
        # Generate mock response based on last user message
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break
        
        # Simple response generation based on query
        if "fund" in user_message.lower():
            response = "Based on the context provided, this mutual fund appears to be a solid investment option. However, please consider your risk tolerance and investment goals before making any decisions."
        elif "returns" in user_message.lower():
            response = "The fund has shown consistent returns over the past few years, but past performance doesn't guarantee future results. Please review the fund's prospectus for detailed information."
        elif "nav" in user_message.lower():
            response = "The NAV (Net Asset Value) represents the per-unit value of the fund's assets. It's calculated daily and reflects the fund's performance."
        else:
            response = "I understand your question. Based on the information available, I recommend consulting with a financial advisor for personalized investment guidance."
        
        return response
    
    async def _generate_streaming_response(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        model_name: str
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response from LLM"""
        # Get full response first
        full_response = await self._generate_single_response(
            messages, temperature, max_tokens, model_name
        )
        
        # Stream it word by word
        words = full_response.split()
        for word in words:
            yield word + " "
            await asyncio.sleep(0.05)


class ConversationManager:
    """Manages conversation state and context"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__)
        self.conversations = {}  # session_id -> conversation_data
    
    async def start_conversation(
        self,
        session_id: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Start a new conversation"""
        conversation_data = {
            "session_id": session_id,
            "user_id": user_id,
            "messages": [],
            "started_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat()
        }
        
        self.conversations[session_id] = conversation_data
        
        return {
            "session_id": session_id,
            "status": "started",
            "message": "Conversation started successfully"
        }
    
    async def add_message(
        self,
        session_id: str,
        message: str,
        role: str = "user",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Add message to conversation"""
        if session_id not in self.conversations:
            raise ValueError(f"Conversation {session_id} not found")
        
        conversation = self.conversations[session_id]
        
        message_data = {
            "role": role,
            "content": message,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        conversation["messages"].append(message_data)
        conversation["last_activity"] = datetime.now().isoformat()
        
        return message_data
    
    async def get_conversation(
        self,
        session_id: str
    ) -> Dict[str, Any]:
        """Get conversation data"""
        if session_id not in self.conversations:
            raise ValueError(f"Conversation {session_id} not found")
        
        return self.conversations[session_id]
    
    async def end_conversation(
        self,
        session_id: str
    ) -> Dict[str, Any]:
        """End conversation"""
        if session_id not in self.conversations:
            raise ValueError(f"Conversation {session_id} not found")
        
        conversation = self.conversations[session_id]
        conversation["ended_at"] = datetime.now().isoformat()
        conversation["status"] = "ended"
        
        # Remove from active conversations
        del self.conversations[session_id]
        
        return {
            "session_id": session_id,
            "status": "ended",
            "message_count": len(conversation["messages"])
        }
    
    async def get_conversation_summary(
        self,
        session_id: str
    ) -> str:
        """Get conversation summary"""
        if session_id not in self.conversations:
            raise ValueError(f"Conversation {session_id} not found")
        
        conversation = self.conversations[session_id]
        messages = conversation["messages"]
        
        # Extract user messages
        user_messages = [
            msg["content"] for msg in messages 
            if msg["role"] == "user"
        ]
        
        if not user_messages:
            return "No user messages found in conversation."
        
        # Generate summary
        summary_text = " ".join(user_messages)
        summary = await self.llm_client.summarize_documents([summary_text], 100)
        
        return summary
