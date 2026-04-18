"""
WebSocket Manager for Phase 5-6 Application
Real-time communication and streaming responses
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from dataclasses import dataclass, asdict
from fastapi import WebSocket, WebSocketDisconnect
import uuid


@dataclass
class WebSocketConnection:
    """WebSocket connection data"""
    connection_id: str
    websocket: WebSocket
    user_id: Optional[str]
    session_id: Optional[str]
    connected_at: float
    last_activity: float
    metadata: Dict[str, Any]


@dataclass
class WebSocketMessage:
    """WebSocket message data"""
    message_id: str
    connection_id: str
    message_type: str
    content: Dict[str, Any]
    timestamp: float
    metadata: Dict[str, Any]


class WebSocketManager:
    """Manages WebSocket connections and real-time communication"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.connections: Dict[str, WebSocketConnection] = {}
        self.user_connections: Dict[str, Set[str]] = defaultdict(set)  # user_id -> connection_ids
        self.session_connections: Dict[str, Set[str]] = defaultdict(set)  # session_id -> connection_ids
        self.message_queue = asyncio.Queue()
        self.broadcast_queue = asyncio.Queue()
        self.connection_timeout = 3600  # 1 hour
        self.heartbeat_interval = 30  # 30 seconds
        
        # Start background tasks
        self.background_tasks = []
        self._start_background_tasks()
    
    async def connect(self, websocket: WebSocket, user_id: Optional[str] = None, session_id: Optional[str] = None) -> str:
        """
        Accept and register new WebSocket connection
        """
        try:
            await websocket.accept()
            
            # Generate connection ID
            connection_id = str(uuid.uuid4())
            
            # Create connection object
            connection = WebSocketConnection(
                connection_id=connection_id,
                websocket=websocket,
                user_id=user_id,
                session_id=session_id,
                connected_at=asyncio.get_event_loop().time(),
                last_activity=asyncio.get_event_loop().time(),
                metadata={}
            )
            
            # Register connection
            self.connections[connection_id] = connection
            
            if user_id:
                self.user_connections[user_id].add(connection_id)
            
            if session_id:
                self.session_connections[session_id].add(connection_id)
            
            # Send welcome message
            await self.send_message(connection_id, {
                "type": "connection_established",
                "connection_id": connection_id,
                "user_id": user_id,
                "session_id": session_id,
                "timestamp": connection.connected_at
            })
            
            self.logger.info(f"WebSocket connection established: {connection_id}")
            return connection_id
            
        except Exception as e:
            self.logger.error(f"Error establishing WebSocket connection: {e}")
            raise
    
    async def disconnect(self, connection_id: str) -> None:
        """
        Disconnect and cleanup WebSocket connection
        """
        try:
            if connection_id in self.connections:
                connection = self.connections[connection_id]
                
                # Remove from user connections
                if connection.user_id:
                    self.user_connections[connection.user_id].discard(connection_id)
                    if not self.user_connections[connection.user_id]:
                        del self.user_connections[connection.user_id]
                
                # Remove from session connections
                if connection.session_id:
                    self.session_connections[connection.session_id].discard(connection_id)
                    if not self.session_connections[connection.session_id]:
                        del self.session_connections[connection.session_id]
                
                # Remove from connections
                del self.connections[connection_id]
                
                self.logger.info(f"WebSocket connection disconnected: {connection_id}")
            
        except Exception as e:
            self.logger.error(f"Error disconnecting WebSocket: {e}")
    
    async def send_message(self, connection_id: str, message: Dict[str, Any]) -> bool:
        """
        Send message to specific connection
        """
        try:
            if connection_id not in self.connections:
                return False
            
            connection = self.connections[connection_id]
            
            # Add metadata
            message["timestamp"] = asyncio.get_event_loop().time()
            message["connection_id"] = connection_id
            
            # Send message
            await connection.websocket.send_text(json.dumps(message))
            
            # Update last activity
            connection.last_activity = asyncio.get_event_loop().time()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending message to {connection_id}: {e}")
            # Connection might be broken, remove it
            await self.disconnect(connection_id)
            return False
    
    async def send_to_user(self, user_id: str, message: Dict[str, Any]) -> int:
        """
        Send message to all connections for a user
        """
        if user_id not in self.user_connections:
            return 0
        
        connection_ids = self.user_connections[user_id].copy()
        sent_count = 0
        
        for connection_id in connection_ids:
            if await self.send_message(connection_id, message):
                sent_count += 1
        
        return sent_count
    
    async def send_to_session(self, session_id: str, message: Dict[str, Any]) -> int:
        """
        Send message to all connections in a session
        """
        if session_id not in self.session_connections:
            return 0
        
        connection_ids = self.session_connections[session_id].copy()
        sent_count = 0
        
        for connection_id in connection_ids:
            if await self.send_message(connection_id, message):
                sent_count += 1
        
        return sent_count
    
    async def broadcast(self, message: Dict[str, Any], exclude_connection_id: Optional[str] = None) -> int:
        """
        Broadcast message to all connected clients
        """
        connection_ids = list(self.connections.keys())
        sent_count = 0
        
        for connection_id in connection_ids:
            if connection_id != exclude_connection_id:
                if await self.send_message(connection_id, message):
                    sent_count += 1
        
        return sent_count
    
    async def handle_message(self, connection_id: str, message_data: Dict[str, Any]) -> None:
        """
        Handle incoming WebSocket message
        """
        try:
            connection = self.connections.get(connection_id)
            if not connection:
                return
            
            # Update last activity
            connection.last_activity = asyncio.get_event_loop().time()
            
            message_type = message_data.get("type", "unknown")
            
            # Handle different message types
            if message_type == "chat":
                await self._handle_chat_message(connection_id, message_data)
            elif message_type == "search":
                await self._handle_search_message(connection_id, message_data)
            elif message_type == "heartbeat":
                await self._handle_heartbeat(connection_id, message_data)
            elif message_type == "subscribe":
                await self._handle_subscription(connection_id, message_data)
            elif message_type == "unsubscribe":
                await self._handle_unsubscription(connection_id, message_data)
            else:
                await self.send_message(connection_id, {
                    "type": "error",
                    "message": f"Unknown message type: {message_type}"
                })
            
        except Exception as e:
            self.logger.error(f"Error handling message from {connection_id}: {e}")
            await self.send_message(connection_id, {
                "type": "error",
                "message": "Internal server error"
            })
    
    async def _handle_chat_message(self, connection_id: str, message_data: Dict[str, Any]) -> None:
        """Handle chat message"""
        try:
            # Extract chat data
            message = message_data.get("message", "")
            user_id = message_data.get("user_id")
            session_id = message_data.get("session_id")
            
            # Process chat message
            from api.chat_endpoints import chat_message
            from pydantic import BaseModel
            
            # Create chat request (simplified)
            chat_request = {
                "message": message,
                "user_id": user_id,
                "session_id": session_id,
                "stream": True
            }
            
            # Send typing indicator
            await self.send_message(connection_id, {
                "type": "typing",
                "status": "started"
            })
            
            # Process chat and stream response
            # In a real implementation, this would integrate with the chat endpoints
            await self._stream_chat_response(connection_id, chat_request)
            
        except Exception as e:
            self.logger.error(f"Error handling chat message: {e}")
            await self.send_message(connection_id, {
                "type": "chat_error",
                "message": str(e)
            })
    
    async def _handle_search_message(self, connection_id: str, message_data: Dict[str, Any]) -> None:
        """Handle search message"""
        try:
            # Extract search data
            query = message_data.get("query", "")
            search_type = message_data.get("search_type", "semantic")
            top_k = message_data.get("top_k", 10)
            
            # Send search status
            await self.send_message(connection_id, {
                "type": "search_status",
                "status": "searching"
            })
            
            # Perform search (mock implementation)
            await asyncio.sleep(0.5)  # Simulate search time
            
            # Send search results
            await self.send_message(connection_id, {
                "type": "search_results",
                "query": query,
                "results": [
                    {
                        "id": f"doc_{i}",
                        "text": f"Search result {i} for query: {query}",
                        "score": 0.8 - (i * 0.1),
                        "metadata": {"source": "test"}
                    }
                    for i in range(min(top_k, 5))
                ],
                "total_results": min(top_k, 5)
            })
            
        except Exception as e:
            self.logger.error(f"Error handling search message: {e}")
            await self.send_message(connection_id, {
                "type": "search_error",
                "message": str(e)
            })
    
    async def _handle_heartbeat(self, connection_id: str, message_data: Dict[str, Any]) -> None:
        """Handle heartbeat message"""
        await self.send_message(connection_id, {
            "type": "heartbeat_response",
            "timestamp": asyncio.get_event_loop().time()
        })
    
    async def _handle_subscription(self, connection_id: str, message_data: Dict[str, Any]) -> None:
        """Handle subscription message"""
        subscription_type = message_data.get("subscription_type")
        
        # Add subscription to connection metadata
        connection = self.connections.get(connection_id)
        if connection:
            if "subscriptions" not in connection.metadata:
                connection.metadata["subscriptions"] = set()
            connection.metadata["subscriptions"].add(subscription_type)
        
        await self.send_message(connection_id, {
            "type": "subscription_confirmed",
            "subscription_type": subscription_type
        })
    
    async def _handle_unsubscription(self, connection_id: str, message_data: Dict[str, Any]) -> None:
        """Handle unsubscription message"""
        subscription_type = message_data.get("subscription_type")
        
        # Remove subscription from connection metadata
        connection = self.connections.get(connection_id)
        if connection and "subscriptions" in connection.metadata:
            connection.metadata["subscriptions"].discard(subscription_type)
        
        await self.send_message(connection_id, {
            "type": "unsubscription_confirmed",
            "subscription_type": subscription_type
        })
    
    async def _stream_chat_response(self, connection_id: str, chat_request: Dict[str, Any]) -> None:
        """Stream chat response to client"""
        try:
            # Mock streaming response
            response_text = f"This is a streaming response to: {chat_request.get('message', '')}"
            words = response_text.split()
            
            await self.send_message(connection_id, {
                "type": "chat_response_start",
                "message_id": str(uuid.uuid4())
            })
            
            # Stream words
            for i, word in enumerate(words):
                await self.send_message(connection_id, {
                    "type": "chat_response_chunk",
                    "chunk": word + " ",
                    "chunk_index": i,
                    "is_complete": False
                })
                await asyncio.sleep(0.05)  # Simulate streaming delay
            
            # Send completion
            await self.send_message(connection_id, {
                "type": "chat_response_complete",
                "is_complete": True
            })
            
        except Exception as e:
            self.logger.error(f"Error streaming chat response: {e}")
            await self.send_message(connection_id, {
                "type": "chat_error",
                "message": str(e)
            })
    
    async def get_connection_status(self) -> Dict[str, Any]:
        """Get WebSocket connection status"""
        return {
            "total_connections": len(self.connections),
            "user_connections": len(self.user_connections),
            "session_connections": len(self.session_connections),
            "active_connections": len([
                conn for conn in self.connections.values()
                if asyncio.get_event_loop().time() - conn.last_activity < 300  # Active in last 5 minutes
            ])
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get WebSocket manager status"""
        return {
            "connections": len(self.connections),
            "users": len(self.user_connections),
            "sessions": len(self.session_connections),
            "message_queue_size": self.message_queue.qsize(),
            "broadcast_queue_size": self.broadcast_queue.qsize()
        }
    
    async def disconnect_all(self) -> int:
        """Disconnect all connections"""
        connection_ids = list(self.connections.keys())
        disconnected_count = 0
        
        for connection_id in connection_ids:
            try:
                await self.disconnect(connection_id)
                disconnected_count += 1
            except Exception as e:
                self.logger.error(f"Error disconnecting {connection_id}: {e}")
        
        return disconnected_count
    
    def _start_background_tasks(self):
        """Start background tasks for WebSocket management"""
        # Heartbeat task
        heartbeat_task = asyncio.create_task(self._heartbeat_task())
        self.background_tasks.append(heartbeat_task)
        
        # Cleanup task
        cleanup_task = asyncio.create_task(self._cleanup_task())
        self.background_tasks.append(cleanup_task)
        
        # Message processor task
        message_processor_task = asyncio.create_task(self._message_processor_task())
        self.background_tasks.append(message_processor_task)
    
    async def _heartbeat_task(self):
        """Send periodic heartbeats to all connections"""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                heartbeat_message = {
                    "type": "server_heartbeat",
                    "timestamp": asyncio.get_event_loop().time()
                }
                
                await self.broadcast(heartbeat_message)
                
            except Exception as e:
                self.logger.error(f"Error in heartbeat task: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _cleanup_task(self):
        """Cleanup inactive connections"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                current_time = asyncio.get_event_loop().time()
                inactive_connections = []
                
                for connection_id, connection in self.connections.items():
                    if current_time - connection.last_activity > self.connection_timeout:
                        inactive_connections.append(connection_id)
                
                for connection_id in inactive_connections:
                    await self.disconnect(connection_id)
                
                if inactive_connections:
                    self.logger.info(f"Cleaned up {len(inactive_connections)} inactive connections")
                
            except Exception as e:
                self.logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(10)  # Wait before retrying
    
    async def _message_processor_task(self):
        """Process queued messages"""
        while True:
            try:
                # Process message queue
                while not self.message_queue.empty():
                    message = await self.message_queue.get()
                    await self._process_queued_message(message)
                
                # Process broadcast queue
                while not self.broadcast_queue.empty():
                    message = await self.broadcast_queue.get()
                    await self._process_queued_broadcast(message)
                
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                self.logger.error(f"Error in message processor task: {e}")
                await asyncio.sleep(1)  # Wait before retrying
    
    async def _process_queued_message(self, message: WebSocketMessage) -> None:
        """Process queued message"""
        try:
            await self.send_message(message.connection_id, message.content)
        except Exception as e:
            self.logger.error(f"Error processing queued message: {e}")
    
    async def _process_queued_broadcast(self, message: Dict[str, Any]) -> None:
        """Process queued broadcast"""
        try:
            exclude_connection_id = message.get("exclude_connection_id")
            await self.broadcast(message.get("content", {}), exclude_connection_id)
        except Exception as e:
            self.logger.error(f"Error processing queued broadcast: {e}")
    
    async def close(self):
        """Close WebSocket manager and cleanup"""
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Disconnect all connections
        await self.disconnect_all()
        
        self.logger.info("WebSocket manager closed")


# Import defaultdict for type hints
from collections import defaultdict
