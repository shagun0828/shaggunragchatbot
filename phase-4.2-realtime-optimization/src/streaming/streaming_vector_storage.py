"""
Streaming Vector Storage for Phase 4.2
Real-time vector storage with streaming capabilities and performance optimization
"""

import asyncio
import logging
import time
import json
import numpy as np
from typing import List, Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import uuid
from pathlib import Path

# Import from previous phases
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "phase-4.1-advanced-chunking-embedding" / "src"))

from storage.advanced_vector_storage import AdvancedVectorStorage
from embedders.embedding_quality_checker import EmbeddingQualityChecker
from models.chunk import Chunk


class StorageMode(Enum):
    """Storage modes for streaming"""
    STREAMING = "streaming"
    BATCH = "batch"
    HYBRID = "hybrid"


class StreamStatus(Enum):
    """Stream processing status"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class StreamMetrics:
    """Metrics for streaming operations"""
    vectors_streamed: int = 0
    streaming_time: float = 0.0
    throughput: float = 0.0
    quality_score: float = 0.0
    buffer_utilization: float = 0.0
    error_count: int = 0
    last_update: float = field(default_factory=time.time)


@dataclass
class StreamRequest:
    """Request for streaming vector storage"""
    id: str
    chunks: List[Chunk]
    embeddings: np.ndarray
    metadata: Dict[str, Any]
    priority: int = 1
    timestamp: float = field(default_factory=time.time)
    callback: Optional[callable] = None


class StreamingVectorStorage:
    """Streaming vector storage with real-time capabilities"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        
        # Initialize advanced storage from Phase 4.1
        self.vector_storage = AdvancedVectorStorage()
        self.quality_checker = EmbeddingQualityChecker()
        
        # Streaming configuration
        self.storage_mode = StorageMode.HYBRID
        self.buffer_size = self.config['buffer_size']
        self.batch_size = self.config['batch_size']
        self.flush_interval = self.config['flush_interval']
        
        # Streaming state
        self.is_streaming = False
        self.stream_status = StreamStatus.ACTIVE
        self.metrics = StreamMetrics()
        
        # Buffers and queues
        self.stream_buffer = deque(maxlen=self.buffer_size)
        self.priority_queue = asyncio.PriorityQueue(maxsize=1000)
        self.result_queue = asyncio.Queue(maxsize=500)
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.quality_history = deque(maxlen=100)
        self.error_history = deque(maxlen=50)
        
        # Background tasks
        self.stream_processor_task = None
        self.flush_task = None
        self.monitor_task = None
        
        # Storage paths
        self.storage_path = Path("data/streaming_storage")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Stream sessions
        self.active_sessions = {}
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'buffer_size': 1000,
            'batch_size': 50,
            'flush_interval': 5.0,  # seconds
            'max_concurrent_streams': 10,
            'quality_threshold': 0.7,
            'compression_enabled': True,
            'indexing_enabled': True,
            'monitoring_interval': 10.0,
            'auto_cleanup': True,
            'cleanup_interval': 3600.0,  # 1 hour
            'max_session_duration': 3600.0  # 1 hour
        }
    
    async def start_streaming(self) -> None:
        """Start streaming vector storage"""
        if self.is_streaming:
            self.logger.warning("Streaming is already active")
            return
        
        self.logger.info("Starting streaming vector storage")
        self.is_streaming = True
        self.stream_status = StreamStatus.ACTIVE
        
        # Start background tasks
        self.stream_processor_task = asyncio.create_task(self._stream_processor())
        self.flush_task = asyncio.create_task(self._periodic_flush())
        self.monitor_task = asyncio.create_task(self._monitor_streaming())
        
        self.logger.info("Streaming vector storage started")
    
    async def stop_streaming(self) -> None:
        """Stop streaming vector storage"""
        if not self.is_streaming:
            return
        
        self.logger.info("Stopping streaming vector storage")
        self.is_streaming = False
        self.stream_status = StreamStatus.COMPLETED
        
        # Cancel background tasks
        if self.stream_processor_task:
            self.stream_processor_task.cancel()
        if self.flush_task:
            self.flush_task.cancel()
        if self.monitor_task:
            self.monitor_task.cancel()
        
        # Final flush
        await self._flush_buffer()
        
        # Wait for tasks to complete
        tasks = [self.stream_processor_task, self.flush_task, self.monitor_task]
        await asyncio.gather(*[t for t in tasks if t is not None], return_exceptions=True)
        
        self.logger.info("Streaming vector storage stopped")
    
    async def stream_vectors(self, chunks: List[Chunk], embeddings: np.ndarray, 
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """Stream vectors for storage"""
        if not self.is_streaming:
            raise RuntimeError("Streaming is not active")
        
        stream_id = str(uuid.uuid4())
        
        # Create stream request
        request = StreamRequest(
            id=stream_id,
            chunks=chunks,
            embeddings=embeddings,
            metadata=metadata or {},
            priority=1
        )
        
        # Add to priority queue
        await self.priority_queue.put((request.priority, request))
        
        # Track session
        self.active_sessions[stream_id] = {
            'start_time': time.time(),
            'chunk_count': len(chunks),
            'status': StreamStatus.ACTIVE.value
        }
        
        self.logger.debug(f"Started stream {stream_id} with {len(chunks)} chunks")
        return stream_id
    
    async def get_stream_result(self, stream_id: str, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Get result for a specific stream"""
        try:
            while True:
                # Check result queue
                try:
                    result = await asyncio.wait_for(self.result_queue.get(), timeout=1.0)
                    if result.get('stream_id') == stream_id:
                        return result
                    else:
                        # Put back different result
                        await self.result_queue.put(result)
                except asyncio.TimeoutError:
                    # Check session status
                    if stream_id in self.active_sessions:
                        session = self.active_sessions[stream_id]
                        if session['status'] == StreamStatus.FAILED.value:
                            return {'stream_id': stream_id, 'status': 'failed', 'error': 'Stream processing failed'}
                    else:
                        return None
                
                if timeout and (time.time() - timeout) > 0:
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error getting stream result {stream_id}: {e}")
            return None
    
    async def _stream_processor(self) -> None:
        """Main stream processor"""
        self.logger.info("Stream processor started")
        
        while self.is_streaming:
            try:
                # Get next request from priority queue
                try:
                    priority, request = await asyncio.wait_for(
                        self.priority_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process the request
                await self._process_stream_request(request)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Stream processor error: {e}")
                self.metrics.error_count += 1
                await asyncio.sleep(1.0)
        
        self.logger.info("Stream processor stopped")
    
    async def _process_stream_request(self, request: StreamRequest) -> None:
        """Process a single stream request"""
        start_time = time.time()
        
        try:
            # Quality check
            quality_report = self.quality_checker.check_embedding_quality(
                request.embeddings, request.chunks
            )
            
            # Update session
            if request.id in self.active_sessions:
                self.active_sessions[request.id]['quality_score'] = quality_report.overall_score
            
            # Add to buffer if quality is acceptable
            if quality_report.overall_score >= self.config['quality_threshold']:
                await self._add_to_buffer(request, quality_report)
            else:
                # Try to improve quality
                improved_request = await self._improve_stream_quality(request, quality_report)
                if improved_request:
                    await self._add_to_buffer(improved_request, quality_report)
                else:
                    # Reject low quality stream
                    await self._reject_stream(request, "Low quality score")
                    return
            
            # Update metrics
            processing_time = time.time() - start_time
            self._update_stream_metrics(request, processing_time, quality_report)
            
            # Send intermediate result
            await self._send_stream_result(request, 'processing', {
                'quality_score': quality_report.overall_score,
                'processing_time': processing_time
            })
            
        except Exception as e:
            self.logger.error(f"Error processing stream request {request.id}: {e}")
            await self._reject_stream(request, str(e))
    
    async def _add_to_buffer(self, request: StreamRequest, quality_report) -> None:
        """Add request to buffer for batch processing"""
        buffer_item = {
            'request': request,
            'quality_report': quality_report,
            'timestamp': time.time()
        }
        
        self.stream_buffer.append(buffer_item)
        
        # Trigger flush if buffer is full
        if len(self.stream_buffer) >= self.batch_size:
            await self._flush_buffer()
    
    async def _improve_stream_quality(self, request: StreamRequest, 
                                    quality_report) -> Optional[StreamRequest]:
        """Attempt to improve stream quality"""
        # This would implement quality improvement logic from Phase 4.1
        # For now, return None (no improvement)
        return None
    
    async def _reject_stream(self, request: StreamRequest, reason: str) -> None:
        """Reject a stream request"""
        await self._send_stream_result(request, 'rejected', {'reason': reason})
        
        # Update session
        if request.id in self.active_sessions:
            self.active_sessions[request.id]['status'] = StreamStatus.FAILED.value
            self.active_sessions[request.id]['error'] = reason
    
    async def _flush_buffer(self) -> None:
        """Flush buffer to storage"""
        if len(self.stream_buffer) == 0:
            return
        
        self.logger.debug(f"Flushing buffer with {len(self.stream_buffer)} items")
        
        # Collect batch items
        batch_items = list(self.stream_buffer)
        self.stream_buffer.clear()
        
        # Prepare data for storage
        all_chunks = []
        all_embeddings = []
        stream_ids = set()
        
        for item in batch_items:
            request = item['request']
            all_chunks.extend(request.chunks)
            all_embeddings.append(request.embeddings)
            stream_ids.add(request.id)
        
        if not all_chunks or not all_embeddings:
            return
        
        # Combine embeddings
        combined_embeddings = np.vstack(all_embeddings)
        
        # Store using advanced storage
        processed_data = [{'chunks': all_chunks}]
        storage_results = await self.vector_storage.store_embeddings_with_quality_assurance(processed_data)
        
        # Send completion results
        for stream_id in stream_ids:
            await self._send_stream_result(stream_id, 'completed', {
                'storage_results': storage_results,
                'chunks_stored': len(all_chunks)
            })
            
            # Update session
            if stream_id in self.active_sessions:
                self.active_sessions[stream_id]['status'] = StreamStatus.COMPLETED.value
                self.active_sessions[stream_id]['end_time'] = time.time()
        
        self.logger.info(f"Flushed {len(all_chunks)} chunks to storage")
    
    async def _periodic_flush(self) -> None:
        """Periodic buffer flush"""
        while self.is_streaming:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_buffer()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Periodic flush error: {e}")
    
    async def _send_stream_result(self, request: StreamRequest, status: str, 
                                data: Optional[Dict[str, Any]] = None) -> None:
        """Send stream result"""
        result = {
            'stream_id': request.id,
            'status': status,
            'timestamp': time.time(),
            **(data or {})
        }
        
        await self.result_queue.put(result)
        
        # Call callback if provided
        if request.callback:
            try:
                if asyncio.iscoroutinefunction(request.callback):
                    await request.callback(result)
                else:
                    request.callback(result)
            except Exception as e:
                self.logger.error(f"Stream callback error: {e}")
    
    def _update_stream_metrics(self, request: StreamRequest, processing_time: float, 
                               quality_report) -> None:
        """Update streaming metrics"""
        self.metrics.vectors_streamed += len(request.chunks)
        self.metrics.streaming_time += processing_time
        self.metrics.quality_score = quality_report.overall_score
        self.metrics.throughput = self.metrics.vectors_streamed / (self.metrics.streaming_time + 0.001)
        self.metrics.buffer_utilization = len(self.stream_buffer) / self.buffer_size
        self.metrics.last_update = time.time()
        
        # Add to history
        self.performance_history.append({
            'timestamp': time.time(),
            'stream_id': request.id,
            'processing_time': processing_time,
            'chunk_count': len(request.chunks),
            'quality_score': quality_report.overall_score
        })
        
        self.quality_history.append(quality_report.overall_score)
    
    async def _monitor_streaming(self) -> None:
        """Monitor streaming performance"""
        while self.is_streaming:
            try:
                await asyncio.sleep(self.config['monitoring_interval'])
                
                # Calculate current metrics
                current_time = time.time()
                recent_performance = [
                    p for p in self.performance_history
                    if current_time - p['timestamp'] < 300  # Last 5 minutes
                ]
                
                if len(recent_performance) > 0:
                    avg_processing_time = np.mean([p['processing_time'] for p in recent_performance])
                    avg_quality = np.mean([p['quality_score'] for p in recent_performance])
                    
                    self.logger.info(
                        f"Streaming Performance - Avg processing time: {avg_processing_time:.3f}s, "
                        f"Avg quality: {avg_quality:.3f}, "
                        f"Throughput: {self.metrics.throughput:.2f} chunks/s, "
                        f"Buffer utilization: {self.metrics.buffer_utilization:.2%}"
                    )
                
                # Auto cleanup old sessions
                if self.config['auto_cleanup']:
                    await self._cleanup_old_sessions()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Streaming monitoring error: {e}")
    
    async def _cleanup_old_sessions(self) -> None:
        """Clean up old streaming sessions"""
        current_time = time.time()
        expired_sessions = []
        
        for session_id, session in self.active_sessions.items():
            if current_time - session.get('start_time', 0) > self.config['max_session_duration']:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.active_sessions[session_id]
            self.logger.debug(f"Cleaned up expired session {session_id}")
        
        if expired_sessions:
            self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    async def stream_search(self, query_embedding: np.ndarray, top_k: int = 5, 
                           filters: Optional[Dict[str, Any]] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream search results"""
        # This would implement streaming search functionality
        # For now, yield empty results
        yield {
            'query_embedding_dim': query_embedding.shape[0],
            'top_k': top_k,
            'filters': filters,
            'results': [],
            'timestamp': time.time()
        }
    
    def get_streaming_metrics(self) -> Dict[str, Any]:
        """Get comprehensive streaming metrics"""
        return {
            'is_streaming': self.is_streaming,
            'stream_status': self.stream_status.value,
            'storage_mode': self.storage_mode.value,
            'metrics': {
                'vectors_streamed': self.metrics.vectors_streamed,
                'streaming_time': self.metrics.streaming_time,
                'throughput': self.metrics.throughput,
                'quality_score': self.metrics.quality_score,
                'buffer_utilization': self.metrics.buffer_utilization,
                'error_count': self.metrics.error_count
            },
            'active_sessions': len(self.active_sessions),
            'buffer_size': len(self.stream_buffer),
            'queue_sizes': {
                'priority_queue': self.priority_queue.qsize(),
                'result_queue': self.result_queue.qsize()
            },
            'config': {
                'buffer_size': self.buffer_size,
                'batch_size': self.batch_size,
                'flush_interval': self.flush_interval,
                'quality_threshold': self.config['quality_threshold']
            }
        }
    
    async def create_stream_session(self, session_config: Dict[str, Any]) -> str:
        """Create a new streaming session"""
        session_id = str(uuid.uuid4())
        
        self.active_sessions[session_id] = {
            'start_time': time.time(),
            'config': session_config,
            'status': StreamStatus.ACTIVE.value,
            'chunk_count': 0,
            'quality_score': 0.0
        }
        
        return session_id
    
    async def close_stream_session(self, session_id: str) -> Dict[str, Any]:
        """Close a streaming session and return summary"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        session['end_time'] = time.time()
        session['duration'] = session['end_time'] - session['start_time']
        session['status'] = StreamStatus.COMPLETED.value
        
        # Remove from active sessions
        del self.active_sessions[session_id]
        
        return session
