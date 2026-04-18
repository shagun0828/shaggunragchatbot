"""
Real-time Chunk Processor for Phase 4.2
Handles streaming chunk processing with dynamic optimization
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import deque
import json

# Import from previous phases
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "phase-4.1-advanced-chunking-embedding" / "src"))

from chunkers.enhanced_semantic_chunker import EnhancedSemanticChunker
from chunkers.mutual_fund_chunker_v2 import MutualFundChunkerV2
from models.chunk import Chunk


class ProcessingMode(Enum):
    """Processing modes for real-time chunking"""
    STREAMING = "streaming"
    BATCH = "batch"
    HYBRID = "hybrid"


@dataclass
class ProcessingMetrics:
    """Real-time processing metrics"""
    chunks_processed: int = 0
    processing_time: float = 0.0
    avg_chunk_size: float = 0.0
    quality_score: float = 0.0
    throughput: float = 0.0  # chunks per second
    memory_usage: float = 0.0
    error_count: int = 0
    last_update: float = field(default_factory=time.time)


@dataclass
class ChunkingRequest:
    """Chunking request for real-time processing"""
    id: str
    text: str
    metadata: Dict[str, Any]
    priority: int = 1
    timestamp: float = field(default_factory=time.time)
    callback: Optional[callable] = None


class RealtimeChunkProcessor:
    """Real-time chunk processor with dynamic optimization"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        
        # Initialize chunkers from Phase 4.1
        self.semantic_chunker = EnhancedSemanticChunker()
        self.mutual_fund_chunker = MutualFundChunkerV2()
        
        # Processing queues
        self.request_queue = asyncio.Queue(maxsize=self.config['queue_size'])
        self.priority_queue = asyncio.PriorityQueue(maxsize=self.config['priority_queue_size'])
        self.result_queue = asyncio.Queue(maxsize=self.config['result_queue_size'])
        
        # Processing state
        self.is_running = False
        self.processing_mode = ProcessingMode.HYBRID
        self.metrics = ProcessingMetrics()
        
        # Dynamic optimization parameters
        self.adaptive_threshold = self.config['adaptive_threshold']
        self.batch_size = self.config['batch_size']
        self.max_concurrent = self.config['max_concurrent']
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.quality_history = deque(maxlen=100)
        self.error_history = deque(maxlen=50)
        
        # Worker tasks
        self.workers = []
        self.monitor_task = None
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'queue_size': 1000,
            'priority_queue_size': 100,
            'result_queue_size': 500,
            'batch_size': 10,
            'max_concurrent': 4,
            'adaptive_threshold': True,
            'quality_threshold': 0.7,
            'max_chunk_size': 1000,
            'min_chunk_size': 50,
            'processing_timeout': 30.0,
            'monitoring_interval': 5.0,
            'optimization_interval': 60.0
        }
    
    async def start(self) -> None:
        """Start the real-time chunk processor"""
        if self.is_running:
            self.logger.warning("Processor is already running")
            return
        
        self.logger.info("Starting real-time chunk processor")
        self.is_running = True
        
        # Start worker tasks
        for i in range(self.max_concurrent):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        
        # Start monitoring task
        self.monitor_task = asyncio.create_task(self._monitor_performance())
        
        # Start optimization task
        optimization_task = asyncio.create_task(self._optimize_parameters())
        
        self.logger.info(f"Started {len(self.workers)} workers and monitoring")
    
    async def stop(self) -> None:
        """Stop the real-time chunk processor"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping real-time chunk processor")
        self.is_running = False
        
        # Cancel all tasks
        for worker in self.workers:
            worker.cancel()
        
        if self.monitor_task:
            self.monitor_task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.workers, return_exceptions=True)
        if self.monitor_task:
            await self.monitor_task
        
        self.logger.info("Real-time chunk processor stopped")
    
    async def submit_request(self, request: ChunkingRequest) -> str:
        """Submit a chunking request for processing"""
        if not self.is_running:
            raise RuntimeError("Processor is not running")
        
        # Add to appropriate queue based on priority
        if request.priority > 1:
            await self.priority_queue.put((request.priority, request))
        else:
            await self.request_queue.put(request)
        
        self.logger.debug(f"Submitted request {request.id} with priority {request.priority}")
        return request.id
    
    async def get_result(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Get processing result"""
        try:
            result = await asyncio.wait_for(self.result_queue.get(), timeout=timeout)
            return result
        except asyncio.TimeoutError:
            return None
    
    async def _worker(self, worker_name: str) -> None:
        """Worker task for processing chunking requests"""
        self.logger.info(f"Worker {worker_name} started")
        
        while self.is_running:
            try:
                # Get next request (check priority queue first)
                request = None
                
                try:
                    _, request = await asyncio.wait_for(
                        self.priority_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    try:
                        request = await asyncio.wait_for(
                            self.request_queue.get(), timeout=1.0
                        )
                    except asyncio.TimeoutError:
                        continue
                
                if request is None:
                    continue
                
                # Process the request
                start_time = time.time()
                try:
                    result = await self._process_request(request)
                    processing_time = time.time() - start_time
                    
                    # Update metrics
                    self._update_metrics(result, processing_time)
                    
                    # Put result in result queue
                    await self.result_queue.put(result)
                    
                    # Call callback if provided
                    if request.callback:
                        await self._safe_callback(request.callback, result)
                    
                except Exception as e:
                    self.logger.error(f"Error processing request {request.id}: {e}")
                    self.metrics.error_count += 1
                    self.error_history.append({
                        'request_id': request.id,
                        'error': str(e),
                        'timestamp': time.time()
                    })
                    
                    # Put error result
                    error_result = {
                        'request_id': request.id,
                        'status': 'error',
                        'error': str(e),
                        'timestamp': time.time()
                    }
                    await self.result_queue.put(error_result)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Worker {worker_name} error: {e}")
                await asyncio.sleep(1.0)
        
        self.logger.info(f"Worker {worker_name} stopped")
    
    async def _process_request(self, request: ChunkingRequest) -> Dict[str, Any]:
        """Process a single chunking request"""
        self.logger.debug(f"Processing request {request.id}")
        
        # Determine chunking strategy based on content
        chunking_strategy = self._determine_strategy(request.text, request.metadata)
        
        # Apply chunking
        if chunking_strategy == 'mutual_fund':
            chunks = self._process_mutual_fund_chunking(request)
        elif chunking_strategy == 'semantic':
            chunks = self._process_semantic_chunking(request)
        else:
            chunks = self._process_hybrid_chunking(request)
        
        # Quality assessment
        quality_score = self._assess_chunk_quality(chunks)
        
        # Optimize chunks if needed
        if quality_score < self.config['quality_threshold']:
            chunks = self._optimize_chunks(chunks)
        
        # Create result
        result = {
            'request_id': request.id,
            'status': 'success',
            'chunks': [self._chunk_to_dict(chunk) for chunk in chunks],
            'metadata': {
                'chunk_count': len(chunks),
                'strategy': chunking_strategy,
                'quality_score': quality_score,
                'processing_time': time.time() - request.timestamp
            },
            'timestamp': time.time()
        }
        
        return result
    
    def _determine_strategy(self, text: str, metadata: Dict[str, Any]) -> str:
        """Determine optimal chunking strategy"""
        # Check if it's mutual fund data
        financial_keywords = ['fund', 'nav', 'returns', 'aum', 'holding', 'allocation']
        text_lower = text.lower()
        
        financial_score = sum(1 for keyword in financial_keywords if keyword in text_lower)
        
        if financial_score >= 3 or metadata.get('section_type'):
            return 'mutual_fund'
        elif len(text) > 500:
            return 'semantic'
        else:
            return 'hybrid'
    
    def _process_mutual_fund_chunking(self, request: ChunkingRequest) -> List[Chunk]:
        """Process mutual fund chunking"""
        # Create fund data structure
        fund_data = {
            'fund_name': request.metadata.get('fund_name', 'Unknown Fund'),
            'description': request.text,
            **request.metadata
        }
        
        return self.mutual_fund_chunker.chunk_fund_data_advanced(fund_data)
    
    def _process_semantic_chunking(self, request: ChunkingRequest) -> List[Chunk]:
        """Process semantic chunking"""
        return self.semantic_chunker.chunk_semantic_enhanced(request.text, request.metadata)
    
    def _process_hybrid_chunking(self, request: ChunkingRequest) -> List[Chunk]:
        """Process hybrid chunking"""
        # Try semantic first, fall back to simple splitting if needed
        try:
            chunks = self.semantic_chunker.chunk_semantic_enhanced(request.text, request.metadata)
            if len(chunks) == 0:
                # Fallback to simple splitting
                chunks = self._simple_chunking(request.text, request.metadata)
            return chunks
        except Exception:
            return self._simple_chunking(request.text, request.metadata)
    
    def _simple_chunking(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Simple fallback chunking"""
        sentences = text.split('. ')
        chunks = []
        
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk + sentence) <= self.config['max_chunk_size']:
                current_chunk += sentence + ". "
            else:
                if current_chunk.strip():
                    chunk = Chunk(
                        text=current_chunk.strip(),
                        metadata={**metadata, 'chunk_type': 'simple'}
                    )
                    chunks.append(chunk)
                current_chunk = sentence + ". "
        
        if current_chunk.strip():
            chunk = Chunk(
                text=current_chunk.strip(),
                metadata={**metadata, 'chunk_type': 'simple'}
            )
            chunks.append(chunk)
        
        return chunks
    
    def _assess_chunk_quality(self, chunks: List[Chunk]) -> float:
        """Assess quality of generated chunks"""
        if not chunks:
            return 0.0
        
        quality_scores = []
        
        for chunk in chunks:
            score = 0.0
            
            # Length score
            if self.config['min_chunk_size'] <= len(chunk.text) <= self.config['max_chunk_size']:
                score += 0.3
            
            # Content score
            if chunk.metadata.get('chunk_type') in ['structured', 'semantic']:
                score += 0.3
            elif chunk.metadata.get('chunk_type') == 'simple':
                score += 0.1
            
            # Financial content score
            financial_keywords = ['%', 'nav', 'return', 'fund', 'allocation']
            text_lower = chunk.text.lower()
            financial_count = sum(1 for keyword in financial_keywords if keyword in text_lower)
            score += min(financial_count * 0.1, 0.4)
            
            quality_scores.append(score)
        
        return np.mean(quality_scores)
    
    def _optimize_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Optimize chunks for better quality"""
        optimized_chunks = []
        
        for chunk in chunks:
            # Remove redundant whitespace
            text = ' '.join(chunk.text.split())
            
            # Check if chunk is still meaningful
            if len(text) >= self.config['min_chunk_size']:
                chunk.text = text
                chunk.metadata['optimized'] = True
                optimized_chunks.append(chunk)
        
        return optimized_chunks
    
    def _chunk_to_dict(self, chunk: Chunk) -> Dict[str, Any]:
        """Convert chunk to dictionary"""
        return {
            'id': chunk.id,
            'text': chunk.text,
            'metadata': chunk.metadata,
            'text_length': len(chunk.text)
        }
    
    def _update_metrics(self, result: Dict[str, Any], processing_time: float) -> None:
        """Update processing metrics"""
        chunk_count = result['metadata']['chunk_count']
        quality_score = result['metadata']['quality_score']
        
        self.metrics.chunks_processed += chunk_count
        self.metrics.processing_time += processing_time
        self.metrics.avg_chunk_size = np.mean([
            chunk['text_length'] for chunk in result['chunks']
        ])
        self.metrics.quality_score = (self.metrics.quality_score + quality_score) / 2
        self.metrics.throughput = self.metrics.chunks_processed / (self.metrics.processing_time + 0.001)
        self.metrics.last_update = time.time()
        
        # Add to history
        self.performance_history.append({
            'timestamp': time.time(),
            'processing_time': processing_time,
            'chunk_count': chunk_count,
            'quality_score': quality_score
        })
        
        self.quality_history.append(quality_score)
    
    async def _monitor_performance(self) -> None:
        """Monitor performance and adjust parameters"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config['monitoring_interval'])
                
                # Calculate current performance metrics
                current_time = time.time()
                recent_performance = [
                    p for p in self.performance_history
                    if current_time - p['timestamp'] < 300  # Last 5 minutes
                ]
                
                if len(recent_performance) > 0:
                    avg_processing_time = np.mean([p['processing_time'] for p in recent_performance])
                    avg_quality = np.mean([p['quality_score'] for p in recent_performance])
                    
                    self.logger.info(
                        f"Performance - Avg processing time: {avg_processing_time:.3f}s, "
                        f"Avg quality: {avg_quality:.3f}, "
                        f"Throughput: {self.metrics.throughput:.2f} chunks/s"
                    )
                    
                    # Trigger optimization if performance is poor
                    if avg_processing_time > 5.0 or avg_quality < 0.6:
                        await self._trigger_optimization()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
    
    async def _optimize_parameters(self) -> None:
        """Periodic parameter optimization"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config['optimization_interval'])
                
                if self.adaptive_threshold:
                    await self._optimize_threshold()
                
                await self._optimize_batch_size()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Parameter optimization error: {e}")
    
    async def _optimize_threshold(self) -> None:
        """Optimize adaptive threshold based on recent performance"""
        if len(self.quality_history) < 10:
            return
        
        recent_quality = list(self.quality_history)[-10:]
        avg_quality = np.mean(recent_quality)
        
        # Adjust threshold based on quality
        if avg_quality > 0.8:
            # High quality, can be more selective
            self.config['quality_threshold'] = min(0.9, self.config['quality_threshold'] + 0.05)
        elif avg_quality < 0.6:
            # Low quality, be less selective
            self.config['quality_threshold'] = max(0.5, self.config['quality_threshold'] - 0.05)
        
        self.logger.debug(f"Adjusted quality threshold to {self.config['quality_threshold']:.3f}")
    
    async def _optimize_batch_size(self) -> None:
        """Optimize batch size based on throughput"""
        if len(self.performance_history) < 20:
            return
        
        recent_performance = list(self.performance_history)[-20:]
        
        # Calculate optimal batch size based on processing time
        avg_processing_time = np.mean([p['processing_time'] for p in recent_performance])
        
        if avg_processing_time < 1.0:
            # Fast processing, can increase batch size
            self.batch_size = min(50, self.batch_size + 5)
        elif avg_processing_time > 3.0:
            # Slow processing, decrease batch size
            self.batch_size = max(5, self.batch_size - 5)
        
        self.logger.debug(f"Adjusted batch size to {self.batch_size}")
    
    async def _trigger_optimization(self) -> None:
        """Trigger immediate optimization"""
        self.logger.info("Triggering immediate optimization")
        
        # Reset metrics
        old_metrics = self.metrics
        self.metrics = ProcessingMetrics()
        
        # Adjust parameters
        await self._optimize_threshold()
        await self._optimize_batch_size()
    
    async def _safe_callback(self, callback: callable, result: Dict[str, Any]) -> None:
        """Safely execute callback"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(result)
            else:
                callback(result)
        except Exception as e:
            self.logger.error(f"Callback execution error: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current processing metrics"""
        return {
            'chunks_processed': self.metrics.chunks_processed,
            'processing_time': self.metrics.processing_time,
            'avg_chunk_size': self.metrics.avg_chunk_size,
            'quality_score': self.metrics.quality_score,
            'throughput': self.metrics.throughput,
            'error_count': self.metrics.error_count,
            'queue_sizes': {
                'request_queue': self.request_queue.qsize(),
                'priority_queue': self.priority_queue.qsize(),
                'result_queue': self.result_queue.qsize()
            },
            'config': {
                'batch_size': self.batch_size,
                'max_concurrent': self.max_concurrent,
                'quality_threshold': self.config['quality_threshold']
            }
        }
    
    async def stream_process(self, text_stream: AsyncGenerator[str, None]) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream processing for continuous text input"""
        request_id = 0
        
        async for text in text_stream:
            request_id += 1
            request = ChunkingRequest(
                id=f"stream-{request_id}",
                text=text,
                metadata={'streaming': True, 'sequence': request_id}
            )
            
            # Submit and wait for result
            await self.submit_request(request)
            result = await self.get_result(timeout=10.0)
            
            if result:
                yield result
