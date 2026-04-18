#!/usr/bin/env python3
"""
Standalone Main entry point for Phase 4.2: Real-time Optimization
Demonstrates real-time chunking and embedding optimization without dependencies
"""

import asyncio
import logging
import sys
import time
import numpy as np
from typing import Dict, Any, List
from dataclasses import dataclass
from collections import deque
import uuid
import json


@dataclass
class Chunk:
    """Simple chunk representation"""
    id: str
    text: str
    metadata: Dict[str, Any]


@dataclass
class ProcessingMetrics:
    """Processing metrics"""
    chunks_processed: int = 0
    processing_time: float = 0.0
    quality_score: float = 0.0
    throughput: float = 0.0


class SimpleChunkProcessor:
    """Simple real-time chunk processor"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.request_queue = asyncio.Queue(maxsize=100)
        self.result_queue = asyncio.Queue(maxsize=100)
        self.workers = []
        self.metrics = ProcessingMetrics()
        self.is_running = False
    
    async def start(self):
        """Start the processor"""
        self.is_running = True
        for i in range(2):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        self.logger.info("Simple chunk processor started")
    
    async def stop(self):
        """Stop the processor"""
        self.is_running = False
        for worker in self.workers:
            worker.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.logger.info("Simple chunk processor stopped")
    
    async def submit_request(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """Submit chunking request"""
        request_id = str(uuid.uuid4())
        request = {
            'id': request_id,
            'text': text,
            'metadata': metadata or {},
            'timestamp': time.time()
        }
        await self.request_queue.put(request)
        return request_id
    
    async def get_result(self, timeout: float = 10.0) -> Dict[str, Any]:
        """Get processing result"""
        try:
            result = await asyncio.wait_for(self.result_queue.get(), timeout=timeout)
            return result
        except asyncio.TimeoutError:
            return None
    
    async def _worker(self, name: str):
        """Worker task"""
        while self.is_running:
            try:
                request = await asyncio.wait_for(self.request_queue.get(), timeout=1.0)
                
                start_time = time.time()
                chunks = self._process_text(request['text'], request['metadata'])
                processing_time = time.time() - start_time
                
                quality_score = self._assess_quality(chunks)
                
                result = {
                    'request_id': request['id'],
                    'status': 'success',
                    'chunks': [{'id': chunk.id, 'text': chunk.text, 'metadata': chunk.metadata} for chunk in chunks],
                    'metadata': {
                        'chunk_count': len(chunks),
                        'quality_score': quality_score,
                        'processing_time': processing_time
                    }
                }
                
                await self.result_queue.put(result)
                self._update_metrics(len(chunks), processing_time, quality_score)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Worker {name} error: {e}")
    
    def _process_text(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Simple text processing"""
        sentences = text.split('. ')
        chunks = []
        
        current_chunk = ""
        for i, sentence in enumerate(sentences):
            if len(current_chunk + sentence) <= 500:
                current_chunk += sentence + ". "
            else:
                if current_chunk.strip():
                    chunk = Chunk(
                        id=f"chunk-{uuid.uuid4()}",
                        text=current_chunk.strip(),
                        metadata={**metadata, 'chunk_index': len(chunks)}
                    )
                    chunks.append(chunk)
                current_chunk = sentence + ". "
        
        if current_chunk.strip():
            chunk = Chunk(
                id=f"chunk-{uuid.uuid4()}",
                text=current_chunk.strip(),
                metadata={**metadata, 'chunk_index': len(chunks)}
            )
            chunks.append(chunk)
        
        return chunks
    
    def _assess_quality(self, chunks: List[Chunk]) -> float:
        """Simple quality assessment"""
        if not chunks:
            return 0.0
        
        scores = []
        for chunk in chunks:
            score = 0.0
            if 50 <= len(chunk.text) <= 500:
                score += 0.4
            if any(word in chunk.text.lower() for word in ['fund', 'nav', 'return', '%']):
                score += 0.3
            if len(chunk.text.split()) > 10:
                score += 0.3
            scores.append(score)
        
        return np.mean(scores)
    
    def _update_metrics(self, chunk_count: int, processing_time: float, quality_score: float):
        """Update metrics"""
        self.metrics.chunks_processed += chunk_count
        self.metrics.processing_time += processing_time
        self.metrics.quality_score = (self.metrics.quality_score + quality_score) / 2
        self.metrics.throughput = self.metrics.chunks_processed / (self.metrics.processing_time + 0.001)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return {
            'chunks_processed': self.metrics.chunks_processed,
            'processing_time': self.metrics.processing_time,
            'quality_score': self.metrics.quality_score,
            'throughput': self.metrics.throughput,
            'queue_sizes': {
                'request_queue': self.request_queue.qsize(),
                'result_queue': self.result_queue.qsize()
            }
        }


class SimpleEmbeddingOptimizer:
    """Simple embedding optimizer"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.current_model = "simple-embeddings"
        self.metrics = {
            'embeddings_generated': 0,
            'avg_quality': 0.0,
            'optimization_count': 0
        }
    
    async def optimize_embeddings(self, chunks: List[Chunk]) -> tuple:
        """Simple embedding optimization"""
        # Simulate embedding generation
        await asyncio.sleep(0.1 * len(chunks))
        
        # Generate random embeddings (simulated)
        embeddings = np.random.rand(len(chunks), 384)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Quality assessment
        quality_score = np.random.uniform(0.6, 0.9)
        
        # Update metrics
        self.metrics['embeddings_generated'] += len(chunks)
        self.metrics['avg_quality'] = (self.metrics['avg_quality'] + quality_score) / 2
        
        optimization_metadata = {
            'model_used': self.current_model,
            'quality_score': quality_score,
            'chunks_processed': len(chunks),
            'embedding_dimension': 384
        }
        
        return embeddings, optimization_metadata


class SimpleVectorStorage:
    """Simple vector storage"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.storage = []
        self.metrics = {
            'vectors_stored': 0,
            'storage_time': 0.0
        }
    
    async def store_vectors(self, chunks: List[Chunk], embeddings: np.ndarray) -> Dict[str, Any]:
        """Store vectors"""
        start_time = time.time()
        
        # Simulate storage
        await asyncio.sleep(0.05 * len(chunks))
        
        # Store in memory
        for i, chunk in enumerate(chunks):
            self.storage.append({
                'chunk_id': chunk.id,
                'text': chunk.text,
                'embedding': embeddings[i].tolist(),
                'metadata': chunk.metadata
            })
        
        storage_time = time.time() - start_time
        self.metrics['vectors_stored'] += len(chunks)
        self.metrics['storage_time'] += storage_time
        
        return {
            'status': 'success',
            'vectors_stored': len(chunks),
            'storage_time': storage_time,
            'total_stored': len(self.storage)
        }


class RealtimeOptimizationSystem:
    """Main system for Phase 4.2 demonstration"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.chunk_processor = SimpleChunkProcessor()
        self.embedding_optimizer = SimpleEmbeddingOptimizer()
        self.vector_storage = SimpleVectorStorage()
        self.is_running = False
        self.start_time = None
        self.system_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0
        }
    
    async def start(self):
        """Start the system"""
        self.is_running = True
        self.start_time = time.time()
        
        await self.chunk_processor.start()
        
        self.logger.info("Phase 4.2 Real-time Optimization System started")
    
    async def stop(self):
        """Stop the system"""
        self.is_running = False
        await self.chunk_processor.stop()
        
        uptime = time.time() - self.start_time if self.start_time else 0
        self.logger.info(f"System stopped after {uptime:.2f} seconds")
    
    async def process_text(self, text: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a single text"""
        # Submit for chunking
        request_id = await self.chunk_processor.submit_request(text, metadata)
        
        # Get chunking result
        chunk_result = await self.chunk_processor.get_result()
        
        if not chunk_result or chunk_result.get('status') != 'success':
            self.system_metrics['failed_requests'] += 1
            return {'status': 'error', 'message': 'Chunking failed'}
        
        # Convert to chunks
        chunks = []
        for chunk_dict in chunk_result['chunks']:
            chunk = Chunk(
                id=chunk_dict['id'],
                text=chunk_dict['text'],
                metadata=chunk_dict['metadata']
            )
            chunks.append(chunk)
        
        # Optimize embeddings
        embeddings, optimization_metadata = await self.embedding_optimizer.optimize_embeddings(chunks)
        
        # Store vectors
        storage_result = await self.vector_storage.store_vectors(chunks, embeddings)
        
        # Combine results
        result = {
            'status': 'success',
            'request_id': request_id,
            'chunking_result': chunk_result,
            'optimization_metadata': optimization_metadata,
            'storage_result': storage_result,
            'timestamp': time.time()
        }
        
        self.system_metrics['total_requests'] += 1
        self.system_metrics['successful_requests'] += 1
        
        return result
    
    async def process_text_stream(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Process multiple texts"""
        results = []
        
        for i, text in enumerate(texts):
            result = await self.process_text(text, {'stream_index': i})
            results.append(result)
        
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            'is_running': self.is_running,
            'uptime': time.time() - self.start_time if self.start_time else 0,
            'system_metrics': self.system_metrics,
            'component_metrics': {
                'chunking': self.chunk_processor.get_metrics(),
                'embedding': self.embedding_optimizer.metrics,
                'storage': self.vector_storage.metrics
            }
        }


async def main():
    """Main demonstration"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Phase 4.2 Real-time Optimization System Demo")
    
    # Initialize system
    system = RealtimeOptimizationSystem()
    
    try:
        # Start system
        await system.start()
        
        # Sample mutual fund data
        sample_data = [
            "HDFC Mid-Cap Fund Direct Growth has generated impressive returns of 24.5% in the last year. The current NAV stands at â¹175.43 with AUM of â¹28,432 Cr.",
            
            "The fund's top holdings include Reliance Industries (8.5%), TCS (7.2%), and HDFC Bank (6.8%). Sector allocation shows 25% in Financial Services and 20% in Technology.",
            
            "Performance metrics show 3-year returns of 18.2% and 5-year returns of 16.8%. The expense ratio is competitive at 1.25% with an exit load of 1%.",
            
            "Risk analysis indicates a Very High risk rating with a beta of 1.2. The fund's standard deviation stands at 18.5%, reflecting mid-cap volatility.",
            
            "Investment objective focuses on long-term capital appreciation by investing in a diversified portfolio of mid-cap companies with growth potential."
        ]
        
        logger.info(f"Processing {len(sample_data)} sample texts")
        
        # Process texts
        results = await system.process_text_stream(sample_data)
        
        # Display results
        logger.info("Processing Results:")
        for i, result in enumerate(results):
            if result['status'] == 'success':
                chunk_count = result['chunking_result']['metadata']['chunk_count']
                quality = result['optimization_metadata']['quality_score']
                storage_time = result['storage_result']['storage_time']
                logger.info(f"  Text {i+1}: {chunk_count} chunks, quality: {quality:.3f}, storage_time: {storage_time:.3f}s")
            else:
                logger.error(f"  Text {i+1}: Failed - {result.get('message', 'Unknown error')}")
        
        # Get system status
        status = system.get_system_status()
        logger.info(f"\nSystem Status:")
        logger.info(f"  Uptime: {status['uptime']:.2f}s")
        logger.info(f"  Total Requests: {status['system_metrics']['total_requests']}")
        logger.info(f"  Success Rate: {status['system_metrics']['successful_requests'] / status['system_metrics']['total_requests']:.1%}")
        logger.info(f"  Chunks Processed: {status['component_metrics']['chunking']['chunks_processed']}")
        logger.info(f"  Throughput: {status['component_metrics']['chunking']['throughput']:.2f} chunks/s")
        logger.info(f"  Avg Quality: {status['component_metrics']['embedding']['avg_quality']:.3f}")
        logger.info(f"  Vectors Stored: {status['component_metrics']['storage']['vectors_stored']}")
        
        # Demonstrate real-time capability
        logger.info("\nDemonstrating real-time processing...")
        
        # Process additional texts while system is running
        additional_texts = [
            "SBI Small Cap Fund Direct Plan has delivered 22.3% returns with NAV of â¹145.67.",
            "ICICI Prudential Technology Fund shows 28.7% returns in the current market conditions."
        ]
        
        realtime_results = await system.process_text_stream(additional_texts)
        logger.info(f"Real-time processing completed: {len(realtime_results)} additional texts processed")
        
        # Wait to show real-time capabilities
        logger.info("System running in real-time mode... (waiting 10 seconds)")
        await asyncio.sleep(10)
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"System error: {e}")
        raise
    finally:
        # Stop system
        await system.stop()
        logger.info("Phase 4.2 Real-time Optimization System Demo completed")


if __name__ == "__main__":
    asyncio.run(main())
