#!/usr/bin/env python3
"""
Main entry point for Phase 4.2: Real-time Optimization
Real-time chunking and embedding optimization with streaming capabilities
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from processors.realtime_chunk_processor import RealtimeChunkProcessor, ChunkingRequest, ProcessingMode
from optimizers.dynamic_embedding_optimizer import DynamicEmbeddingOptimizer, OptimizationStrategy
from streaming.streaming_vector_storage import StreamingVectorStorage, StorageMode
from utils.logger import setup_logger, PerformanceLogger
from utils.config_loader import ConfigLoader
from utils.notifications import NotificationManager
from utils.performance_monitor import PerformanceMonitor


class RealtimeOptimizationSystem:
    """Main system for Phase 4.2 real-time optimization"""
    
    def __init__(self, config_path: str = "config/realtime_config.yaml"):
        self.logger = setup_logger("realtime_optimization")
        self.performance_logger = PerformanceLogger("performance")
        self.config = ConfigLoader.load_config(config_path)
        
        # Initialize components
        self.chunk_processor = RealtimeChunkProcessor(self.config.get('chunking', {}))
        self.embedding_optimizer = DynamicEmbeddingOptimizer(self.config.get('embedding', {}))
        self.vector_storage = StreamingVectorStorage(self.config.get('storage', {}))
        self.performance_monitor = PerformanceMonitor(self.config.get('monitoring', {}))
        self.notification_manager = NotificationManager()
        
        # System state
        self.is_running = False
        self.start_time = None
        
        # Performance metrics
        self.system_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_processing_time': 0.0,
            'total_chunks_processed': 0,
            'total_embeddings_generated': 0,
            'system_throughput': 0.0
        }
    
    async def start(self) -> None:
        """Start the real-time optimization system"""
        if self.is_running:
            self.logger.warning("System is already running")
            return
        
        self.logger.info("Starting Phase 4.2 Real-time Optimization System")
        self.start_time = time.time()
        self.is_running = True
        
        try:
            # Start all components
            await self.chunk_processor.start()
            await self.embedding_optimizer.start()
            await self.vector_storage.start_streaming()
            await self.performance_monitor.start()
            
            self.logger.info("All components started successfully")
            
            # Send startup notification
            await self.notification_manager.send_success_notification(
                "Phase 4.2 Real-time Optimization System started successfully"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to start system: {e}")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop the real-time optimization system"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping Phase 4.2 Real-time Optimization System")
        self.is_running = False
        
        try:
            # Stop all components
            await self.chunk_processor.stop()
            await self.embedding_optimizer.stop()
            await self.vector_storage.stop_streaming()
            await self.performance_monitor.stop()
            
            # Calculate final metrics
            uptime = time.time() - self.start_time if self.start_time else 0
            self.system_metrics['uptime'] = uptime
            
            self.logger.info(f"System stopped after {uptime:.2f} seconds")
            
            # Send shutdown notification
            await self.notification_manager.send_success_notification(
                f"Phase 4.2 System stopped. Uptime: {uptime:.2f}s, "
                f"Requests processed: {self.system_metrics['total_requests']}"
            )
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    async def process_text_stream(self, text_stream: List[str], 
                                metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Process a stream of texts through the complete pipeline"""
        self.performance_logger.start_timer("text_stream_processing")
        
        results = []
        
        try:
            for i, text in enumerate(text_stream):
                # Create chunking request
                request = ChunkingRequest(
                    id=f"stream-{int(time.time())}-{i}",
                    text=text,
                    metadata={**(metadata or {}), 'stream_index': i}
                )
                
                # Submit for chunking
                await self.chunk_processor.submit_request(request)
                
                # Get chunking result
                chunk_result = await self.chunk_processor.get_result(timeout=10.0)
                
                if not chunk_result or chunk_result.get('status') != 'success':
                    self.logger.error(f"Chunking failed for text {i}")
                    continue
                
                # Convert to chunks
                chunks = self._dict_to_chunks(chunk_result['chunks'])
                
                if not chunks:
                    continue
                
                # Optimize embeddings
                embeddings, optimization_metadata = await self.embedding_optimizer.optimize_embeddings(chunks)
                
                # Stream to storage
                stream_id = await self.vector_storage.stream_vectors(chunks, embeddings, optimization_metadata)
                
                # Get storage result
                storage_result = await self.vector_storage.get_stream_result(stream_id, timeout=30.0)
                
                # Combine results
                combined_result = {
                    'text_index': i,
                    'chunking_result': chunk_result,
                    'optimization_metadata': optimization_metadata,
                    'stream_id': stream_id,
                    'storage_result': storage_result,
                    'timestamp': time.time()
                }
                
                results.append(combined_result)
                
                # Update metrics
                self._update_system_metrics(combined_result)
        
        except Exception as e:
            self.logger.error(f"Error processing text stream: {e}")
            raise
        
        finally:
            processing_time = self.performance_logger.end_timer("text_stream_processing")
            self.logger.info(f"Processed {len(text_stream)} texts in {processing_time:.2f}s")
        
        return results
    
    async def process_single_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a single text through the complete pipeline"""
        self.performance_logger.start_timer("single_text_processing")
        
        try:
            # Create chunking request
            request = ChunkingRequest(
                id=f"single-{int(time.time())}",
                text=text,
                metadata=metadata or {}
            )
            
            # Submit for chunking
            await self.chunk_processor.submit_request(request)
            
            # Get chunking result
            chunk_result = await self.chunk_processor.get_result(timeout=10.0)
            
            if not chunk_result or chunk_result.get('status') != 'success':
                raise RuntimeError("Chunking failed")
            
            # Convert to chunks
            chunks = self._dict_to_chunks(chunk_result['chunks'])
            
            # Optimize embeddings
            embeddings, optimization_metadata = await self.embedding_optimizer.optimize_embeddings(chunks)
            
            # Stream to storage
            stream_id = await self.vector_storage.stream_vectors(chunks, embeddings, optimization_metadata)
            
            # Get storage result
            storage_result = await self.vector_storage.get_stream_result(stream_id, timeout=30.0)
            
            # Combine results
            result = {
                'chunking_result': chunk_result,
                'optimization_metadata': optimization_metadata,
                'stream_id': stream_id,
                'storage_result': storage_result,
                'timestamp': time.time()
            }
            
            # Update metrics
            self._update_system_metrics(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing single text: {e}")
            raise
        
        finally:
            processing_time = self.performance_logger.end_timer("single_text_processing")
            self.logger.info(f"Processed single text in {processing_time:.2f}s")
    
    def _dict_to_chunks(self, chunk_dicts: List[Dict[str, Any]]) -> List:
        """Convert chunk dictionaries to Chunk objects"""
        from models.chunk import Chunk
        
        chunks = []
        for chunk_dict in chunk_dicts:
            chunk = Chunk(
                text=chunk_dict['text'],
                metadata=chunk_dict.get('metadata', {})
            )
            chunks.append(chunk)
        
        return chunks
    
    def _update_system_metrics(self, result: Dict[str, Any]) -> None:
        """Update system performance metrics"""
        self.system_metrics['total_requests'] += 1
        
        if result.get('storage_result', {}).get('status') == 'completed':
            self.system_metrics['successful_requests'] += 1
        else:
            self.system_metrics['failed_requests'] += 1
        
        # Update processing time
        processing_time = result.get('chunking_result', {}).get('metadata', {}).get('processing_time', 0)
        if processing_time > 0:
            current_avg = self.system_metrics['avg_processing_time']
            total_requests = self.system_metrics['total_requests']
            self.system_metrics['avg_processing_time'] = (
                (current_avg * (total_requests - 1) + processing_time) / total_requests
            )
        
        # Update chunk count
        chunk_count = result.get('chunking_result', {}).get('metadata', {}).get('chunk_count', 0)
        self.system_metrics['total_chunks_processed'] += chunk_count
        
        # Update embedding count
        self.system_metrics['total_embeddings_generated'] += chunk_count
        
        # Calculate throughput
        if self.start_time:
            uptime = time.time() - self.start_time
            self.system_metrics['system_throughput'] = self.system_metrics['total_requests'] / uptime
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        # Get component metrics
        chunking_metrics = self.chunk_processor.get_metrics()
        embedding_metrics = self.embedding_optimizer.get_optimization_metrics()
        storage_metrics = self.vector_storage.get_streaming_metrics()
        performance_metrics = await self.performance_monitor.get_metrics()
        
        return {
            'system_status': {
                'is_running': self.is_running,
                'uptime': time.time() - self.start_time if self.start_time else 0,
                'start_time': self.start_time
            },
            'system_metrics': self.system_metrics,
            'component_metrics': {
                'chunking': chunking_metrics,
                'embedding': embedding_metrics,
                'storage': storage_metrics,
                'performance': performance_metrics
            },
            'config': {
                'chunking_mode': self.chunk_processor.processing_mode.value,
                'embedding_strategy': self.embedding_optimizer.optimization_strategy.value,
                'storage_mode': self.vector_storage.storage_mode.value
            }
        }
    
    async def run_performance_test(self, test_data: List[str]) -> Dict[str, Any]:
        """Run performance test with sample data"""
        self.logger.info(f"Starting performance test with {len(test_data)} samples")
        
        test_start_time = time.time()
        
        try:
            # Process test data
            results = await self.process_text_stream(test_data, {'test_mode': True})
            
            test_end_time = time.time()
            test_duration = test_end_time - test_start_time
            
            # Calculate test metrics
            successful_results = [r for r in results if r.get('storage_result', {}).get('status') == 'completed']
            
            test_metrics = {
                'test_duration': test_duration,
                'total_samples': len(test_data),
                'successful_samples': len(successful_results),
                'success_rate': len(successful_results) / len(test_data),
                'avg_processing_time': np.mean([r.get('chunking_result', {}).get('metadata', {}).get('processing_time', 0) for r in results]),
                'throughput': len(test_data) / test_duration,
                'avg_quality_score': np.mean([r.get('optimization_metadata', {}).get('quality_score', 0) for r in results]),
                'total_chunks': sum([r.get('chunking_result', {}).get('metadata', {}).get('chunk_count', 0) for r in results])
            }
            
            self.logger.info(f"Performance test completed: {test_metrics['success_rate']:.1%} success rate")
            
            return test_metrics
            
        except Exception as e:
            self.logger.error(f"Performance test failed: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform system health check"""
        health_status = {
            'overall_status': 'healthy',
            'components': {},
            'issues': [],
            'timestamp': time.time()
        }
        
        # Check each component
        try:
            chunking_metrics = self.chunk_processor.get_metrics()
            health_status['components']['chunking'] = {
                'status': 'healthy' if chunking_metrics['error_count'] < 5 else 'degraded',
                'metrics': chunking_metrics
            }
        except Exception as e:
            health_status['components']['chunking'] = {'status': 'unhealthy', 'error': str(e)}
            health_status['issues'].append('Chunking component error')
        
        try:
            embedding_metrics = self.embedding_optimizer.get_optimization_metrics()
            health_status['components']['embedding'] = {
                'status': 'healthy' if embedding_metrics['metrics']['quality_score'] > 0.6 else 'degraded',
                'metrics': embedding_metrics
            }
        except Exception as e:
            health_status['components']['embedding'] = {'status': 'unhealthy', 'error': str(e)}
            health_status['issues'].append('Embedding component error')
        
        try:
            storage_metrics = self.vector_storage.get_streaming_metrics()
            health_status['components']['storage'] = {
                'status': 'healthy' if storage_metrics['metrics']['error_count'] < 5 else 'degraded',
                'metrics': storage_metrics
            }
        except Exception as e:
            health_status['components']['storage'] = {'status': 'unhealthy', 'error': str(e)}
            health_status['issues'].append('Storage component error')
        
        # Determine overall status
        if any(comp['status'] == 'unhealthy' for comp in health_status['components'].values()):
            health_status['overall_status'] = 'unhealthy'
        elif any(comp['status'] == 'degraded' for comp in health_status['components'].values()):
            health_status['overall_status'] = 'degraded'
        
        return health_status


async def main():
    """Main entry point"""
    logger = setup_logger("main")
    logger.info("Starting Phase 4.2: Real-time Optimization System")
    
    # Initialize system
    system = RealtimeOptimizationSystem()
    
    try:
        # Start system
        await system.start()
        
        # Sample data for testing
        sample_fund_data = [
            "HDFC Mid-Cap Fund Direct Growth has generated impressive returns of 24.5% in the last year. The current NAV stands at â¹175.43 with AUM of â¹28,432 Cr. The fund manager Rashmi Joshi has successfully navigated market volatility.",
            
            "The fund's top holdings include Reliance Industries (8.5%), TCS (7.2%), and HDFC Bank (6.8%). Sector allocation shows 25% in Financial Services, 20% in Technology, and 15% in Healthcare.",
            
            "Performance metrics show 3-year returns of 18.2% and 5-year returns of 16.8%. The expense ratio is competitive at 1.25% with an exit load of 1% for redemption within 1 year.",
            
            "Risk analysis indicates a Very High risk rating with a beta of 1.2 relative to the benchmark. The fund's standard deviation stands at 18.5%, reflecting the mid-cap segment's inherent volatility.",
            
            "Investment objective focuses on long-term capital appreciation by investing in a diversified portfolio of mid-cap companies. The fund follows a growth-oriented investment strategy with focus on quality businesses."
        ]
        
        logger.info(f"Processing {len(sample_fund_data)} sample texts")
        
        # Run performance test
        test_results = await system.run_performance_test(sample_fund_data)
        
        logger.info("Performance Test Results:")
        for key, value in test_results.items():
            logger.info(f"  {key}: {value}")
        
        # Get system status
        status = await system.get_system_status()
        logger.info(f"System Status: {status['system_status']['is_running']}")
        logger.info(f"Total Requests: {status['system_metrics']['total_requests']}")
        logger.info(f"Success Rate: {status['system_metrics']['successful_requests'] / status['system_metrics']['total_requests']:.1%}")
        
        # Health check
        health = await system.health_check()
        logger.info(f"Health Status: {health['overall_status']}")
        
        # Wait for a bit to show real-time capabilities
        logger.info("System running... (will stop after 30 seconds)")
        await asyncio.sleep(30)
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"System error: {e}")
        raise
    finally:
        # Stop system
        await system.stop()
        logger.info("Phase 4.2 Real-time Optimization System stopped")


if __name__ == "__main__":
    asyncio.run(main())
