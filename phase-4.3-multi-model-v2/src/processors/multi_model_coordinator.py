"""
Multi-Model Coordinator for Phase 4.3
Orchestrates BGE-base and BGE-small embedders with intelligent routing
Coordinates processing, quality management, and performance optimization
"""

import asyncio
import logging
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json

from routers.intelligent_url_router import IntelligentURLRouter, RoutingDecision, ModelType, ContentType
from embedders.bge_base_embedder import BGEBaseEmbedder
from embedders.bge_small_embedder import BGESmallEmbedder, ProcessingMode
from models.chunk import Chunk


class ProcessingStrategy(Enum):
    """Multi-model processing strategies"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


class CoordinationMode(Enum):
    """Coordination modes for different scenarios"""
    QUALITY_FOCUSED = "quality_focused"
    SPEED_FOCUSED = "speed_focused"
    BALANCED = "balanced"
    RESOURCE_OPTIMIZED = "resource_optimized"


@dataclass
class MultiModelMetrics:
    """Comprehensive metrics for multi-model coordination"""
    total_urls: int = 0
    base_model_urls: int = 0
    small_model_urls: int = 0
    total_chunks: int = 0
    total_embeddings: int = 0
    total_processing_time: float = 0.0
    avg_quality_score: float = 0.0
    model_efficiency: Dict[str, float] = field(default_factory=dict)
    routing_efficiency: float = 0.0
    coordination_overhead: float = 0.0
    resource_utilization: Dict[str, float] = field(default_factory=dict)
    processing_strategy: str = "adaptive"
    batch_count: int = 0
    error_count: int = 0
    retry_count: int = 0


@dataclass
class ProcessingResult:
    """Result from multi-model processing"""
    status: str
    model_type: ModelType
    urls_processed: int
    chunks_generated: int
    embeddings_created: int
    processing_time: float
    avg_quality_score: float
    throughput: float
    embedding_dimension: int
    quality_distribution: Dict[str, int]
    metadata: Dict[str, Any]
    errors: List[str] = field(default_factory=list)


class MultiModelCoordinator:
    """Advanced multi-model coordinator with intelligent orchestration"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.url_router = IntelligentURLRouter(self.config.get('router', {}))
        self.bge_base_embedder = BGEBaseEmbedder(self.config.get('bge_base', {}))
        self.bge_small_embedder = BGESmallEmbedder(self.config.get('bge_small', {}))
        
        # Coordination settings
        self.processing_strategy = ProcessingStrategy(self.config.get('processing_strategy', 'adaptive'))
        self.coordination_mode = CoordinationMode(self.config.get('coordination_mode', 'balanced'))
        
        # Metrics tracking
        self.metrics = MultiModelMetrics()
        self.processing_history = deque(maxlen=1000)
        self.performance_history = deque(maxlen=500)
        
        # Resource management
        self.resource_limits = self._initialize_resource_limits()
        self.current_usage = defaultdict(int)
        
        # Quality management
        self.quality_thresholds = self._initialize_quality_thresholds()
        
        # Error handling
        self.error_handlers = self._initialize_error_handlers()
        
        self.logger.info(f"Multi-model coordinator initialized: strategy={self.processing_strategy.value}, mode={self.coordination_mode.value}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'processing_strategy': 'adaptive',
            'coordination_mode': 'balanced',
            'enable_parallel_processing': True,
            'max_concurrent_models': 2,
            'quality_threshold': 0.7,
            'enable_quality_assessment': True,
            'enable_performance_monitoring': True,
            'enable_adaptive_routing': True,
            'enable_load_balancing': True,
            'retry_attempts': 3,
            'retry_delay': 1.0,
            'timeout_per_url': 30.0,
            'router': {},
            'bge_base': {},
            'bge_small': {}
        }
    
    def _initialize_resource_limits(self) -> Dict[str, Any]:
        """Initialize resource limits for coordination"""
        return {
            ModelType.BGE_BASE: {
                'max_urls': 20,
                'max_concurrent_batches': 4,
                'memory_limit_mb': 2048,
                'processing_timeout': 300.0
            },
            ModelType.BGE_SMALL: {
                'max_urls': 5,
                'max_concurrent_batches': 8,
                'memory_limit_mb': 1024,
                'processing_timeout': 180.0
            }
        }
    
    def _initialize_quality_thresholds(self) -> Dict[str, float]:
        """Initialize quality thresholds"""
        return {
            'minimum_acceptable': 0.5,
            'good_quality': 0.7,
            'excellent_quality': 0.85,
            'quality_drop_threshold': 0.2,
            'consistency_threshold': 0.1
        }
    
    def _initialize_error_handlers(self) -> Dict[str, Any]:
        """Initialize error handling strategies"""
        return {
            'network_errors': {
                'retry': True,
                'max_retries': 3,
                'backoff_factor': 2.0
            },
            'memory_errors': {
                'retry': True,
                'batch_size_reduction': 0.5,
                'max_retries': 2
            },
            'quality_errors': {
                'retry': False,
                'fallback_model': True,
                'quality_adjustment': True
            },
            'timeout_errors': {
                'retry': True,
                'timeout_increase': 1.5,
                'max_retries': 2
            }
        }
    
    async def process_urls(self, url_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process URLs with intelligent multi-model coordination"""
        start_time = time.time()
        
        self.logger.info(f"Starting multi-model coordination for {len(url_data)} URLs")
        
        # Extract URLs for routing
        urls = [item.get('url', '') for item in url_data]
        
        # Intelligent routing
        routing_decisions = self.url_router.route_urls(urls)
        
        # Group URLs by model
        model_groups = self._group_urls_by_model(routing_decisions, url_data)
        
        # Determine optimal processing strategy
        optimal_strategy = self._determine_optimal_strategy(model_groups)
        
        # Process with selected strategy
        processing_results = await self._execute_processing_strategy(model_groups, optimal_strategy)
        
        # Combine and analyze results
        combined_result = self._combine_processing_results(processing_results, routing_decisions)
        
        # Update metrics
        processing_time = time.time() - start_time
        self._update_coordinator_metrics(len(url_data), processing_time, combined_result, routing_decisions)
        
        # Add comprehensive metadata
        combined_result['coordination_metadata'] = {
            'processing_strategy': optimal_strategy.value,
            'coordination_mode': self.coordination_mode.value,
            'routing_decisions_count': len(routing_decisions),
            'model_groups': {model_type.value: len(urls) for model_type, urls in model_groups.items()},
            'routing_efficiency': self._calculate_routing_efficiency(routing_decisions),
            'coordination_overhead': self._calculate_coordination_overhead(processing_time, processing_results),
            'resource_utilization': self._calculate_resource_utilization(),
            'quality_consistency': self._assess_quality_consistency(processing_results)
        }
        
        self.logger.info(f"Multi-model coordination completed in {processing_time:.2f}s: "
                        f"{combined_result['bge_base_results']['urls_processed']} URLs with BGE-base, "
                        f"{combined_result['bge_small_results']['urls_processed']} URLs with BGE-small")
        
        return combined_result
    
    def _group_urls_by_model(self, routing_decisions: List[RoutingDecision], 
                             url_data: List[Dict[str, Any]]) -> Dict[ModelType, List[Dict[str, Any]]]:
        """Group URLs by assigned model"""
        model_groups = defaultdict(list)
        
        # Create URL to data mapping
        url_to_data = {item['url']: item for item in url_data}
        
        for decision in routing_decisions:
            if decision.url in url_to_data:
                model_groups[decision.model_type].append({
                    'url': decision.url,
                    'data': url_to_data[decision.url],
                    'routing_decision': decision
                })
        
        return dict(model_groups)
    
    def _determine_optimal_strategy(self, model_groups: Dict[ModelType, List[Dict[str, Any]]]) -> ProcessingStrategy:
        """Determine optimal processing strategy based on workload and resources"""
        
        # Check if adaptive strategy is enabled
        if self.processing_strategy == ProcessingStrategy.ADAPTIVE:
            return self._select_adaptive_strategy(model_groups)
        
        # Use configured strategy
        return self.processing_strategy
    
    def _select_adaptive_strategy(self, model_groups: Dict[ModelType, List[Dict[str, Any]]]) -> ProcessingStrategy:
        """Select adaptive strategy based on current conditions"""
        
        # Count URLs per model
        base_count = len(model_groups.get(ModelType.BGE_BASE, []))
        small_count = len(model_groups.get(ModelType.BGE_SMALL, []))
        total_count = base_count + small_count
        
        # Check resource availability
        base_capacity_available = base_count <= self.resource_limits[ModelType.BGE_BASE]['max_urls']
        small_capacity_available = small_count <= self.resource_limits[ModelType.BGE_SMALL]['max_urls']
        
        # Strategy selection logic
        if total_count <= 5:
            return ProcessingStrategy.SEQUENTIAL
        elif base_capacity_available and small_capacity_available and self.config['enable_parallel_processing']:
            return ProcessingStrategy.PARALLEL
        elif base_count > 10 and small_count > 2:
            return ProcessingStrategy.HYBRID
        else:
            return ProcessingStrategy.ADAPTIVE
    
    async def _execute_processing_strategy(self, model_groups: Dict[ModelType, List[Dict[str, Any]]], 
                                        strategy: ProcessingStrategy) -> Dict[ModelType, ProcessingResult]:
        """Execute processing with selected strategy"""
        
        if strategy == ProcessingStrategy.SEQUENTIAL:
            return await self._execute_sequential_processing(model_groups)
        elif strategy == ProcessingStrategy.PARALLEL:
            return await self._execute_parallel_processing(model_groups)
        elif strategy == ProcessingStrategy.HYBRID:
            return await self._execute_hybrid_processing(model_groups)
        else:  # ADAPTIVE
            return await self._execute_adaptive_processing(model_groups)
    
    async def _execute_sequential_processing(self, model_groups: Dict[ModelType, List[Dict[str, Any]]]) -> Dict[ModelType, ProcessingResult]:
        """Execute sequential processing"""
        self.logger.info("Executing sequential processing strategy")
        
        results = {}
        
        # Process BGE-base first (higher priority)
        if ModelType.BGE_BASE in model_groups:
            base_result = await self._process_model_group(ModelType.BGE_BASE, model_groups[ModelType.BGE_BASE])
            results[ModelType.BGE_BASE] = base_result
        
        # Process BGE-small
        if ModelType.BGE_SMALL in model_groups:
            small_result = await self._process_model_group(ModelType.BGE_SMALL, model_groups[ModelType.BGE_SMALL])
            results[ModelType.BGE_SMALL] = small_result
        
        return results
    
    async def _execute_parallel_processing(self, model_groups: Dict[ModelType, List[Dict[str, Any]]]) -> Dict[ModelType, ProcessingResult]:
        """Execute parallel processing"""
        self.logger.info("Executing parallel processing strategy")
        
        tasks = []
        
        # Create tasks for parallel execution
        if ModelType.BGE_BASE in model_groups:
            task = asyncio.create_task(
                self._process_model_group(ModelType.BGE_BASE, model_groups[ModelType.BGE_BASE])
            )
            tasks.append(('bge_base', task))
        
        if ModelType.BGE_SMALL in model_groups:
            task = asyncio.create_task(
                self._process_model_group(ModelType.BGE_SMALL, model_groups[ModelType.BGE_SMALL])
            )
            tasks.append(('bge_small', task))
        
        # Wait for all tasks
        results = {}
        for model_name, task in tasks:
            try:
                result = await task
                model_type = ModelType.BGE_BASE if model_name == 'bge_base' else ModelType.BGE_SMALL
                results[model_type] = result
            except Exception as e:
                self.logger.error(f"Parallel processing failed for {model_name}: {e}")
                # Create error result
                model_type = ModelType.BGE_BASE if model_name == 'bge_base' else ModelType.BGE_SMALL
                results[model_type] = self._create_error_result(model_type, str(e))
        
        return results
    
    async def _execute_hybrid_processing(self, model_groups: Dict[ModelType, List[Dict[str, Any]]]) -> Dict[ModelType, ProcessingResult]:
        """Execute hybrid processing (optimized for mixed workloads)"""
        self.logger.info("Executing hybrid processing strategy")
        
        results = {}
        
        # Start with parallel processing for smaller groups
        small_groups = {k: v for k, v in model_groups.items() if len(v) <= 3}
        large_groups = {k: v for k, v in model_groups.items() if len(v) > 3}
        
        # Process small groups in parallel
        if small_groups:
            parallel_results = await self._execute_parallel_processing(small_groups)
            results.update(parallel_results)
        
        # Process large groups sequentially
        if large_groups:
            sequential_results = await self._execute_sequential_processing(large_groups)
            results.update(sequential_results)
        
        return results
    
    async def _execute_adaptive_processing(self, model_groups: Dict[ModelType, List[Dict[str, Any]]]) -> Dict[ModelType, ProcessingResult]:
        """Execute adaptive processing with dynamic optimization"""
        self.logger.info("Executing adaptive processing strategy")
        
        # Analyze current system state
        system_state = self._analyze_system_state(model_groups)
        
        # Select optimal approach based on system state
        if system_state['load_level'] == 'low':
            return await self._execute_parallel_processing(model_groups)
        elif system_state['load_level'] == 'medium':
            return await self._execute_hybrid_processing(model_groups)
        else:  # high load
            return await self._execute_sequential_processing(model_groups)
    
    async def _process_model_group(self, model_type: ModelType, url_group: List[Dict[str, Any]]) -> ProcessingResult:
        """Process a group of URLs with specific model"""
        start_time = time.time()
        
        try:
            # Extract URL data
            url_data = [item['data'] for item in url_group]
            
            # Select appropriate embedder
            if model_type == ModelType.BGE_BASE:
                result = await self.bge_base_embedder.process_urls(url_data)
                embedding_dimension = 768
            else:  # BGE_SMALL
                result = await self.bge_small_embedder.process_urls(url_data)
                embedding_dimension = 384
            
            # Create processing result
            processing_result = ProcessingResult(
                status='success',
                model_type=model_type,
                urls_processed=result['urls_processed'],
                chunks_generated=result['chunks_generated'],
                embeddings_created=result['embeddings_created'],
                processing_time=result['processing_time'],
                avg_quality_score=result['avg_quality_score'],
                throughput=result['throughput'],
                embedding_dimension=embedding_dimension,
                quality_distribution=result.get('quality_distribution', {}),
                metadata=result.get('enhancement_metadata', {}),
                errors=[]
            )
            
            # Update resource usage
            self.current_usage[model_type.value] += result['urls_processed']
            
            return processing_result
            
        except Exception as e:
            self.logger.error(f"Processing failed for {model_type.value}: {e}")
            self.metrics.error_count += 1
            
            return self._create_error_result(model_type, str(e))
    
    def _create_error_result(self, model_type: ModelType, error_message: str) -> ProcessingResult:
        """Create error result for failed processing"""
        return ProcessingResult(
            status='error',
            model_type=model_type,
            urls_processed=0,
            chunks_generated=0,
            embeddings_created=0,
            processing_time=0.0,
            avg_quality_score=0.0,
            throughput=0.0,
            embedding_dimension=768 if model_type == ModelType.BGE_BASE else 384,
            quality_distribution={},
            metadata={'error': error_message},
            errors=[error_message]
        )
    
    def _combine_processing_results(self, processing_results: Dict[ModelType, ProcessingResult], 
                                  routing_decisions: List[RoutingDecision]) -> Dict[str, Any]:
        """Combine results from multiple models"""
        
        combined = {
            'status': 'success',
            'total_urls': 0,
            'total_chunks': 0,
            'total_embeddings': 0,
            'bge_base_results': {},
            'bge_small_results': {},
            'routing_summary': {},
            'quality_comparison': {},
            'performance_metrics': {},
            'errors': []
        }
        
        # Process each model result
        for model_type, result in processing_results.items():
            model_name = model_type.value
            combined[model_name + '_results'] = {
                'status': result.status,
                'urls_processed': result.urls_processed,
                'chunks_generated': result.chunks_generated,
                'embeddings_created': result.embeddings_created,
                'processing_time': result.processing_time,
                'avg_quality_score': result.avg_quality_score,
                'throughput': result.throughput,
                'embedding_dimension': result.embedding_dimension,
                'quality_distribution': result.quality_distribution,
                'metadata': result.metadata,
                'errors': result.errors
            }
            
            # Add to totals
            combined['total_urls'] += result.urls_processed
            combined['total_chunks'] += result.chunks_generated
            combined['total_embeddings'] += result.embeddings_created
            
            # Collect errors
            combined['errors'].extend(result.errors)
        
        # Create routing summary
        routing_summary = defaultdict(int)
        for decision in routing_decisions:
            routing_summary[decision.model_type.value] += 1
        combined['routing_summary'] = dict(routing_summary)
        
        # Quality comparison
        if ModelType.BGE_BASE in processing_results and ModelType.BGE_SMALL in processing_results:
            base_quality = processing_results[ModelType.BGE_BASE].avg_quality_score
            small_quality = processing_results[ModelType.BGE_SMALL].avg_quality_score
            
            combined['quality_comparison'] = {
                'bge_base_quality': base_quality,
                'bge_small_quality': small_quality,
                'quality_difference': base_quality - small_quality,
                'better_model': 'bge_base' if base_quality > small_quality else 'bge_small',
                'quality_consistency': abs(base_quality - small_quality) < self.quality_thresholds['consistency_threshold']
            }
        
        # Performance metrics
        combined['performance_metrics'] = self._calculate_performance_metrics(processing_results)
        
        # Overall status
        if combined['errors']:
            combined['status'] = 'partial_success'
        
        return combined
    
    def _analyze_system_state(self, model_groups: Dict[ModelType, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze current system state for adaptive processing"""
        
        total_urls = sum(len(urls) for urls in model_groups.values())
        avg_urls_per_model = total_urls / len(model_groups) if model_groups else 0
        
        # Determine load level
        if total_urls <= 5:
            load_level = 'low'
        elif total_urls <= 15:
            load_level = 'medium'
        else:
            load_level = 'high'
        
        return {
            'total_urls': total_urls,
            'model_count': len(model_groups),
            'avg_urls_per_model': avg_urls_per_model,
            'load_level': load_level,
            'resource_utilization': self._calculate_resource_utilization()
        }
    
    def _calculate_routing_efficiency(self, routing_decisions: List[RoutingDecision]) -> float:
        """Calculate routing efficiency"""
        if not routing_decisions:
            return 0.0
        
        avg_confidence = np.mean([d.confidence for d in routing_decisions])
        optimal_assignments = sum(1 for d in routing_decisions if d.confidence >= 0.8)
        
        efficiency = (avg_confidence * 0.6) + (optimal_assignments / len(routing_decisions) * 0.4)
        
        return efficiency
    
    def _calculate_coordination_overhead(self, total_time: float, 
                                        processing_results: Dict[ModelType, ProcessingResult]) -> float:
        """Calculate coordination overhead"""
        if not processing_results:
            return 0.0
        
        actual_processing_time = sum(result.processing_time for result in processing_results.values())
        overhead = total_time - actual_processing_time
        
        return overhead / total_time if total_time > 0 else 0.0
    
    def _calculate_resource_utilization(self) -> Dict[str, float]:
        """Calculate resource utilization"""
        utilization = {}
        
        for model_type, limits in self.resource_limits.items():
            current = self.current_usage.get(model_type.value, 0)
            max_urls = limits['max_urls']
            utilization[model_type.value] = current / max_urls if max_urls > 0 else 0.0
        
        return utilization
    
    def _assess_quality_consistency(self, processing_results: Dict[ModelType, ProcessingResult]) -> float:
        """Assess quality consistency across models"""
        if len(processing_results) < 2:
            return 1.0
        
        qualities = [result.avg_quality_score for result in processing_results.values()]
        quality_variance = np.var(qualities)
        
        # Lower variance = higher consistency
        consistency = 1.0 - min(quality_variance, 1.0)
        
        return consistency
    
    def _calculate_performance_metrics(self, processing_results: Dict[ModelType, ProcessingResult]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        metrics = {
            'total_processing_time': 0.0,
            'avg_throughput': 0.0,
            'model_efficiency': {},
            'dimension_efficiency': {},
            'quality_efficiency': {}
        }
        
        total_time = 0.0
        total_chunks = 0.0
        
        for model_type, result in processing_results.items():
            if result.status == 'success':
                total_time += result.processing_time
                total_chunks += result.chunks_generated
                
                # Model efficiency
                metrics['model_efficiency'][model_type.value] = {
                    'throughput': result.throughput,
                    'quality_score': result.avg_quality_score,
                    'dimension': result.embedding_dimension,
                    'time_per_chunk': result.processing_time / result.chunks_generated if result.chunks_generated > 0 else 0,
                    'dimension_throughput_ratio': result.throughput / result.embedding_dimension if result.embedding_dimension > 0 else 0
                }
        
        metrics['total_processing_time'] = total_time
        metrics['avg_throughput'] = total_chunks / total_time if total_time > 0 else 0
        
        return metrics
    
    def _update_coordinator_metrics(self, url_count: int, processing_time: float, 
                                  combined_result: Dict[str, Any], routing_decisions: List[RoutingDecision]) -> None:
        """Update coordinator metrics"""
        self.metrics.total_urls += url_count
        self.metrics.base_model_urls += combined_result['bge_base_results'].get('urls_processed', 0)
        self.metrics.small_model_urls += combined_result['bge_small_results'].get('urls_processed', 0)
        self.metrics.total_chunks += combined_result['total_chunks']
        self.metrics.total_embeddings += combined_result['total_embeddings']
        self.metrics.total_processing_time += processing_time
        self.metrics.avg_quality_score = combined_result.get('quality_comparison', {}).get('bge_base_quality', 0)
        self.metrics.routing_efficiency = self._calculate_routing_efficiency(routing_decisions)
        self.metrics.coordination_overhead = self._calculate_coordination_overhead(processing_time, {})
        self.metrics.resource_utilization = self._calculate_resource_utilization()
        self.metrics.processing_strategy = self.processing_strategy.value
        
        # Update model efficiency
        performance_metrics = combined_result.get('performance_metrics', {})
        model_efficiency = performance_metrics.get('model_efficiency', {})
        for model_name, efficiency in model_efficiency.items():
            self.metrics.model_efficiency[model_name] = efficiency.get('throughput', 0)
        
        # Add to history
        self.processing_history.append({
            'timestamp': time.time(),
            'url_count': url_count,
            'processing_time': processing_time,
            'strategy': self.processing_strategy.value,
            'mode': self.coordination_mode.value,
            'avg_quality': self.metrics.avg_quality_score,
            'routing_efficiency': self.metrics.routing_efficiency
        })
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics from all components"""
        return {
            'coordinator_metrics': {
                'total_urls': self.metrics.total_urls,
                'base_model_urls': self.metrics.base_model_urls,
                'small_model_urls': self.metrics.small_model_urls,
                'total_chunks': self.metrics.total_chunks,
                'total_embeddings': self.metrics.total_embeddings,
                'total_processing_time': self.metrics.total_processing_time,
                'avg_quality_score': self.metrics.avg_quality_score,
                'routing_efficiency': self.metrics.routing_efficiency,
                'coordination_overhead': self.metrics.coordination_overhead,
                'processing_strategy': self.metrics.processing_strategy,
                'model_efficiency': self.metrics.model_efficiency,
                'resource_utilization': self.metrics.resource_utilization
            },
            'bge_base_metrics': self.bge_base_embedder.get_metrics(),
            'bge_small_metrics': self.bge_small_embedder.get_metrics(),
            'router_metrics': self.url_router.get_routing_statistics(),
            'processing_history_size': len(self.processing_history),
            'performance_history_size': len(self.performance_history)
        }
    
    def reset_all_metrics(self) -> None:
        """Reset all metrics"""
        self.metrics = MultiModelMetrics()
        self.processing_history.clear()
        self.performance_history.clear()
        self.current_usage.clear()
        self.bge_base_embedder.reset_metrics()
        self.bge_small_embedder.reset_metrics()
        self.url_router.clear_cache()
        self.logger.info("All multi-model coordinator metrics reset")
    
    def set_coordination_mode(self, mode: CoordinationMode) -> None:
        """Set coordination mode"""
        self.coordination_mode = mode
        self.logger.info(f"Coordination mode set to: {mode.value}")
    
    def set_processing_strategy(self, strategy: ProcessingStrategy) -> None:
        """Set processing strategy"""
        self.processing_strategy = strategy
        self.logger.info(f"Processing strategy set to: {strategy.value}")
    
    async def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        # Get current metrics
        coordinator_metrics = self.get_comprehensive_metrics()
        
        # Analyze performance trends
        performance_trends = self._analyze_performance_trends()
        
        # Generate recommendations
        recommendations = self._generate_performance_recommendations(coordinator_metrics)
        
        return {
            'report_timestamp': time.time(),
            'coordinator_metrics': coordinator_metrics['coordinator_metrics'],
            'performance_trends': performance_trends,
            'recommendations': recommendations,
            'system_health': self._assess_system_health(),
            'optimization_opportunities': self._identify_optimization_opportunities()
        }
    
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends from history"""
        if len(self.processing_history) < 5:
            return {'status': 'insufficient_data'}
        
        recent_history = list(self.processing_history)[-10:]
        
        # Calculate trends
        processing_times = [h['processing_time'] for h in recent_history]
        quality_scores = [h['avg_quality'] for h in recent_history]
        routing_efficiencies = [h['routing_efficiency'] for h in recent_history]
        
        return {
            'processing_time_trend': 'improving' if processing_times[-1] < processing_times[0] else 'degrading',
            'quality_trend': 'improving' if quality_scores[-1] > quality_scores[0] else 'degrading',
            'routing_efficiency_trend': 'improving' if routing_efficiencies[-1] > routing_efficiencies[0] else 'degrading',
            'avg_processing_time': np.mean(processing_times),
            'avg_quality_score': np.mean(quality_scores),
            'avg_routing_efficiency': np.mean(routing_efficiencies)
        }
    
    def _generate_performance_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        coordinator_metrics = metrics['coordinator_metrics']
        
        # Processing strategy recommendations
        if coordinator_metrics['coordination_overhead'] > 0.2:
            recommendations.append("High coordination overhead detected - consider optimizing processing strategy")
        
        # Routing efficiency recommendations
        if coordinator_metrics['routing_efficiency'] < 0.7:
            recommendations.append("Low routing efficiency - consider updating URL patterns or model rules")
        
        # Resource utilization recommendations
        for model, utilization in coordinator_metrics['resource_utilization'].items():
            if utilization > 0.9:
                recommendations.append(f"High utilization for {model} - consider scaling resources")
            elif utilization < 0.3:
                recommendations.append(f"Low utilization for {model} - consider adjusting routing rules")
        
        # Quality recommendations
        if coordinator_metrics['avg_quality_score'] < self.quality_thresholds['good_quality']:
            recommendations.append("Quality score below threshold - consider enabling enhancement features")
        
        return recommendations
    
    def _assess_system_health(self) -> Dict[str, Any]:
        """Assess overall system health"""
        health_score = 1.0
        
        # Check routing efficiency
        if self.metrics.routing_efficiency < 0.7:
            health_score -= 0.2
        
        # Check coordination overhead
        if self.metrics.coordination_overhead > 0.3:
            health_score -= 0.2
        
        # Check error rate
        total_processed = self.metrics.total_urls
        if total_processed > 0:
            error_rate = self.metrics.error_count / total_processed
            if error_rate > 0.1:
                health_score -= 0.3
        
        # Check resource utilization
        for utilization in self.metrics.resource_utilization.values():
            if utilization > 0.95:
                health_score -= 0.1
        
        health_status = 'excellent' if health_score >= 0.9 else 'good' if health_score >= 0.7 else 'fair' if health_score >= 0.5 else 'poor'
        
        return {
            'health_score': max(health_score, 0.0),
            'health_status': health_status,
            'issues_identified': self._identify_health_issues()
        }
    
    def _identify_health_issues(self) -> List[str]:
        """Identify specific health issues"""
        issues = []
        
        if self.metrics.routing_efficiency < 0.7:
            issues.append("Low routing efficiency")
        
        if self.metrics.coordination_overhead > 0.3:
            issues.append("High coordination overhead")
        
        if self.metrics.error_count > 0:
            issues.append(f"Processing errors detected: {self.metrics.error_count}")
        
        for model, utilization in self.metrics.resource_utilization.items():
            if utilization > 0.95:
                issues.append(f"High resource utilization for {model}")
        
        return issues
    
    def _identify_optimization_opportunities(self) -> List[str]:
        """Identify optimization opportunities"""
        opportunities = []
        
        # Strategy optimization
        if self.processing_strategy == ProcessingStrategy.SEQUENTIAL and self.metrics.total_urls > 10:
            opportunities.append("Consider switching to parallel or hybrid processing strategy")
        
        # Model optimization
        if self.metrics.avg_quality_score < self.quality_thresholds['excellent_quality']:
            opportunities.append("Consider enabling enhanced processing modes for better quality")
        
        # Resource optimization
        underutilized_models = [model for model, util in self.metrics.resource_utilization.items() if util < 0.5]
        if underutilized_models:
            opportunities.append(f"Underutilized models detected: {', '.join(underutilized_models)}")
        
        return opportunities
