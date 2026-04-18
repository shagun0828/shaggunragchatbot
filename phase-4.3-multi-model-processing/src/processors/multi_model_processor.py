"""
Multi-Model Processor for Phase 4.3
Coordinates BGE-base and BGE-small embedders based on URL routing
"""

import asyncio
import logging
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import json

from routers.url_model_router import URLModelRouter, RoutingDecision, ModelType
from embedders.bge_base_embedder import BGEBaseEmbedder
from embedders.bge_small_embedder import BGESmallEmbedder
from models.chunk import Chunk


@dataclass
class MultiModelMetrics:
    """Metrics for multi-model processing"""
    total_urls: int = 0
    base_model_urls: int = 0
    small_model_urls: int = 0
    total_chunks: int = 0
    total_embeddings: int = 0
    total_processing_time: float = 0.0
    avg_quality_score: float = 0.0
    model_efficiency: Dict[str, float] = None
    
    def __post_init__(self):
        if self.model_efficiency is None:
            self.model_efficiency = {}


class MultiModelProcessor:
    """Coordinates multiple BGE models for optimal processing"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.url_router = URLModelRouter(self.config.get('router', {}))
        self.bge_base_embedder = BGEBaseEmbedder(self.config.get('bge_base', {}))
        self.bge_small_embedder = BGESmallEmbedder(self.config.get('bge_small', {}))
        
        # Metrics tracking
        self.metrics = MultiModelMetrics()
        self.processing_history = []
        
        # Model coordination
        self.model_limits = {
            ModelType.BGE_BASE: self.bge_base_embedder.max_urls,
            ModelType.BGE_SMALL: self.bge_small_embedder.max_urls
        }
        
        self.logger.info("Multi-model processor initialized with BGE-base and BGE-small")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'enable_parallel_processing': True,
            'max_concurrent_models': 2,
            'quality_threshold': 0.7,
            'enable_model_comparison': True,
            'optimize_batch_order': True,
            'router': {},
            'bge_base': {},
            'bge_small': {}
        }
    
    async def process_urls(self, url_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process URLs using optimal model selection"""
        start_time = time.time()
        
        self.logger.info(f"Starting multi-model processing for {len(url_data)} URLs")
        
        # Extract URLs for routing
        urls = [item.get('url', '') for item in url_data]
        
        # Route URLs to models
        routing_decisions = self.url_router.route_urls(urls)
        
        # Group URLs by model
        model_groups = self._group_by_model(routing_decisions, url_data)
        
        # Process groups
        processing_results = await self._process_model_groups(model_groups)
        
        # Combine results
        combined_result = self._combine_results(processing_results, routing_decisions)
        
        # Update metrics
        processing_time = time.time() - start_time
        self._update_metrics(len(url_data), processing_time, combined_result)
        
        # Add processing metadata
        combined_result['processing_metadata'] = {
            'total_processing_time': processing_time,
            'routing_decisions': len(routing_decisions),
            'model_groups': {model_type.value: len(groups) for model_type, groups in model_groups.items()},
            'model_utilization': self._calculate_model_utilization(model_groups),
            'efficiency_comparison': self._calculate_efficiency_comparison(processing_results) if self.config['enable_model_comparison'] else None
        }
        
        self.logger.info(f"Multi-model processing completed in {processing_time:.2f}s: "
                        f"{combined_result['bge_base_results']['urls_processed']} URLs with BGE-base, "
                        f"{combined_result['bge_small_results']['urls_processed']} URLs with BGE-small")
        
        return combined_result
    
    def _group_by_model(self, routing_decisions: List[RoutingDecision], 
                        url_data: List[Dict[str, Any]]) -> Dict[ModelType, List[Dict[str, Any]]]:
        """Group URL data by assigned model"""
        model_groups = defaultdict(list)
        
        # Create URL to data mapping
        url_to_data = {item['url']: item for item in url_data}
        
        for decision in routing_decisions:
            if decision.url in url_to_data:
                model_groups[decision.model_type].append(url_to_data[decision.url])
        
        return dict(model_groups)
    
    async def _process_model_groups(self, model_groups: Dict[ModelType, List[Dict[str, Any]]]) -> Dict[ModelType, Dict[str, Any]]:
        """Process each model group"""
        processing_results = {}
        
        if self.config['enable_parallel_processing']:
            # Process models in parallel
            tasks = []
            
            if ModelType.BGE_BASE in model_groups:
                task = asyncio.create_task(
                    self._process_with_bge_base(model_groups[ModelType.BGE_BASE])
                )
                tasks.append(('bge_base', task))
            
            if ModelType.BGE_SMALL in model_groups:
                task = asyncio.create_task(
                    self._process_with_bge_small(model_groups[ModelType.BGE_SMALL])
                )
                tasks.append(('bge_small', task))
            
            # Wait for all tasks
            for model_name, task in tasks:
                try:
                    result = await task
                    processing_results[ModelType.BGE_BASE if model_name == 'bge_base' else ModelType.BGE_SMALL] = result
                except Exception as e:
                    self.logger.error(f"Error processing with {model_name}: {e}")
                    processing_results[ModelType.BGE_BASE if model_name == 'bge_base' else ModelType.BGE_SMALL] = {
                        'status': 'error',
                        'error': str(e),
                        'urls_processed': 0,
                        'chunks_generated': 0
                    }
        else:
            # Process sequentially
            if ModelType.BGE_BASE in model_groups:
                processing_results[ModelType.BGE_BASE] = await self._process_with_bge_base(model_groups[ModelType.BGE_BASE])
            
            if ModelType.BGE_SMALL in model_groups:
                processing_results[ModelType.BGE_SMALL] = await self._process_with_bge_small(model_groups[ModelType.BGE_SMALL])
        
        return processing_results
    
    async def _process_with_bge_base(self, url_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process URLs with BGE-base embedder"""
        self.logger.info(f"Processing {len(url_data)} URLs with BGE-base")
        
        try:
            result = await self.bge_base_embedder.process_urls(url_data)
            result['model_type'] = 'bge_base'
            result['status'] = 'success'
            return result
        except Exception as e:
            self.logger.error(f"BGE-base processing failed: {e}")
            return {
                'model_type': 'bge_base',
                'status': 'error',
                'error': str(e),
                'urls_processed': 0,
                'chunks_generated': 0,
                'embeddings_created': 0
            }
    
    async def _process_with_bge_small(self, url_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process URLs with BGE-small embedder"""
        self.logger.info(f"Processing {len(url_data)} URLs with BGE-small")
        
        try:
            result = await self.bge_small_embedder.process_urls(url_data)
            result['model_type'] = 'bge_small'
            result['status'] = 'success'
            return result
        except Exception as e:
            self.logger.error(f"BGE-small processing failed: {e}")
            return {
                'model_type': 'bge_small',
                'status': 'error',
                'error': str(e),
                'urls_processed': 0,
                'chunks_generated': 0,
                'embeddings_created': 0
            }
    
    def _combine_results(self, processing_results: Dict[ModelType, Dict[str, Any]], 
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
            'efficiency_metrics': {}
        }
        
        # Process each model result
        for model_type, result in processing_results.items():
            model_name = model_type.value
            combined[model_name + '_results'] = result
            
            # Add to totals
            combined['total_urls'] += result.get('urls_processed', 0)
            combined['total_chunks'] += result.get('chunks_generated', 0)
            combined['total_embeddings'] += result.get('embeddings_created', 0)
        
        # Create routing summary
        routing_summary = defaultdict(int)
        for decision in routing_decisions:
            routing_summary[decision.model_type.value] += 1
        combined['routing_summary'] = dict(routing_summary)
        
        # Quality comparison
        if ModelType.BGE_BASE in processing_results and ModelType.BGE_SMALL in processing_results:
            base_quality = processing_results[ModelType.BGE_BASE].get('avg_quality_score', 0)
            small_quality = processing_results[ModelType.BGE_SMALL].get('avg_quality_score', 0)
            
            combined['quality_comparison'] = {
                'bge_base_quality': base_quality,
                'bge_small_quality': small_quality,
                'quality_difference': base_quality - small_quality,
                'better_model': 'bge_base' if base_quality > small_quality else 'bge_small'
            }
        
        # Efficiency metrics
        combined['efficiency_metrics'] = self._calculate_efficiency_metrics(processing_results)
        
        return combined
    
    def _calculate_model_utilization(self, model_groups: Dict[ModelType, List[Dict[str, Any]]]) -> Dict[str, float]:
        """Calculate model utilization rates"""
        utilization = {}
        
        for model_type, urls in model_groups.items():
            max_urls = self.model_limits[model_type]
            utilization[model_type.value] = len(urls) / max_urls if max_urls > 0 else 0
        
        return utilization
    
    def _calculate_efficiency_comparison(self, processing_results: Dict[ModelType, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate efficiency comparison between models"""
        efficiency = {}
        
        if ModelType.BGE_BASE in processing_results and ModelType.BGE_SMALL in processing_results:
            base_result = processing_results[ModelType.BGE_BASE]
            small_result = processing_results[ModelType.BGE_SMALL]
            
            base_throughput = base_result.get('throughput', 0)
            small_throughput = small_result.get('throughput', 0)
            
            base_quality = base_result.get('avg_quality_score', 0)
            small_quality = small_result.get('avg_quality_score', 0)
            
            efficiency = {
                'throughput_ratio': small_throughput / base_throughput if base_throughput > 0 else 0,
                'quality_ratio': base_quality / small_quality if small_quality > 0 else 0,
                'efficiency_score': (small_throughput / (base_throughput + 0.001)) * (base_quality / (small_quality + 0.001))
            }
        
        return efficiency
    
    def _calculate_efficiency_metrics(self, processing_results: Dict[ModelType, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall efficiency metrics"""
        metrics = {
            'total_processing_time': 0,
            'avg_throughput': 0,
            'model_efficiency': {},
            'dimension_efficiency': {}
        }
        
        total_time = 0
        total_chunks = 0
        
        for model_type, result in processing_results.items():
            if result.get('status') == 'success':
                processing_time = result.get('processing_time', 0)
                chunk_count = result.get('chunks_generated', 0)
                throughput = result.get('throughput', 0)
                dimension = result.get('embedding_dimension', 0)
                
                total_time += processing_time
                total_chunks += chunk_count
                
                # Model efficiency
                metrics['model_efficiency'][model_type.value] = {
                    'throughput': throughput,
                    'quality_score': result.get('avg_quality_score', 0),
                    'dimension': dimension,
                    'time_per_chunk': processing_time / chunk_count if chunk_count > 0 else 0,
                    'dimension_throughput_ratio': throughput / dimension if dimension > 0 else 0
                }
        
        metrics['total_processing_time'] = total_time
        metrics['avg_throughput'] = total_chunks / total_time if total_time > 0 else 0
        
        return metrics
    
    def _update_metrics(self, url_count: int, processing_time: float, combined_result: Dict[str, Any]) -> None:
        """Update multi-model metrics"""
        self.metrics.total_urls += url_count
        self.metrics.base_model_urls += combined_result['bge_base_results'].get('urls_processed', 0)
        self.metrics.small_model_urls += combined_result['bge_small_results'].get('urls_processed', 0)
        self.metrics.total_chunks += combined_result['total_chunks']
        self.metrics.total_embeddings += combined_result['total_embeddings']
        self.metrics.total_processing_time += processing_time
        self.metrics.avg_quality_score = combined_result.get('quality_comparison', {}).get('bge_base_quality', 0)
        
        # Update model efficiency
        for model_type, metrics in combined_result.get('efficiency_metrics', {}).get('model_efficiency', {}).items():
            self.metrics.model_efficiency[model_type] = metrics.get('throughput', 0)
        
        # Add to history
        self.processing_history.append({
            'timestamp': time.time(),
            'url_count': url_count,
            'processing_time': processing_time,
            'base_model_urls': combined_result['bge_base_results'].get('urls_processed', 0),
            'small_model_urls': combined_result['bge_small_results'].get('urls_processed', 0),
            'avg_quality': combined_result.get('quality_comparison', {}).get('bge_base_quality', 0)
        })
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics from all components"""
        return {
            'multi_model_metrics': {
                'total_urls': self.metrics.total_urls,
                'base_model_urls': self.metrics.base_model_urls,
                'small_model_urls': self.metrics.small_model_urls,
                'total_chunks': self.metrics.total_chunks,
                'total_embeddings': self.metrics.total_embeddings,
                'total_processing_time': self.metrics.total_processing_time,
                'avg_quality_score': self.metrics.avg_quality_score,
                'model_efficiency': self.metrics.model_efficiency
            },
            'bge_base_metrics': self.bge_base_embedder.get_metrics(),
            'bge_small_metrics': self.bge_small_embedder.get_metrics(),
            'router_metrics': self.url_router.get_routing_statistics(),
            'processing_history_size': len(self.processing_history)
        }
    
    def reset_all_metrics(self) -> None:
        """Reset all metrics"""
        self.metrics = MultiModelMetrics()
        self.processing_history.clear()
        self.bge_base_embedder.reset_metrics()
        self.bge_small_embedder.reset_metrics()
        self.url_router.clear_cache()
        self.logger.info("All multi-model metrics reset")
    
    async def generate_comparison_report(self) -> Dict[str, Any]:
        """Generate detailed comparison report between models"""
        # Get current metrics
        base_metrics = self.bge_base_embedder.get_metrics()['current_metrics']
        small_metrics = self.bge_small_embedder.get_metrics()['current_metrics']
        
        # Calculate comparison metrics
        comparison = {
            'performance_comparison': {
                'bge_base': {
                    'avg_throughput': base_metrics['throughput'],
                    'avg_quality': base_metrics['avg_quality_score'],
                    'dimension': 768,
                    'processing_time': base_metrics['processing_time'],
                    'urls_processed': base_metrics['urls_processed']
                },
                'bge_small': {
                    'avg_throughput': small_metrics['throughput'],
                    'avg_quality': small_metrics['avg_quality_score'],
                    'dimension': 384,
                    'processing_time': small_metrics['processing_time'],
                    'urls_processed': small_metrics['urls_processed']
                }
            },
            'efficiency_analysis': {
                'throughput_ratio': small_metrics['throughput'] / base_metrics['throughput'] if base_metrics['throughput'] > 0 else 0,
                'quality_ratio': base_metrics['avg_quality_score'] / small_metrics['avg_quality_score'] if small_metrics['avg_quality_score'] > 0 else 0,
                'dimension_efficiency': (768 - 384) / 768,  # Storage savings
                'cost_efficiency': 1.0,  # Both are local models
                'speed_advantage': small_metrics['throughput'] > base_metrics['throughput']
            },
            'recommendations': self._generate_recommendations(base_metrics, small_metrics)
        }
        
        return comparison
    
    def _generate_recommendations(self, base_metrics: Dict, small_metrics: Dict) -> List[str]:
        """Generate recommendations based on metrics"""
        recommendations = []
        
        base_throughput = base_metrics['throughput']
        small_throughput = small_metrics['throughput']
        base_quality = base_metrics['avg_quality_score']
        small_quality = small_metrics['avg_quality_score']
        
        # Speed recommendations
        if small_throughput > base_throughput * 1.2:
            recommendations.append("BGE-small shows significantly better throughput - consider using it for time-sensitive applications")
        elif base_throughput > small_throughput * 1.2:
            recommendations.append("BGE-base shows better throughput - consider using it for large batch processing")
        
        # Quality recommendations
        if base_quality > small_quality * 1.1:
            recommendations.append("BGE-base provides significantly better quality - use for critical applications")
        elif small_quality > base_quality * 1.1:
            recommendations.append("BGE-small provides surprisingly good quality - excellent cost-effective option")
        
        # General recommendations
        if base_quality > 0.8 and small_quality > 0.7:
            recommendations.append("Both models show excellent quality - use multi-model approach for optimal results")
        
        return recommendations
