"""
Multi-Model Processor with Chroma Cloud Integration for Phase 4.3
Coordinates BGE-base and BGE-small models with Chroma Cloud storage
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
from storage.chroma_cloud_manager import ChromaCloudManager
from models.chunk import Chunk


@dataclass
class ChromaCloudProcessingMetrics:
    """Metrics for multi-model processing with Chroma Cloud"""
    total_urls: int = 0
    base_model_urls: int = 0
    small_model_urls: int = 0
    total_chunks: int = 0
    total_embeddings: int = 0
    cloud_uploads: int = 0
    total_processing_time: float = 0.0
    avg_quality_score: float = 0.0
    cloud_upload_time: float = 0.0
    model_efficiency: Dict[str, float] = None
    
    def __post_init__(self):
        if self.model_efficiency is None:
            self.model_efficiency = {}


class MultiModelChromaProcessor:
    """Multi-model processor with Chroma Cloud integration"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.url_router = URLModelRouter(self.config.get('router', {}))
        self.bge_base_embedder = BGEBaseEmbedder(self.config.get('bge_base', {}))
        self.bge_small_embedder = BGESmallEmbedder(self.config.get('bge_small', {}))
        self.chroma_cloud_manager = ChromaCloudManager(self.config.get('chroma_cloud', {}))
        
        # Metrics tracking
        self.metrics = ChromaCloudProcessingMetrics()
        self.processing_history = []
        
        # Model coordination
        self.model_limits = {
            ModelType.BGE_BASE: self.bge_base_embedder.max_urls,
            ModelType.BGE_SMALL: self.bge_small_embedder.max_urls
        }
        
        # Chroma Cloud collections
        self.collections = {
            'bge_base': 'mutual_funds',
            'bge_small': 'financial_news'
        }
        
        self.logger.info("Multi-model processor with Chroma Cloud initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'enable_parallel_processing': True,
            'max_concurrent_models': 2,
            'quality_threshold': 0.7,
            'enable_model_comparison': True,
            'enable_cloud_upload': True,
            'cloud_upload_batch_size': 50,
            'router': {},
            'bge_base': {},
            'bge_small': {},
            'chroma_cloud': {
                'api_key': '',
                'tenant': 'default',
                'database': 'mutual-funds-db',
                'host': 'https://api.trychroma.com'
            }
        }
    
    async def process_and_upload_urls(self, url_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process URLs and upload to Chroma Cloud"""
        start_time = time.time()
        
        self.logger.info(f"Processing {len(url_data)} URLs and uploading to Chroma Cloud")
        
        # Extract URLs for routing
        urls = [item.get('url', '') for item in url_data]
        
        # Route URLs to models
        routing_decisions = self.url_router.route_urls(urls)
        
        # Group URLs by model
        model_groups = self._group_by_model(routing_decisions, url_data)
        
        # Process groups
        processing_results = await self._process_model_groups(model_groups)
        
        # Upload to Chroma Cloud
        upload_results = await self._upload_to_chroma_cloud(processing_results)
        
        # Combine results
        combined_result = self._combine_results(processing_results, upload_results, routing_decisions)
        
        # Update metrics
        processing_time = time.time() - start_time
        self._update_metrics(len(url_data), processing_time, combined_result)
        
        # Add processing metadata
        combined_result['processing_metadata'] = {
            'total_processing_time': processing_time,
            'routing_decisions': len(routing_decisions),
            'model_groups': {model_type.value: len(groups) for model_type, groups in model_groups.items()},
            'cloud_uploads': upload_results.get('total_uploads', 0),
            'cloud_upload_time': upload_results.get('total_upload_time', 0.0),
            'model_utilization': self._calculate_model_utilization(model_groups),
            'chroma_cloud_stats': upload_results.get('cloud_stats', {})
        }
        
        self.logger.info(f"Multi-model processing with Chroma Cloud completed in {processing_time:.2f}s: "
                        f"{combined_result['bge_base_results']['urls_processed']} URLs with BGE-base, "
                        f"{combined_result['bge_small_results']['urls_processed']} URLs with BGE-small, "
                        f"{combined_result['cloud_results']['total_uploads']} uploads to Chroma Cloud")
        
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
    
    async def _upload_to_chroma_cloud(self, processing_results: Dict[ModelType, Dict[str, Any]]) -> Dict[str, Any]:
        """Upload processed results to Chroma Cloud"""
        if not self.config['enable_cloud_upload']:
            return {'status': 'skipped', 'reason': 'Cloud upload disabled'}
        
        start_time = time.time()
        total_uploads = 0
        upload_results = {}
        
        for model_type, result in processing_results.items():
            if result.get('status') != 'success':
                continue
            
            try:
                # Extract data for upload
                embeddings = np.array(result['embeddings'])
                documents = [chunk['text'] for chunk in result['chunks']]
                metadata = [chunk['metadata'] for chunk in result['chunks']]
                
                # Determine collection name
                collection_name = self.collections.get(model_type.value, 'default')
                
                # Upload to Chroma Cloud
                collection_id = await self.chroma_cloud_manager.upload_embeddings(
                    embeddings, documents, metadata, collection_name
                )
                
                upload_results[model_type.value] = {
                    'status': 'success',
                    'collection_id': collection_id,
                    'collection_name': collection_name,
                    'documents_uploaded': len(documents),
                    'embeddings_uploaded': len(embeddings)
                }
                
                total_uploads += len(documents)
                
            except Exception as e:
                self.logger.error(f"Failed to upload {model_type.value} results to Chroma Cloud: {e}")
                upload_results[model_type.value] = {
                    'status': 'error',
                    'error': str(e),
                    'documents_uploaded': 0,
                    'embeddings_uploaded': 0
                }
        
        upload_time = time.time() - start_time
        
        # Get cloud statistics
        cloud_stats = {}
        for model_type, result in processing_results.items():
            if result.get('status') == 'success':
                collection_name = self.collections.get(model_type.value, 'default')
                stats = await self.chroma_cloud_manager.get_collection_stats(collection_name)
                cloud_stats[model_type.value] = stats
        
        return {
            'status': 'completed',
            'total_uploads': total_uploads,
            'total_upload_time': upload_time,
            'upload_results': upload_results,
            'cloud_stats': cloud_stats
        }
    
    def _combine_results(self, processing_results: Dict[ModelType, Dict[str, Any]], 
                        upload_results: Dict[str, Any],
                        routing_decisions: List[RoutingDecision]) -> Dict[str, Any]:
        """Combine processing and upload results"""
        combined = {
            'status': 'success',
            'total_urls': 0,
            'total_chunks': 0,
            'total_embeddings': 0,
            'bge_base_results': {},
            'bge_small_results': {},
            'cloud_results': upload_results,
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
        combined['efficiency_metrics'] = self._calculate_efficiency_metrics(processing_results, upload_results)
        
        return combined
    
    def _calculate_model_utilization(self, model_groups: Dict[ModelType, List[Dict[str, Any]]]) -> Dict[str, float]:
        """Calculate model utilization rates"""
        utilization = {}
        
        for model_type, urls in model_groups.items():
            max_urls = self.model_limits[model_type]
            utilization[model_type.value] = len(urls) / max_urls if max_urls > 0 else 0
        
        return utilization
    
    def _calculate_efficiency_metrics(self, processing_results: Dict[ModelType, Dict[str, Any]], 
                                    upload_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall efficiency metrics"""
        metrics = {
            'total_processing_time': 0,
            'total_cloud_upload_time': upload_results.get('total_upload_time', 0),
            'avg_throughput': 0,
            'model_efficiency': {},
            'cloud_efficiency': {},
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
        
        # Cloud efficiency
        if upload_results.get('status') == 'completed':
            total_uploads = upload_results.get('total_uploads', 0)
            upload_time = upload_results.get('total_upload_time', 0)
            
            metrics['cloud_efficiency'] = {
                'upload_throughput': total_uploads / upload_time if upload_time > 0 else 0,
                'upload_success_rate': 1.0,  # Simplified
                'collections_created': len(upload_results.get('upload_results', {}))
            }
        
        return metrics
    
    def _update_metrics(self, url_count: int, processing_time: float, combined_result: Dict[str, Any]) -> None:
        """Update multi-model metrics"""
        self.metrics.total_urls += url_count
        self.metrics.base_model_urls += combined_result['bge_base_results'].get('urls_processed', 0)
        self.metrics.small_model_urls += combined_result['bge_small_results'].get('urls_processed', 0)
        self.metrics.total_chunks += combined_result['total_chunks']
        self.metrics.total_embeddings += combined_result['total_embeddings']
        self.metrics.cloud_uploads += combined_result['cloud_results'].get('total_uploads', 0)
        self.metrics.total_processing_time += processing_time
        self.metrics.cloud_upload_time += combined_result['cloud_results'].get('total_upload_time', 0.0)
        self.metrics.avg_quality_score = combined_result.get('quality_comparison', {}).get('bge_base_quality', 0)
        
        # Update model efficiency
        for model_type, metrics in combined_result.get('efficiency_metrics', {}).get('model_efficiency', {}).items():
            self.metrics.model_efficiency[model_type] = metrics.get('throughput', 0)
        
        # Add to history
        self.processing_history.append({
            'timestamp': time.time(),
            'url_count': url_count,
            'processing_time': processing_time,
            'cloud_uploads': combined_result['cloud_results'].get('total_uploads', 0),
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
                'cloud_uploads': self.metrics.cloud_uploads,
                'total_processing_time': self.metrics.total_processing_time,
                'cloud_upload_time': self.metrics.cloud_upload_time,
                'avg_quality_score': self.metrics.avg_quality_score,
                'model_efficiency': self.metrics.model_efficiency
            },
            'bge_base_metrics': self.bge_base_embedder.get_metrics(),
            'bge_small_metrics': self.bge_small_embedder.get_metrics(),
            'chroma_cloud_metrics': self.chroma_cloud_manager.get_metrics(),
            'router_metrics': self.url_router.get_routing_statistics(),
            'processing_history_size': len(self.processing_history)
        }
    
    def reset_all_metrics(self) -> None:
        """Reset all metrics"""
        self.metrics = ChromaCloudProcessingMetrics()
        self.processing_history.clear()
        self.bge_base_embedder.reset_metrics()
        self.bge_small_embedder.reset_metrics()
        self.chroma_cloud_manager.reset_metrics()
        self.url_router.clear_cache()
        self.logger.info("All multi-model Chroma Cloud metrics reset")
    
    async def search_chroma_cloud(self, query_embedding: np.ndarray, 
                                collection_name: str = 'mutual_funds',
                                top_k: int = 5) -> List[Dict[str, Any]]:
        """Search embeddings in Chroma Cloud"""
        try:
            results = await self.chroma_cloud_manager.search_embeddings(
                query_embedding, collection_name, top_k
            )
            
            self.logger.info(f"Found {len(results)} results in Chroma Cloud collection '{collection_name}'")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to search Chroma Cloud: {e}")
            return []
    
    async def get_cloud_health_status(self) -> Dict[str, Any]:
        """Get health status of Chroma Cloud connection"""
        return await self.chroma_cloud_manager.health_check()
    
    async def generate_cloud_report(self) -> Dict[str, Any]:
        """Generate comprehensive Chroma Cloud report"""
        # Get collection statistics
        collection_stats = {}
        for collection_name in self.collections.values():
            stats = await self.chroma_cloud_manager.get_collection_stats(collection_name)
            collection_stats[collection_name] = stats
        
        return {
            'report_timestamp': time.time(),
            'collection_statistics': collection_stats,
            'cloud_metrics': self.chroma_cloud_manager.get_metrics(),
            'upload_history': list(self.chroma_cloud_manager.upload_history),
            'health_status': await self.get_cloud_health_status(),
            'recommendations': self._generate_cloud_recommendations()
        }
    
    def _generate_cloud_recommendations(self) -> List[str]:
        """Generate recommendations for Chroma Cloud usage"""
        recommendations = []
        
        if self.metrics.cloud_uploads == 0:
            recommendations.append("No data uploaded to Chroma Cloud yet - consider enabling cloud upload")
        
        if self.metrics.cloud_upload_time > self.metrics.total_processing_time * 0.5:
            recommendations.append("Cloud upload time is high - consider increasing batch size")
        
        if self.metrics.total_embeddings > 10000 and self.metrics.collections_created < 3:
            recommendations.append("Consider creating additional collections for better organization")
        
        return recommendations
