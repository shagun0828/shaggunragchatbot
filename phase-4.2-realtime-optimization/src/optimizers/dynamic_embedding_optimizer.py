"""
Dynamic Embedding Optimizer for Phase 4.2
Real-time embedding optimization with adaptive model selection
"""

import asyncio
import logging
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import json

# Import from previous phases
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "phase-4.1-advanced-chunking-embedding" / "src"))

from embedders.enhanced_financial_embedder import EnhancedFinancialEmbedder
from embedders.embedding_quality_checker import EmbeddingQualityChecker, QualityReport
from models.chunk import Chunk


class OptimizationStrategy(Enum):
    """Optimization strategies for embeddings"""
    QUALITY_FOCUSED = "quality_focused"
    SPEED_FOCUSED = "speed_focused"
    BALANCED = "balanced"
    ADAPTIVE = "adaptive"


@dataclass
class EmbeddingMetrics:
    """Metrics for embedding performance"""
    embedding_time: float = 0.0
    quality_score: float = 0.0
    throughput: float = 0.0
    memory_usage: float = 0.0
    model_performance: Dict[str, float] = field(default_factory=dict)
    optimization_count: int = 0
    last_optimization: float = field(default_factory=time.time)


@dataclass
class ModelPerformance:
    """Performance metrics for a specific model"""
    model_name: str
    avg_quality: float = 0.0
    avg_speed: float = 0.0
    usage_count: int = 0
    success_rate: float = 1.0
    last_used: float = field(default_factory=time.time)


class DynamicEmbeddingOptimizer:
    """Dynamic embedding optimizer with real-time model selection"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        
        # Initialize embedders
        self.financial_embedder = EnhancedFinancialEmbedder()
        self.quality_checker = EmbeddingQualityChecker()
        
        # Available models and their performance
        self.available_models = {
            'bge': {
                'bge-small-en-v1.5': {'dimension': 384, 'speed': 'fast', 'quality': 'high'},
                'bge-base-en-v1.5': {'dimension': 768, 'speed': 'medium', 'quality': 'high'},
                'bge-large-en-v1.5': {'dimension': 1024, 'speed': 'slow', 'quality': 'very_high'}
            },
            'sentence_transformers': {
                'all-MiniLM-L6-v2': {'dimension': 384, 'speed': 'fast', 'quality': 'medium'},
                'all-mpnet-base-v2': {'dimension': 768, 'speed': 'medium', 'quality': 'high'},
                'multi-qa-mpnet-base-dot-v1': {'dimension': 768, 'speed': 'medium', 'quality': 'high'}
            },
            'openai': {
                'text-embedding-3-small': {'dimension': 1536, 'speed': 'medium', 'quality': 'high'},
                'text-embedding-3-large': {'dimension': 3072, 'speed': 'slow', 'quality': 'very_high'}
            }
        }
        
        # Model performance tracking
        self.model_performance = {}
        self._initialize_model_performance()
        
        # Optimization state
        self.current_model = 'bge:bge-small-en-v1.5'
        self.optimization_strategy = OptimizationStrategy.ADAPTIVE
        self.metrics = EmbeddingMetrics()
        
        # Performance history
        self.performance_history = deque(maxlen=1000)
        self.quality_history = deque(maxlen=100)
        self.optimization_history = deque(maxlen=50)
        
        # Optimization parameters
        self.optimization_threshold = self.config['optimization_threshold']
        self.optimization_interval = self.config['optimization_interval']
        self.quality_threshold = self.config['quality_threshold']
        
        # Background tasks
        self.optimization_task = None
        self.is_running = False
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'optimization_threshold': 0.1,
            'optimization_interval': 60.0,
            'quality_threshold': 0.7,
            'speed_threshold': 2.0,  # seconds per chunk
            'memory_threshold': 0.8,  # 80% memory usage
            'model_switch_cooldown': 300.0,  # 5 minutes
            'performance_window': 300.0,  # 5 minutes
            'min_samples_for_optimization': 20
        }
    
    def _initialize_model_performance(self) -> None:
        """Initialize model performance tracking"""
        for provider, models in self.available_models.items():
            for model_name, specs in models.items():
                self.model_performance[f"{provider}:{model_name}"] = ModelPerformance(
                    model_name=f"{provider}:{model_name}"
                )
    
    async def start(self) -> None:
        """Start the dynamic optimizer"""
        if self.is_running:
            self.logger.warning("Optimizer is already running")
            return
        
        self.logger.info("Starting dynamic embedding optimizer")
        self.is_running = True
        
        # Start optimization task
        self.optimization_task = asyncio.create_task(self._continuous_optimization())
        
        self.logger.info("Dynamic embedding optimizer started")
    
    async def stop(self) -> None:
        """Stop the dynamic optimizer"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping dynamic embedding optimizer")
        self.is_running = False
        
        if self.optimization_task:
            self.optimization_task.cancel()
            await self.optimization_task
        
        self.logger.info("Dynamic embedding optimizer stopped")
    
    async def optimize_embeddings(self, chunks: List[Chunk], 
                                 strategy: Optional[OptimizationStrategy] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Optimize embeddings with dynamic model selection"""
        start_time = time.time()
        
        # Determine optimization strategy
        if strategy:
            optimization_strategy = strategy
        else:
            optimization_strategy = self._determine_strategy(chunks)
        
        # Select optimal model
        optimal_model = await self._select_optimal_model(chunks, optimization_strategy)
        
        # Generate embeddings with selected model
        embeddings, generation_metrics = await self._generate_embeddings_with_model(
            chunks, optimal_model
        )
        
        # Apply financial enhancement
        enhanced_embeddings = self.financial_embedder.enhance_financial_embeddings(chunks, embeddings)
        
        # Quality assessment
        quality_report = self.quality_checker.check_embedding_quality(enhanced_embeddings, chunks)
        
        # Update performance metrics
        processing_time = time.time() - start_time
        self._update_metrics(optimal_model, processing_time, quality_report, chunks)
        
        # Create optimization metadata
        optimization_metadata = {
            'model_used': optimal_model,
            'strategy': optimization_strategy.value,
            'quality_score': quality_report.overall_score,
            'processing_time': processing_time,
            'chunks_processed': len(chunks),
            'generation_metrics': generation_metrics,
            'quality_report': {
                'overall_score': quality_report.overall_score,
                'issues_count': sum(len(indices) for indices in quality_report.issues.values()),
                'recommendations': quality_report.recommendations[:3]  # Top 3 recommendations
            }
        }
        
        return enhanced_embeddings, optimization_metadata
    
    def _determine_strategy(self, chunks: List[Chunk]) -> OptimizationStrategy:
        """Determine optimal optimization strategy"""
        # Analyze chunk characteristics
        avg_chunk_length = np.mean([len(chunk.text) for chunk in chunks])
        total_chunks = len(chunks)
        
        # Check system load (simplified)
        memory_pressure = self._estimate_memory_pressure()
        
        if memory_pressure > self.config['memory_threshold']:
            return OptimizationStrategy.SPEED_FOCUSED
        elif avg_chunk_length > 800 or total_chunks > 100:
            return OptimizationStrategy.BALANCED
        elif self.metrics.quality_score < self.config['quality_threshold']:
            return OptimizationStrategy.QUALITY_FOCUSED
        else:
            return OptimizationStrategy.ADAPTIVE
    
    async def _select_optimal_model(self, chunks: List[Chunk], 
                                   strategy: OptimizationStrategy) -> str:
        """Select optimal model based on strategy and performance"""
        current_time = time.time()
        
        # Get candidate models
        candidates = self._get_candidate_models(strategy)
        
        # Score each candidate
        model_scores = {}
        for model_key in candidates:
            performance = self.model_performance[model_key]
            
            # Calculate score based on strategy
            if strategy == OptimizationStrategy.QUALITY_FOCUSED:
                score = performance.avg_quality * 0.7 + performance.success_rate * 0.3
            elif strategy == OptimizationStrategy.SPEED_FOCUSED:
                score = (1.0 / max(performance.avg_speed, 0.1)) * 0.7 + performance.success_rate * 0.3
            elif strategy == OptimizationStrategy.BALANCED:
                score = performance.avg_quality * 0.4 + (1.0 / max(performance.avg_speed, 0.1)) * 0.4 + performance.success_rate * 0.2
            else:  # ADAPTIVE
                score = self._calculate_adaptive_score(performance, chunks)
            
            # Apply cooldown penalty
            cooldown_penalty = self._calculate_cooldown_penalty(model_key, current_time)
            score *= (1.0 - cooldown_penalty)
            
            model_scores[model_key] = score
        
        # Select best model
        best_model = max(model_scores, key=model_scores.get)
        
        self.logger.debug(f"Selected model {best_model} with score {model_scores[best_model]:.3f}")
        
        return best_model
    
    def _get_candidate_models(self, strategy: OptimizationStrategy) -> List[str]:
        """Get candidate models based on strategy"""
        all_models = list(self.model_performance.keys())
        
        if strategy == OptimizationStrategy.QUALITY_FOCUSED:
            # Prefer high-quality models
            return [model for model in all_models 
                   if 'high' in self.available_models.get(model.split(':')[0], {}).get(model.split(':')[1], {}).get('quality', '')]
        elif strategy == OptimizationStrategy.SPEED_FOCUSED:
            # Prefer fast models
            return [model for model in all_models 
                   if self.available_models.get(model.split(':')[0], {}).get(model.split(':')[1], {}).get('speed') == 'fast']
        else:
            return all_models
    
    def _calculate_adaptive_score(self, performance: ModelPerformance, chunks: List[Chunk]) -> float:
        """Calculate adaptive score based on current conditions"""
        base_score = performance.avg_quality * 0.4 + (1.0 / max(performance.avg_speed, 0.1)) * 0.4 + performance.success_rate * 0.2
        
        # Adjust based on recent performance
        recent_quality = np.mean(list(self.quality_history)[-10:]) if len(self.quality_history) >= 10 else performance.avg_quality
        
        if recent_quality < self.config['quality_threshold']:
            # Boost quality-focused models
            if 'high' in self.available_models.get(performance.model_name.split(':')[0], {}).get(performance.model_name.split(':')[1], {}).get('quality', ''):
                base_score *= 1.2
        
        return base_score
    
    def _calculate_cooldown_penalty(self, model_key: str, current_time: float) -> float:
        """Calculate cooldown penalty for model switching"""
        performance = self.model_performance[model_key]
        time_since_last_use = current_time - performance.last_used
        
        if time_since_last_use < self.config['model_switch_cooldown']:
            return 0.5  # 50% penalty
        else:
            return 0.0
    
    async def _generate_embeddings_with_model(self, chunks: List[Chunk], model_key: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate embeddings using specific model"""
        provider, model_name = model_key.split(':')
        
        start_time = time.time()
        
        try:
            if provider == 'bge':
                embeddings = await self._generate_bge_embeddings(chunks, model_name)
            elif provider == 'sentence_transformers':
                embeddings = await self._generate_sentence_transformer_embeddings(chunks, model_name)
            elif provider == 'openai':
                embeddings = await self._generate_openai_embeddings(chunks, model_name)
            else:
                raise ValueError(f"Unknown provider: {provider}")
            
            generation_time = time.time() - start_time
            
            # Update model performance
            self._update_model_performance(model_key, generation_time, len(chunks), success=True)
            
            generation_metrics = {
                'provider': provider,
                'model_name': model_name,
                'generation_time': generation_time,
                'chunks_per_second': len(chunks) / generation_time,
                'embedding_dimension': embeddings.shape[1]
            }
            
            return embeddings, generation_metrics
            
        except Exception as e:
            # Update model performance with failure
            self._update_model_performance(model_key, time.time() - start_time, len(chunks), success=False)
            raise e
    
    async def _generate_bge_embeddings(self, chunks: List[Chunk], model_name: str) -> np.ndarray:
        """Generate embeddings using BGE models"""
        texts = [chunk.text for chunk in chunks]
        
        # Update model if different
        if self.financial_embedder.base_model._modules['0'].auto_model.name_or_path != model_name:
            self.financial_embedder.base_model = self.financial_embedder.base_model.__class__(model_name)
        
        embeddings = self.financial_embedder.base_model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        return embeddings
    
    async def _generate_sentence_transformer_embeddings(self, chunks: List[Chunk], model_name: str) -> np.ndarray:
        """Generate embeddings using sentence transformers"""
        texts = [chunk.text for chunk in chunks]
        
        # Update model if different
        if self.financial_embedder.base_model._modules['0'].auto_model.name_or_path != model_name:
            self.financial_embedder.base_model = self.financial_embedder.base_model.__class__(model_name)
        
        embeddings = self.financial_embedder.base_model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        return embeddings
    
    async def _generate_openai_embeddings(self, chunks: List[Chunk], model_name: str) -> np.ndarray:
        """Generate embeddings using OpenAI"""
        # This would use the OpenAI embedder from Phase 4.1
        # For now, simulate with sentence transformers
        texts = [chunk.text for chunk in chunks]
        
        # Simulate OpenAI processing time
        await asyncio.sleep(0.1 * len(chunks))
        
        # Use sentence transformers as fallback
        embeddings = self.financial_embedder.base_model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        # Adjust dimension for OpenAI models
        if model_name == 'text-embedding-3-small':
            # Pad or truncate to 1536 dimensions
            if embeddings.shape[1] < 1536:
                padding = np.zeros((embeddings.shape[0], 1536 - embeddings.shape[1]))
                embeddings = np.hstack([embeddings, padding])
            elif embeddings.shape[1] > 1536:
                embeddings = embeddings[:, :1536]
        elif model_name == 'text-embedding-3-large':
            # Pad or truncate to 3072 dimensions
            if embeddings.shape[1] < 3072:
                padding = np.zeros((embeddings.shape[0], 3072 - embeddings.shape[1]))
                embeddings = np.hstack([embeddings, padding])
            elif embeddings.shape[1] > 3072:
                embeddings = embeddings[:, :3072]
        
        return embeddings
    
    def _update_model_performance(self, model_key: str, processing_time: float, 
                                chunk_count: int, success: bool) -> None:
        """Update model performance metrics"""
        performance = self.model_performance[model_key]
        
        # Update usage count
        performance.usage_count += 1
        performance.last_used = time.time()
        
        if success:
            # Update averages
            chunks_per_second = chunk_count / processing_time
            
            # Exponential moving average
            alpha = 0.1
            performance.avg_speed = alpha * chunks_per_second + (1 - alpha) * performance.avg_speed
            performance.success_rate = alpha * 1.0 + (1 - alpha) * performance.success_rate
        else:
            # Update success rate with failure
            alpha = 0.1
            performance.success_rate = alpha * 0.0 + (1 - alpha) * performance.success_rate
    
    def _update_metrics(self, model_key: str, processing_time: float, 
                        quality_report: QualityReport, chunks: List[Chunk]) -> None:
        """Update optimization metrics"""
        self.metrics.embedding_time += processing_time
        self.metrics.quality_score = quality_report.overall_score
        self.metrics.throughput = len(chunks) / processing_time
        self.metrics.last_update = time.time()
        
        # Add to history
        self.performance_history.append({
            'timestamp': time.time(),
            'model': model_key,
            'processing_time': processing_time,
            'quality_score': quality_report.overall_score,
            'chunk_count': len(chunks)
        })
        
        self.quality_history.append(quality_report.overall_score)
    
    def _estimate_memory_pressure(self) -> float:
        """Estimate current memory pressure (0-1)"""
        # Simplified memory pressure estimation
        # In real implementation, this would use actual memory monitoring
        recent_throughput = [p['throughput'] for p in list(self.performance_history)[-10:]]
        
        if not recent_throughput:
            return 0.5
        
        avg_throughput = np.mean(recent_throughput)
        
        # High throughput indicates memory pressure
        if avg_throughput > 100:  # More than 100 chunks per second
            return 0.8
        elif avg_throughput > 50:
            return 0.6
        else:
            return 0.3
    
    async def _continuous_optimization(self) -> None:
        """Continuous background optimization"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config['optimization_interval'])
                
                # Check if optimization is needed
                if self._should_optimize():
                    await self._perform_optimization()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Continuous optimization error: {e}")
    
    def _should_optimize(self) -> bool:
        """Check if optimization should be performed"""
        if len(self.quality_history) < self.config['min_samples_for_optimization']:
            return False
        
        # Check quality degradation
        recent_quality = list(self.quality_history)[-20:]
        avg_recent_quality = np.mean(recent_quality)
        
        if avg_recent_quality < self.config['quality_threshold']:
            return True
        
        # Check performance degradation
        recent_performance = [p for p in list(self.performance_history)[-20:] 
                            if time.time() - p['timestamp'] < self.config['performance_window']]
        
        if len(recent_performance) > 0:
            avg_processing_time = np.mean([p['processing_time'] for p in recent_performance])
            if avg_processing_time > self.config['speed_threshold']:
                return True
        
        return False
    
    async def _perform_optimization(self) -> None:
        """Perform optimization"""
        self.logger.info("Performing dynamic optimization")
        
        optimization_start = time.time()
        
        # Analyze recent performance
        recent_performance = self._analyze_recent_performance()
        
        # Adjust optimization strategy
        new_strategy = self._select_optimization_strategy(recent_performance)
        if new_strategy != self.optimization_strategy:
            self.optimization_strategy = new_strategy
            self.logger.info(f"Changed optimization strategy to {new_strategy.value}")
        
        # Adjust quality threshold
        await self._adjust_quality_threshold()
        
        # Update optimization count
        self.metrics.optimization_count += 1
        self.metrics.last_optimization = time.time()
        
        # Record optimization
        self.optimization_history.append({
            'timestamp': time.time(),
            'strategy': new_strategy.value,
            'quality_threshold': self.quality_threshold,
            'optimization_time': time.time() - optimization_start,
            'performance_analysis': recent_performance
        })
        
        self.logger.info(f"Optimization completed in {time.time() - optimization_start:.3f}s")
    
    def _analyze_recent_performance(self) -> Dict[str, Any]:
        """Analyze recent performance data"""
        recent_performance = list(self.performance_history)[-50:]
        
        if not recent_performance:
            return {}
        
        return {
            'avg_quality': np.mean([p['quality_score'] for p in recent_performance]),
            'avg_processing_time': np.mean([p['processing_time'] for p in recent_performance]),
            'avg_throughput': np.mean([len(chunks) / p['processing_time'] for p, chunks in 
                                     [(p, [pc for pc in self.performance_history if pc['timestamp'] <= p['timestamp']]) 
                                      for p in recent_performance] if len(chunks) > 0]),
            'model_distribution': self._get_model_distribution(recent_performance),
            'quality_trend': self._calculate_quality_trend()
        }
    
    def _get_model_distribution(self, recent_performance: List[Dict]) -> Dict[str, int]:
        """Get model usage distribution"""
        model_counts = {}
        for p in recent_performance:
            model = p.get('model', 'unknown')
            model_counts[model] = model_counts.get(model, 0) + 1
        return model_counts
    
    def _calculate_quality_trend(self) -> str:
        """Calculate quality trend"""
        if len(self.quality_history) < 10:
            return 'insufficient_data'
        
        recent = list(self.quality_history)[-5:]
        older = list(self.quality_history)[-10:-5]
        
        recent_avg = np.mean(recent)
        older_avg = np.mean(older)
        
        if recent_avg > older_avg + 0.05:
            return 'improving'
        elif recent_avg < older_avg - 0.05:
            return 'declining'
        else:
            return 'stable'
    
    def _select_optimization_strategy(self, performance: Dict[str, Any]) -> OptimizationStrategy:
        """Select optimal optimization strategy based on performance"""
        avg_quality = performance.get('avg_quality', 0.7)
        avg_processing_time = performance.get('avg_processing_time', 1.0)
        
        if avg_quality < 0.6:
            return OptimizationStrategy.QUALITY_FOCUSED
        elif avg_processing_time > 3.0:
            return OptimizationStrategy.SPEED_FOCUSED
        else:
            return OptimizationStrategy.BALANCED
    
    async def _adjust_quality_threshold(self) -> None:
        """Adjust quality threshold based on performance"""
        if len(self.quality_history) < 20:
            return
        
        recent_quality = list(self.quality_history)[-20:]
        avg_quality = np.mean(recent_quality)
        quality_std = np.std(recent_quality)
        
        # Adjust threshold based on quality distribution
        if avg_quality > 0.8 and quality_std < 0.1:
            # High and stable quality, can be more selective
            self.quality_threshold = min(0.9, self.quality_threshold + 0.05)
        elif avg_quality < 0.6:
            # Low quality, be less selective
            self.quality_threshold = max(0.5, self.quality_threshold - 0.05)
        
        self.logger.debug(f"Adjusted quality threshold to {self.quality_threshold:.3f}")
    
    def get_optimization_metrics(self) -> Dict[str, Any]:
        """Get comprehensive optimization metrics"""
        return {
            'current_model': self.current_model,
            'optimization_strategy': self.optimization_strategy.value,
            'quality_threshold': self.quality_threshold,
            'metrics': {
                'embedding_time': self.metrics.embedding_time,
                'quality_score': self.metrics.quality_score,
                'throughput': self.metrics.throughput,
                'optimization_count': self.metrics.optimization_count,
                'last_optimization': self.metrics.last_optimization
            },
            'model_performance': {
                model_key: {
                    'avg_quality': perf.avg_quality,
                    'avg_speed': perf.avg_speed,
                    'usage_count': perf.usage_count,
                    'success_rate': perf.success_rate,
                    'last_used': perf.last_used
                }
                for model_key, perf in self.model_performance.items()
            },
            'recent_performance': self._analyze_recent_performance(),
            'optimization_history': list(self.optimization_history)[-5:]  # Last 5 optimizations
        }
