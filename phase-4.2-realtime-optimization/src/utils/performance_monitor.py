"""
Performance Monitor for Phase 4.2
Real-time performance monitoring and alerting
"""

import asyncio
import logging
import time
import psutil
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import deque
import json


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: float = field(default_factory=time.time)
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    processing_time: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    queue_depth: float = 0.0
    quality_score: float = 0.0


class PerformanceMonitor:
    """Real-time performance monitoring system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_task = None
        
        # Metrics collection
        self.metrics_history = deque(maxlen=1000)
        self.alert_history = deque(maxlen=100)
        
        # Alert thresholds
        self.alert_thresholds = self.config.get('alert_thresholds', {})
        
        # Performance tracking
        self.current_metrics = PerformanceMetrics()
        self.averaged_metrics = PerformanceMetrics()
        
        # System monitoring
        self.process = psutil.Process()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'metrics_collection_interval': 1.0,
            'alert_thresholds': {
                'error_rate': 0.05,
                'processing_time': 5.0,
                'queue_depth': 0.8,
                'memory_usage': 0.9,
                'cpu_usage': 0.8
            },
            'dashboard_update_interval': 5.0,
            'log_retention_days': 7
        }
    
    async def start(self) -> None:
        """Start performance monitoring"""
        if self.is_monitoring:
            self.logger.warning("Performance monitoring is already active")
            return
        
        self.logger.info("Starting performance monitoring")
        self.is_monitoring = True
        
        # Start monitoring task
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        
        self.logger.info("Performance monitoring started")
    
    async def stop(self) -> None:
        """Stop performance monitoring"""
        if not self.is_monitoring:
            return
        
        self.logger.info("Stopping performance monitoring")
        self.is_monitoring = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            await self.monitor_task
        
        self.logger.info("Performance monitoring stopped")
    
    async def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect current metrics
                await self._collect_metrics()
                
                # Check for alerts
                await self._check_alerts()
                
                # Update averaged metrics
                self._update_averaged_metrics()
                
                # Sleep until next collection
                await asyncio.sleep(self.config['metrics_collection_interval'])
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(1.0)
    
    async def _collect_metrics(self) -> None:
        """Collect current performance metrics"""
        timestamp = time.time()
        
        # System metrics
        cpu_usage = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        memory_usage = memory_info.percent
        
        # Process metrics
        process_memory = self.process.memory_info()
        process_cpu = self.process.cpu_percent()
        
        # Create metrics object
        metrics = PerformanceMetrics(
            timestamp=timestamp,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            processing_time=self.current_metrics.processing_time,
            throughput=self.current_metrics.throughput,
            error_rate=self.current_metrics.error_rate,
            queue_depth=self.current_metrics.queue_depth,
            quality_score=self.current_metrics.quality_score
        )
        
        # Update current metrics
        self.current_metrics = metrics
        
        # Add to history
        self.metrics_history.append(metrics)
    
    async def _check_alerts(self) -> None:
        """Check for performance alerts"""
        alerts = []
        
        # Check CPU usage
        if self.current_metrics.cpu_usage > self.alert_thresholds.get('cpu_usage', 0.8):
            alerts.append({
                'type': 'cpu_high',
                'message': f"High CPU usage: {self.current_metrics.cpu_usage:.1%}",
                'severity': 'warning',
                'timestamp': time.time()
            })
        
        # Check memory usage
        if self.current_metrics.memory_usage > self.alert_thresholds.get('memory_usage', 0.9):
            alerts.append({
                'type': 'memory_high',
                'message': f"High memory usage: {self.current_metrics.memory_usage:.1%}",
                'severity': 'critical',
                'timestamp': time.time()
            })
        
        # Check processing time
        if self.current_metrics.processing_time > self.alert_thresholds.get('processing_time', 5.0):
            alerts.append({
                'type': 'processing_slow',
                'message': f"Slow processing: {self.current_metrics.processing_time:.2f}s",
                'severity': 'warning',
                'timestamp': time.time()
            })
        
        # Check error rate
        if self.current_metrics.error_rate > self.alert_thresholds.get('error_rate', 0.05):
            alerts.append({
                'type': 'error_rate_high',
                'message': f"High error rate: {self.current_metrics.error_rate:.1%}",
                'severity': 'critical',
                'timestamp': time.time()
            })
        
        # Check queue depth
        if self.current_metrics.queue_depth > self.alert_thresholds.get('queue_depth', 0.8):
            alerts.append({
                'type': 'queue_backlog',
                'message': f"High queue depth: {self.current_metrics.queue_depth:.1%}",
                'severity': 'warning',
                'timestamp': time.time()
            })
        
        # Log alerts
        for alert in alerts:
            self.logger.warning(f"Performance Alert: {alert['message']}")
            self.alert_history.append(alert)
    
    def _update_averaged_metrics(self) -> None:
        """Update averaged metrics over recent history"""
        if len(self.metrics_history) < 10:
            return
        
        recent_metrics = list(self.metrics_history)[-60:]  # Last 60 seconds
        
        self.averaged_metrics.cpu_usage = np.mean([m.cpu_usage for m in recent_metrics])
        self.averaged_metrics.memory_usage = np.mean([m.memory_usage for m in recent_metrics])
        self.averaged_metrics.processing_time = np.mean([m.processing_time for m in recent_metrics])
        self.averaged_metrics.throughput = np.mean([m.throughput for m in recent_metrics])
        self.averaged_metrics.error_rate = np.mean([m.error_rate for m in recent_metrics])
        self.averaged_metrics.queue_depth = np.mean([m.queue_depth for m in recent_metrics])
        self.averaged_metrics.quality_score = np.mean([m.quality_score for m in recent_metrics])
    
    def update_processing_metrics(self, processing_time: float, throughput: float, 
                                error_rate: float, queue_depth: float, quality_score: float) -> None:
        """Update processing metrics from external sources"""
        self.current_metrics.processing_time = processing_time
        self.current_metrics.throughput = throughput
        self.current_metrics.error_rate = error_rate
        self.current_metrics.queue_depth = queue_depth
        self.current_metrics.quality_score = quality_score
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            'current': {
                'timestamp': self.current_metrics.timestamp,
                'cpu_usage': self.current_metrics.cpu_usage,
                'memory_usage': self.current_metrics.memory_usage,
                'processing_time': self.current_metrics.processing_time,
                'throughput': self.current_metrics.throughput,
                'error_rate': self.current_metrics.error_rate,
                'queue_depth': self.current_metrics.queue_depth,
                'quality_score': self.current_metrics.quality_score
            },
            'averaged': {
                'cpu_usage': self.averaged_metrics.cpu_usage,
                'memory_usage': self.averaged_metrics.memory_usage,
                'processing_time': self.averaged_metrics.processing_time,
                'throughput': self.averaged_metrics.throughput,
                'error_rate': self.averaged_metrics.error_rate,
                'queue_depth': self.averaged_metrics.queue_depth,
                'quality_score': self.averaged_metrics.quality_score
            },
            'system': {
                'process_memory_mb': self.process.memory_info().rss / 1024 / 1024,
                'process_cpu_percent': self.process.cpu_percent(),
                'thread_count': self.process.num_threads(),
                'open_files': self.process.num_fds() if hasattr(self.process, 'num_fds') else 0
            },
            'alerts': {
                'total_alerts': len(self.alert_history),
                'recent_alerts': list(self.alert_history)[-5:]  # Last 5 alerts
            },
            'history': {
                'total_samples': len(self.metrics_history),
                'time_range': {
                    'start': self.metrics_history[0].timestamp if self.metrics_history else None,
                    'end': self.metrics_history[-1].timestamp if self.metrics_history else None
                }
            }
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for dashboard"""
        if len(self.metrics_history) < 2:
            return {'status': 'insufficient_data'}
        
        recent_metrics = list(self.metrics_history)[-300:]  # Last 5 minutes
        
        # Calculate trends
        cpu_trend = self._calculate_trend([m.cpu_usage for m in recent_metrics])
        memory_trend = self._calculate_trend([m.memory_usage for m in recent_metrics])
        processing_trend = self._calculate_trend([m.processing_time for m in recent_metrics])
        
        # Performance status
        status = 'good'
        issues = []
        
        if self.averaged_metrics.error_rate > 0.05:
            status = 'poor'
            issues.append('High error rate')
        elif self.averaged_metrics.processing_time > 3.0:
            status = 'fair'
            issues.append('Slow processing')
        elif self.averaged_metrics.memory_usage > 0.8:
            status = 'fair'
            issues.append('High memory usage')
        
        return {
            'status': status,
            'issues': issues,
            'trends': {
                'cpu': cpu_trend,
                'memory': memory_trend,
                'processing_time': processing_trend
            },
            'key_metrics': {
                'avg_processing_time': self.averaged_metrics.processing_time,
                'avg_throughput': self.averaged_metrics.throughput,
                'avg_quality_score': self.averaged_metrics.quality_score,
                'error_rate': self.averaged_metrics.error_rate
            },
            'alerts_count': len(self.alert_history),
            'uptime': time.time() - (self.metrics_history[0].timestamp if self.metrics_history else time.time())
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend from values"""
        if len(values) < 10:
            return 'stable'
        
        # Compare first half with second half
        mid_point = len(values) // 2
        first_half = np.mean(values[:mid_point])
        second_half = np.mean(values[mid_point:])
        
        change_percent = ((second_half - first_half) / first_half) * 100 if first_half > 0 else 0
        
        if change_percent > 10:
            return 'increasing'
        elif change_percent < -10:
            return 'decreasing'
        else:
            return 'stable'
    
    def export_metrics(self, filename: str, format_type: str = 'json') -> None:
        """Export metrics to file"""
        metrics_data = {
            'export_timestamp': time.time(),
            'current_metrics': self.current_metrics.__dict__,
            'averaged_metrics': self.averaged_metrics.__dict__,
            'metrics_history': [m.__dict__ for m in list(self.metrics_history)],
            'alert_history': list(self.alert_history)
        }
        
        if format_type.lower() == 'json':
            with open(filename, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)
        else:
            # CSV format for metrics
            import csv
            with open(filename, 'w', newline='') as f:
                if self.metrics_history:
                    writer = csv.DictWriter(f, fieldnames=self.metrics_history[0].__dict__.keys())
                    writer.writeheader()
                    for metric in self.metrics_history:
                        writer.writerow(metric.__dict__)
        
        self.logger.info(f"Metrics exported to {filename}")
    
    def reset_metrics(self) -> None:
        """Reset all metrics"""
        self.metrics_history.clear()
        self.alert_history.clear()
        self.current_metrics = PerformanceMetrics()
        self.averaged_metrics = PerformanceMetrics()
        
        self.logger.info("Performance metrics reset")
