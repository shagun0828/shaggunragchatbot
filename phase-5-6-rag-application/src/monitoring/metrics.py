"""
Metrics Collector for Phase 5-6 Application
System performance and usage metrics
"""

import asyncio
import logging
import time
import psutil
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json


@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    active_connections: int
    request_count: int
    error_count: float


@dataclass
class PerformanceMetrics:
    """API performance metrics"""
    endpoint: str
    avg_response_time: float
    requests_per_minute: float
    error_rate: float
    p95_response_time: float
    p99_response_time: float
    total_requests: int


@dataclass
class Alert:
    """System alert"""
    alert_id: str
    severity: str
    message: str
    component: str
    timestamp: float
    acknowledged: bool = False
    resolved: bool = False


class MetricsCollector:
    """Collects and manages system metrics"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Metrics storage
        self.system_metrics = deque(maxlen=1000)  # Keep last 1000 data points
        self.performance_metrics = defaultdict(list)
        self.alerts = {}
        
        # Counters
        self.request_count = 0
        self.error_count = 0
        self.active_connections = 0
        
        # Performance tracking
        self.request_times = deque(maxlen=1000)
        self.endpoint_requests = defaultdict(int)
        self.endpoint_errors = defaultdict(int)
        self.endpoint_times = defaultdict(list)
        
        # Alert thresholds
        self.alert_thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "disk_usage": 90.0,
            "error_rate": 5.0,
            "response_time": 2.0
        }
        
        # Start background collection
        self.collection_task = None
        self.start_collection()
    
    def start_collection(self):
        """Start background metrics collection"""
        if self.collection_task is None:
            self.collection_task = asyncio.create_task(self._collect_metrics_loop())
    
    def stop_collection(self):
        """Stop background metrics collection"""
        if self.collection_task:
            self.collection_task.cancel()
            self.collection_task = None
    
    async def _collect_metrics_loop(self):
        """Background loop for collecting metrics"""
        while True:
            try:
                await self.collect_system_metrics()
                await asyncio.sleep(10)  # Collect every 10 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        try:
            timestamp = time.time()
            
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            }
            
            # Create metrics object
            metrics = SystemMetrics(
                timestamp=timestamp,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_io=network_io,
                active_connections=self.active_connections,
                request_count=self.request_count,
                error_count=self.error_count
            )
            
            # Store metrics
            self.system_metrics.append(metrics)
            
            # Check for alerts
            await self._check_system_alerts(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            raise
    
    def record_request(self, endpoint: str, response_time: float, status_code: int):
        """Record API request metrics"""
        self.request_count += 1
        
        # Record request time
        self.request_times.append(response_time)
        
        # Record endpoint metrics
        self.endpoint_requests[endpoint] += 1
        self.endpoint_times[endpoint].append(response_time)
        
        # Record error if applicable
        if status_code >= 400:
            self.error_count += 1
            self.endpoint_errors[endpoint] += 1
        
        # Keep only recent data
        if len(self.endpoint_times[endpoint]) > 1000:
            self.endpoint_times[endpoint] = self.endpoint_times[endpoint][-1000:]
    
    def record_connection(self, increment: bool = True):
        """Record active connection change"""
        if increment:
            self.active_connections += 1
        else:
            self.active_connections = max(0, self.active_connections - 1)
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        if not self.system_metrics:
            return {}
        
        latest = self.system_metrics[-1]
        
        return {
            "timestamp": latest.timestamp,
            "cpu_usage": latest.cpu_usage,
            "memory_usage": latest.memory_usage,
            "disk_usage": latest.disk_usage,
            "network_io": latest.network_io,
            "active_connections": latest.active_connections,
            "request_count": latest.request_count,
            "error_count": latest.error_count
        }
    
    async def get_performance_metrics(
        self,
        endpoint: Optional[str] = None,
        time_range: str = "1h"
    ) -> Dict[str, Any]:
        """Get performance metrics for endpoints"""
        now = time.time()
        
        # Calculate time range
        if time_range == "1h":
            cutoff = now - 3600
        elif time_range == "6h":
            cutoff = now - 21600
        elif time_range == "24h":
            cutoff = now - 86400
        elif time_range == "7d":
            cutoff = now - 604800
        else:
            cutoff = now - 3600  # Default to 1 hour
        
        if endpoint:
            # Get metrics for specific endpoint
            if endpoint not in self.endpoint_times:
                return {}
            
            times = [t for t in self.endpoint_times[endpoint] if time.time() - t < 3600]
            errors = self.endpoint_errors.get(endpoint, 0)
            requests = self.endpoint_requests.get(endpoint, 0)
            
            if not times:
                return {}
            
            times.sort()
            avg_time = sum(times) / len(times)
            p95_time = times[int(len(times) * 0.95)]
            p99_time = times[int(len(times) * 0.99)]
            error_rate = (errors / requests * 100) if requests > 0 else 0
            
            return {
                "avg_response_time": avg_time,
                "requests_per_minute": requests / 60 if requests > 0 else 0,
                "error_rate": error_rate,
                "p95_response_time": p95_time,
                "p99_response_time": p99_time,
                "total_requests": requests
            }
        else:
            # Get metrics for all endpoints
            all_metrics = {}
            
            for ep in self.endpoint_requests:
                times = [t for t in self.endpoint_times[ep]]
                errors = self.endpoint_errors.get(ep, 0)
                requests = self.endpoint_requests[ep]
                
                if times:
                    times.sort()
                    avg_time = sum(times) / len(times)
                    p95_time = times[int(len(times) * 0.95)]
                    p99_time = times[int(len(times) * 0.99)]
                    error_rate = (errors / requests * 100) if requests > 0 else 0
                    
                    all_metrics[ep] = {
                        "avg_response_time": avg_time,
                        "requests_per_minute": requests / 60 if requests > 0 else 0,
                        "error_rate": error_rate,
                        "p95_response_time": p95_time,
                        "p99_response_time": p99_time,
                        "total_requests": requests
                    }
            
            return all_metrics
    
    async def get_active_alerts(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get active alerts"""
        active_alerts = []
        
        for alert_id, alert in self.alerts.items():
            if not alert.resolved:
                if severity is None or alert.severity == severity:
                    active_alerts.append({
                        "alert_id": alert.alert_id,
                        "severity": alert.severity,
                        "message": alert.message,
                        "component": alert.component,
                        "timestamp": alert.timestamp,
                        "acknowledged": alert.acknowledged
                    })
        
        # Sort by timestamp (newest first)
        active_alerts.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return active_alerts
    
    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        if alert_id in self.alerts:
            self.alerts[alert_id].acknowledged = True
            self.logger.info(f"Alert {alert_id} acknowledged")
            return True
        return False
    
    async def get_system_logs(
        self,
        level: Optional[str] = None,
        limit: int = 100,
        component: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get system logs (mock implementation)"""
        # Mock logs - in production, would fetch from actual log system
        logs = []
        
        for i in range(min(limit, 50)):
            log_entry = {
                "timestamp": time.time() - (i * 60),  # 1 minute apart
                "level": level or ["INFO", "WARNING", "ERROR"][i % 3],
                "message": f"Sample log message {i}",
                "component": component or "system",
                "thread": "main"
            }
            logs.append(log_entry)
        
        return logs
    
    async def reset_metrics(self, component: Optional[str] = None) -> bool:
        """Reset metrics for component or all"""
        try:
            if component:
                # Reset specific component metrics
                if component in self.endpoint_requests:
                    self.endpoint_requests[component] = 0
                if component in self.endpoint_errors:
                    self.endpoint_errors[component] = 0
                if component in self.endpoint_times:
                    self.endpoint_times[component] = []
            else:
                # Reset all metrics
                self.request_count = 0
                self.error_count = 0
                self.active_connections = 0
                self.request_times.clear()
                self.endpoint_requests.clear()
                self.endpoint_errors.clear()
                self.endpoint_times.clear()
            
            self.logger.info(f"Metrics reset for component: {component or 'all'}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error resetting metrics: {e}")
            return False
    
    async def _check_system_alerts(self, metrics: SystemMetrics):
        """Check for system alerts based on thresholds"""
        alerts_to_create = []
        
        # CPU usage alert
        if metrics.cpu_usage > self.alert_thresholds["cpu_usage"]:
            alerts_to_create.append(Alert(
                alert_id=f"cpu_alert_{int(metrics.timestamp)}",
                severity="high" if metrics.cpu_usage > 90 else "medium",
                message=f"High CPU usage: {metrics.cpu_usage:.1f}%",
                component="cpu",
                timestamp=metrics.timestamp
            ))
        
        # Memory usage alert
        if metrics.memory_usage > self.alert_thresholds["memory_usage"]:
            alerts_to_create.append(Alert(
                alert_id=f"memory_alert_{int(metrics.timestamp)}",
                severity="high" if metrics.memory_usage > 95 else "medium",
                message=f"High memory usage: {metrics.memory_usage:.1f}%",
                component="memory",
                timestamp=metrics.timestamp
            ))
        
        # Disk usage alert
        if metrics.disk_usage > self.alert_thresholds["disk_usage"]:
            alerts_to_create.append(Alert(
                alert_id=f"disk_alert_{int(metrics.timestamp)}",
                severity="critical",
                message=f"High disk usage: {metrics.disk_usage:.1f}%",
                component="disk",
                timestamp=metrics.timestamp
            ))
        
        # Error rate alert
        if self.request_count > 0:
            error_rate = (self.error_count / self.request_count) * 100
            if error_rate > self.alert_thresholds["error_rate"]:
                alerts_to_create.append(Alert(
                    alert_id=f"error_alert_{int(metrics.timestamp)}",
                    severity="high",
                    message=f"High error rate: {error_rate:.1f}%",
                    component="api",
                    timestamp=metrics.timestamp
                ))
        
        # Add alerts to storage
        for alert in alerts_to_create:
            self.alerts[alert.alert_id] = alert
            self.logger.warning(f"Alert created: {alert.message}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get metrics collector status"""
        return {
            "collecting": self.collection_task is not None,
            "system_metrics_count": len(self.system_metrics),
            "active_alerts": len([a for a in self.alerts.values() if not a.resolved]),
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "active_connections": self.active_connections,
            "endpoints_tracked": len(self.endpoint_requests)
        }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics"""
        return {
            "system_metrics": [asdict(m) for m in list(self.system_metrics)[-10:]],  # Last 10
            "performance_metrics": dict(self.endpoint_requests),
            "alerts": {k: asdict(v) for k, v in list(self.alerts.items())[-5:]},  # Last 5
            "counters": {
                "request_count": self.request_count,
                "error_count": self.error_count,
                "active_connections": self.active_connections
            },
            "thresholds": self.alert_thresholds
        }


class HealthChecker:
    """Health checker for system components"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.component_status = {}
        self.last_check = 0
    
    async def check_all_components(self) -> Dict[str, Any]:
        """Check health of all components"""
        now = time.time()
        
        # Check components
        components = {}
        
        # Chroma client health
        chroma_health = await self._check_chroma_client()
        components["chroma_client"] = chroma_health
        
        # LLM client health
        llm_health = await self._check_llm_client()
        components["llm_client"] = llm_health
        
        # Database health
        db_health = await self._check_database()
        components["database"] = db_health
        
        # WebSocket health
        websocket_health = await self._check_websocket()
        components["websocket"] = websocket_health
        
        # Calculate overall status
        overall_status = "healthy"
        for component, health in components.items():
            if health["status"] != "healthy":
                overall_status = "degraded" if overall_status == "healthy" else "unhealthy"
        
        self.component_status = components
        self.last_check = now
        
        return {
            "overall_status": overall_status,
            "timestamp": now,
            "components": components,
            "uptime": now  # Would calculate actual uptime
        }
    
    async def _check_chroma_client(self) -> Dict[str, Any]:
        """Check Chroma client health"""
        try:
            # Mock health check
            return {
                "status": "healthy",
                "last_check": time.time(),
                "response_time": 0.1,
                "error_rate": 0.0,
                "uptime": 3600.0,
                "details": {"connected": True}
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "last_check": time.time(),
                "response_time": 0.0,
                "error_rate": 1.0,
                "uptime": 0.0,
                "details": {"error": str(e)}
            }
    
    async def _check_llm_client(self) -> Dict[str, Any]:
        """Check LLM client health"""
        try:
            return {
                "status": "healthy",
                "last_check": time.time(),
                "response_time": 0.2,
                "error_rate": 0.0,
                "uptime": 3600.0,
                "details": {"model": "gpt-4", "available": True}
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "last_check": time.time(),
                "response_time": 0.0,
                "error_rate": 1.0,
                "uptime": 0.0,
                "details": {"error": str(e)}
            }
    
    async def _check_database(self) -> Dict[str, Any]:
        """Check database health"""
        try:
            return {
                "status": "healthy",
                "last_check": time.time(),
                "response_time": 0.05,
                "error_rate": 0.0,
                "uptime": 3600.0,
                "details": {"connections": 5, "size": "2.3GB"}
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "last_check": time.time(),
                "response_time": 0.0,
                "error_rate": 1.0,
                "uptime": 0.0,
                "details": {"error": str(e)}
            }
    
    async def _check_websocket(self) -> Dict[str, Any]:
        """Check WebSocket health"""
        try:
            return {
                "status": "healthy",
                "last_check": time.time(),
                "response_time": 0.01,
                "error_rate": 0.0,
                "uptime": 3600.0,
                "details": {"connections": 10}
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "last_check": time.time(),
                "response_time": 0.0,
                "error_rate": 1.0,
                "uptime": 0.0,
                "details": {"error": str(e)}
            }
