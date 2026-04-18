"""
Monitoring API Endpoints for Phase 5-6 Application
System metrics, analytics, and health monitoring
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import asyncio
import logging
from datetime import datetime, timedelta

from monitoring.metrics import MetricsCollector
from monitoring.health_checker import HealthChecker
from monitoring.analytics import AnalyticsCollector

router = APIRouter()

# Pydantic models
class SystemMetrics(BaseModel):
    timestamp: float
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    active_connections: int
    request_count: int
    error_count: float

class ComponentHealth(BaseModel):
    component: str
    status: str
    last_check: float
    response_time: float
    error_rate: float
    uptime: float
    details: Dict[str, Any]

class HealthResponse(BaseModel):
    overall_status: str
    timestamp: float
    components: List[ComponentHealth]
    uptime: float
    version: str

class UsageAnalytics(BaseModel):
    date_range: str
    total_queries: int
    unique_users: int
    avg_response_time: float
    top_queries: List[Dict[str, Any]]
    user_activity: List[Dict[str, Any]]
    error_rates: Dict[str, float]

class PerformanceMetrics(BaseModel):
    endpoint: str
    avg_response_time: float
    requests_per_minute: float
    error_rate: float
    p95_response_time: float
    p99_response_time: float

# Dependency injection
async def get_metrics_collector() -> MetricsCollector:
    """Get metrics collector instance"""
    return MetricsCollector()

async def get_health_checker() -> HealthChecker:
    """Get health checker instance"""
    return HealthChecker()

async def get_analytics_collector() -> AnalyticsCollector:
    """Get analytics collector instance"""
    return AnalyticsCollector()

@router.get("/health", response_model=HealthResponse)
async def get_system_health(
    health_checker: HealthChecker = Depends(get_health_checker)
):
    """
    Get comprehensive system health status
    """
    try:
        health_report = await health_checker.check_all_components()
        
        components = []
        for component_name, component_data in health_report["components"].items():
            component = ComponentHealth(
                component=component_name,
                status=component_data.get("status", "unknown"),
                last_check=component_data.get("last_check", 0),
                response_time=component_data.get("response_time", 0),
                error_rate=component_data.get("error_rate", 0),
                uptime=component_data.get("uptime", 0),
                details=component_data.get("details", {})
            )
            components.append(component)
        
        response = HealthResponse(
            overall_status=health_report.get("overall_status", "unknown"),
            timestamp=health_report.get("timestamp", 0),
            components=components,
            uptime=health_report.get("uptime", 0),
            version="2.0.0"
        )
        
        return response
        
    except Exception as e:
        logging.error(f"Error getting system health: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/metrics/system", response_model=SystemMetrics)
async def get_system_metrics(
    metrics_collector: MetricsCollector = Depends(get_metrics_collector)
):
    """
    Get current system metrics
    """
    try:
        metrics = await metrics_collector.get_system_metrics()
        
        return SystemMetrics(
            timestamp=metrics.get("timestamp", 0),
            cpu_usage=metrics.get("cpu_usage", 0),
            memory_usage=metrics.get("memory_usage", 0),
            disk_usage=metrics.get("disk_usage", 0),
            network_io=metrics.get("network_io", {}),
            active_connections=metrics.get("active_connections", 0),
            request_count=metrics.get("request_count", 0),
            error_count=metrics.get("error_count", 0)
        )
        
    except Exception as e:
        logging.error(f"Error getting system metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/metrics/performance")
async def get_performance_metrics(
    endpoint: Optional[str] = Query(None, description="Specific endpoint metrics"),
    time_range: str = Query("1h", description="Time range: 1h, 6h, 24h, 7d"),
    metrics_collector: MetricsCollector = Depends(get_metrics_collector)
):
    """
    Get performance metrics for endpoints
    """
    try:
        metrics = await metrics_collector.get_performance_metrics(endpoint, time_range)
        
        if endpoint:
            # Return specific endpoint metrics
            if endpoint in metrics:
                endpoint_metrics = metrics[endpoint]
                return PerformanceMetrics(
                    endpoint=endpoint,
                    avg_response_time=endpoint_metrics.get("avg_response_time", 0),
                    requests_per_minute=endpoint_metrics.get("requests_per_minute", 0),
                    error_rate=endpoint_metrics.get("error_rate", 0),
                    p95_response_time=endpoint_metrics.get("p95_response_time", 0),
                    p99_response_time=endpoint_metrics.get("p99_response_time", 0)
                )
            else:
                raise HTTPException(status_code=404, detail=f"Endpoint {endpoint} not found")
        else:
            # Return all endpoint metrics
            return {
                "time_range": time_range,
                "endpoints": metrics
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting performance metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/analytics/usage", response_model=UsageAnalytics)
async def get_usage_analytics(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    analytics_collector: AnalyticsCollector = Depends(get_analytics_collector)
):
    """
    Get usage analytics and statistics
    """
    try:
        # Default to last 7 days if no dates provided
        if not start_date:
            start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        analytics = await analytics_collector.get_usage_analytics(start_date, end_date)
        
        return UsageAnalytics(
            date_range=f"{start_date} to {end_date}",
            total_queries=analytics.get("total_queries", 0),
            unique_users=analytics.get("unique_users", 0),
            avg_response_time=analytics.get("avg_response_time", 0),
            top_queries=analytics.get("top_queries", []),
            user_activity=analytics.get("user_activity", []),
            error_rates=analytics.get("error_rates", {})
        )
        
    except Exception as e:
        logging.error(f"Error getting usage analytics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/analytics/queries")
async def get_query_analytics(
    limit: int = Query(100, description="Number of queries to return"),
    sort_by: str = Query("frequency", description="Sort by: frequency, avg_time, error_rate"),
    analytics_collector: AnalyticsCollector = Depends(get_analytics_collector)
):
    """
    Get query analytics and statistics
    """
    try:
        query_analytics = await analytics_collector.get_query_analytics(limit, sort_by)
        
        return {
            "limit": limit,
            "sort_by": sort_by,
            "queries": query_analytics
        }
        
    except Exception as e:
        logging.error(f"Error getting query analytics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/analytics/users")
async def get_user_analytics(
    limit: int = Query(50, description="Number of users to return"),
    sort_by: str = Query("activity", description="Sort by: activity, queries, avg_time"),
    analytics_collector: AnalyticsCollector = Depends(get_analytics_collector)
):
    """
    Get user analytics and statistics
    """
    try:
        user_analytics = await analytics_collector.get_user_analytics(limit, sort_by)
        
        return {
            "limit": limit,
            "sort_by": sort_by,
            "users": user_analytics
        }
        
    except Exception as e:
        logging.error(f"Error getting user analytics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/alerts")
async def get_active_alerts(
    severity: Optional[str] = Query(None, description="Filter by severity: low, medium, high, critical"),
    metrics_collector: MetricsCollector = Depends(get_metrics_collector)
):
    """
    Get active system alerts
    """
    try:
        alerts = await metrics_collector.get_active_alerts(severity)
        
        return {
            "alerts": alerts,
            "total_count": len(alerts),
            "severity_filter": severity
        }
        
    except Exception as e:
        logging.error(f"Error getting alerts: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    metrics_collector: MetricsCollector = Depends(get_metrics_collector)
):
    """
    Acknowledge and resolve an alert
    """
    try:
        success = await metrics_collector.acknowledge_alert(alert_id)
        
        if success:
            return {"status": "success", "message": f"Alert {alert_id} acknowledged"}
        else:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error acknowledging alert: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/logs")
async def get_system_logs(
    level: Optional[str] = Query(None, description="Log level: DEBUG, INFO, WARNING, ERROR"),
    limit: int = Query(100, description="Number of log entries"),
    component: Optional[str] = Query(None, description="Filter by component"),
    metrics_collector: MetricsCollector = Depends(get_metrics_collector)
):
    """
    Get system logs
    """
    try:
        logs = await metrics_collector.get_system_logs(level, limit, component)
        
        return {
            "logs": logs,
            "level_filter": level,
            "component_filter": component,
            "limit": limit
        }
        
    except Exception as e:
        logging.error(f"Error getting system logs: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/dashboard")
async def get_dashboard_data(
    metrics_collector: MetricsCollector = Depends(get_metrics_collector),
    health_checker: HealthChecker = Depends(get_health_checker),
    analytics_collector: AnalyticsCollector = Depends(get_analytics_collector)
):
    """
    Get comprehensive dashboard data
    """
    try:
        # Get all data in parallel
        system_metrics_task = metrics_collector.get_system_metrics()
        health_check_task = health_checker.check_all_components()
        usage_analytics_task = analytics_collector.get_usage_analytics(
            (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
            datetime.now().strftime("%Y-%m-%d")
        )
        alerts_task = metrics_collector.get_active_alerts()
        
        # Wait for all tasks
        system_metrics, health_report, usage_analytics, alerts = await asyncio.gather(
            system_metrics_task,
            health_check_task,
            usage_analytics_task,
            alerts_task,
            return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(system_metrics, Exception):
            system_metrics = {}
        if isinstance(health_report, Exception):
            health_report = {"overall_status": "error", "components": {}}
        if isinstance(usage_analytics, Exception):
            usage_analytics = {}
        if isinstance(alerts, Exception):
            alerts = []
        
        dashboard_data = {
            "timestamp": asyncio.get_event_loop().time(),
            "system_metrics": system_metrics,
            "health_status": health_report.get("overall_status", "unknown"),
            "component_count": len(health_report.get("components", {})),
            "today_queries": usage_analytics.get("total_queries", 0),
            "today_users": usage_analytics.get("unique_users", 0),
            "active_alerts": len(alerts),
            "critical_alerts": len([a for a in alerts if a.get("severity") == "critical"]),
            "avg_response_time": usage_analytics.get("avg_response_time", 0),
            "error_rate": usage_analytics.get("error_rates", {}).get("overall", 0)
        }
        
        return dashboard_data
        
    except Exception as e:
        logging.error(f"Error getting dashboard data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/status/components")
async def get_component_status(
    health_checker: HealthChecker = Depends(get_health_checker)
):
    """
    Get detailed status of all components
    """
    try:
        health_report = await health_checker.check_all_components()
        
        component_status = {}
        for component_name, component_data in health_report.get("components", {}).items():
            component_status[component_name] = {
                "status": component_data.get("status", "unknown"),
                "last_check": component_data.get("last_check", 0),
                "response_time": component_data.get("response_time", 0),
                "error_rate": component_data.get("error_rate", 0),
                "uptime": component_data.get("uptime", 0),
                "details": component_data.get("details", {})
            }
        
        return {
            "timestamp": health_report.get("timestamp", 0),
            "components": component_status,
            "overall_status": health_report.get("overall_status", "unknown")
        }
        
    except Exception as e:
        logging.error(f"Error getting component status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.post("/metrics/reset")
async def reset_metrics(
    component: Optional[str] = Query(None, description="Specific component to reset"),
    metrics_collector: MetricsCollector = Depends(get_metrics_collector)
):
    """
    Reset metrics (admin only)
    """
    try:
        success = await metrics_collector.reset_metrics(component)
        
        if success:
            return {"status": "success", "message": "Metrics reset successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to reset metrics")
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error resetting metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
