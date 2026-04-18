"""
Analytics Collector for Phase 5-6 Application
Usage analytics and business metrics
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict, Counter
import json


@dataclass
class QueryAnalytics:
    """Query analytics data"""
    query: str
    timestamp: float
    user_id: Optional[str]
    session_id: Optional[str]
    response_time: float
    success: bool
    results_count: int
    query_type: str


@dataclass
class UserAnalytics:
    """User analytics data"""
    user_id: str
    first_seen: float
    last_seen: float
    total_queries: int
    avg_session_length: float
    preferred_query_types: List[str]
    feedback_ratings: List[int]


@dataclass
class UsageAnalytics:
    """Usage analytics summary"""
    date_range: str
    total_queries: int
    unique_users: int
    avg_response_time: float
    top_queries: List[Dict[str, Any]]
    user_activity: List[Dict[str, Any]]
    error_rates: Dict[str, float]


class AnalyticsCollector:
    """Collects and analyzes usage data"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Analytics storage
        self.queries: List[QueryAnalytics] = []
        self.users: Dict[str, UserAnalytics] = {}
        self.daily_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Aggregated data
        self.query_counts = Counter()
        self.user_query_counts = Counter()
        self.error_counts = Counter()
        self.response_times = defaultdict(list)
        
        # Start background processing
        self.processing_task = None
        self.start_processing()
    
    def start_processing(self):
        """Start background analytics processing"""
        if self.processing_task is None:
            self.processing_task = asyncio.create_task(self._process_analytics_loop())
    
    def stop_processing(self):
        """Stop background analytics processing"""
        if self.processing_task:
            self.processing_task.cancel()
            self.processing_task = None
    
    async def _process_analytics_loop(self):
        """Background loop for processing analytics"""
        while True:
            try:
                await self._process_analytics()
                await asyncio.sleep(300)  # Process every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in analytics processing: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    def record_query(
        self,
        query: str,
        user_id: Optional[str],
        session_id: Optional[str],
        response_time: float,
        success: bool,
        results_count: int,
        query_type: str = "search"
    ):
        """Record a query for analytics"""
        timestamp = datetime.now().timestamp()
        
        # Create query analytics
        query_analytics = QueryAnalytics(
            query=query,
            timestamp=timestamp,
            user_id=user_id,
            session_id=session_id,
            response_time=response_time,
            success=success,
            results_count=results_count,
            query_type=query_type
        )
        
        # Store query
        self.queries.append(query_analytics)
        
        # Update counters
        self.query_counts[query] += 1
        if user_id:
            self.user_query_counts[user_id] += 1
        
        if not success:
            self.error_counts[query_type] += 1
        
        self.response_times[query_type].append(response_time)
        
        # Update user analytics
        if user_id:
            self._update_user_analytics(user_id, query_analytics)
        
        # Keep only recent data (last 10000 queries)
        if len(self.queries) > 10000:
            self.queries = self.queries[-10000:]
    
    def _update_user_analytics(self, user_id: str, query_analytics: QueryAnalytics):
        """Update user analytics data"""
        if user_id not in self.users:
            self.users[user_id] = UserAnalytics(
                user_id=user_id,
                first_seen=query_analytics.timestamp,
                last_seen=query_analytics.timestamp,
                total_queries=0,
                avg_session_length=0.0,
                preferred_query_types=[],
                feedback_ratings=[]
            )
        
        user = self.users[user_id]
        user.last_seen = query_analytics.timestamp
        user.total_queries += 1
        
        # Update preferred query types
        if query_analytics.query_type not in user.preferred_query_types:
            user.preferred_query_types.append(query_analytics.query_type)
    
    async def get_usage_analytics(
        self,
        start_date: str,
        end_date: str
    ) -> Dict[str, Any]:
        """Get usage analytics for date range"""
        try:
            # Parse dates
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            start_timestamp = start_dt.timestamp()
            end_timestamp = end_dt.timestamp()
            
            # Filter queries in date range
            filtered_queries = [
                q for q in self.queries
                if start_timestamp <= q.timestamp <= end_timestamp
            ]
            
            if not filtered_queries:
                return {
                    "date_range": f"{start_date} to {end_date}",
                    "total_queries": 0,
                    "unique_users": 0,
                    "avg_response_time": 0.0,
                    "top_queries": [],
                    "user_activity": [],
                    "error_rates": {}
                }
            
            # Calculate metrics
            total_queries = len(filtered_queries)
            unique_users = len(set(q.user_id for q in filtered_queries if q.user_id))
            
            # Average response time
            response_times = [q.response_time for q in filtered_queries]
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0
            
            # Top queries
            query_counter = Counter(q.query for q in filtered_queries)
            top_queries = [
                {"query": query, "count": count}
                for query, count in query_counter.most_common(10)
            ]
            
            # User activity
            user_activity = []
            user_counter = Counter(q.user_id for q in filtered_queries if q.user_id)
            
            for user_id, count in user_counter.most_common(10):
                user_data = self.users.get(user_id)
                if user_data:
                    user_activity.append({
                        "user_id": user_id,
                        "query_count": count,
                        "last_seen": user_data.last_seen,
                        "total_queries": user_data.total_queries
                    })
            
            # Error rates
            error_rates = {}
            query_type_counter = Counter(q.query_type for q in filtered_queries)
            
            for query_type, total in query_type_counter.items():
                errors = len([q for q in filtered_queries if q.query_type == query_type and not q.success])
                error_rates[query_type] = (errors / total * 100) if total > 0 else 0.0
            
            return UsageAnalytics(
                date_range=f"{start_date} to {end_date}",
                total_queries=total_queries,
                unique_users=unique_users,
                avg_response_time=avg_response_time,
                top_queries=top_queries,
                user_activity=user_activity,
                error_rates=error_rates
            ).__dict__
            
        except Exception as e:
            self.logger.error(f"Error getting usage analytics: {e}")
            return {}
    
    async def get_query_analytics(
        self,
        limit: int = 100,
        sort_by: str = "frequency"
    ) -> List[Dict[str, Any]]:
        """Get query analytics"""
        try:
            if sort_by == "frequency":
                # Sort by query frequency
                query_stats = {}
                for query_analytics in self.queries:
                    query = query_analytics.query
                    if query not in query_stats:
                        query_stats[query] = {
                            "query": query,
                            "count": 0,
                            "avg_response_time": 0.0,
                            "success_rate": 0.0,
                            "unique_users": set(),
                            "first_seen": query_analytics.timestamp,
                            "last_seen": query_analytics.timestamp
                        }
                    
                    stats = query_stats[query]
                    stats["count"] += 1
                    stats["avg_response_time"] += query_analytics.response_time
                    
                    if query_analytics.success:
                        stats["success_rate"] += 1
                    
                    if query_analytics.user_id:
                        stats["unique_users"].add(query_analytics.user_id)
                    
                    stats["first_seen"] = min(stats["first_seen"], query_analytics.timestamp)
                    stats["last_seen"] = max(stats["last_seen"], query_analytics.timestamp)
                
                # Finalize stats
                for stats in query_stats.values():
                    if stats["count"] > 0:
                        stats["avg_response_time"] /= stats["count"]
                        stats["success_rate"] = (stats["success_rate"] / stats["count"]) * 100
                        stats["unique_users"] = len(stats["unique_users"])
                    del stats["unique_users"]  # Remove set for JSON serialization
                
                # Sort and limit
                sorted_queries = sorted(
                    query_stats.values(),
                    key=lambda x: x["count"],
                    reverse=True
                )
                
                return sorted_queries[:limit]
            
            elif sort_by == "avg_time":
                # Sort by average response time
                query_times = defaultdict(list)
                for query_analytics in self.queries:
                    query_times[query_analytics.query].append(query_analytics.response_time)
                
                query_avg_times = []
                for query, times in query_times.items():
                    avg_time = sum(times) / len(times)
                    query_avg_times.append({
                        "query": query,
                        "avg_response_time": avg_time,
                        "count": len(times)
                    })
                
                sorted_queries = sorted(
                    query_avg_times,
                    key=lambda x: x["avg_response_time"],
                    reverse=True
                )
                
                return sorted_queries[:limit]
            
            elif sort_by == "error_rate":
                # Sort by error rate
                query_errors = defaultdict(lambda: {"total": 0, "errors": 0})
                for query_analytics in self.queries:
                    query_errors[query_analytics.query]["total"] += 1
                    if not query_analytics.success:
                        query_errors[query_analytics.query]["errors"] += 1
                
                query_error_rates = []
                for query, stats in query_errors.items():
                    error_rate = (stats["errors"] / stats["total"]) * 100 if stats["total"] > 0 else 0
                    query_error_rates.append({
                        "query": query,
                        "error_rate": error_rate,
                        "total_queries": stats["total"],
                        "errors": stats["errors"]
                    })
                
                sorted_queries = sorted(
                    query_error_rates,
                    key=lambda x: x["error_rate"],
                    reverse=True
                )
                
                return sorted_queries[:limit]
            
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting query analytics: {e}")
            return []
    
    async def get_user_analytics(
        self,
        limit: int = 50,
        sort_by: str = "activity"
    ) -> List[Dict[str, Any]]:
        """Get user analytics"""
        try:
            user_list = list(self.users.values())
            
            if sort_by == "activity":
                # Sort by total queries
                sorted_users = sorted(
                    user_list,
                    key=lambda x: x.total_queries,
                    reverse=True
                )
            elif sort_by == "recency":
                # Sort by last seen
                sorted_users = sorted(
                    user_list,
                    key=lambda x: x.last_seen,
                    reverse=True
                )
            elif sort_by == "queries":
                # Sort by average session length
                sorted_users = sorted(
                    user_list,
                    key=lambda x: x.avg_session_length,
                    reverse=True
                )
            else:
                sorted_users = user_list
            
            # Convert to dictionaries
            user_analytics = []
            for user in sorted_users[:limit]:
                user_data = {
                    "user_id": user.user_id,
                    "first_seen": user.first_seen,
                    "last_seen": user.last_seen,
                    "total_queries": user.total_queries,
                    "avg_session_length": user.avg_session_length,
                    "preferred_query_types": user.preferred_query_types,
                    "feedback_ratings": user.feedback_ratings,
                    "days_active": (user.last_seen - user.first_seen) / 86400 if user.first_seen else 0
                }
                user_analytics.append(user_data)
            
            return user_analytics
            
        except Exception as e:
            self.logger.error(f"Error getting user analytics: {e}")
            return []
    
    def record_feedback(
        self,
        user_id: str,
        feedback_data: Dict[str, Any]
    ):
        """Record user feedback"""
        if user_id in self.users:
            user = self.users[user_id]
            rating = feedback_data.get("rating", 0)
            if rating:
                user.feedback_ratings.append(rating)
    
    async def _process_analytics(self):
        """Process and aggregate analytics data"""
        try:
            # Update daily statistics
            today = datetime.now().strftime("%Y-%m-%d")
            
            # Count today's queries
            today_timestamp = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
            today_queries = [
                q for q in self.queries if q.timestamp >= today_timestamp
            ]
            
            self.daily_stats[today] = {
                "total_queries": len(today_queries),
                "unique_users": len(set(q.user_id for q in today_queries if q.user_id)),
                "avg_response_time": sum(q.response_time for q in today_queries) / len(today_queries) if today_queries else 0,
                "success_rate": sum(1 for q in today_queries if q.success) / len(today_queries) * 100 if today_queries else 0
            }
            
            # Clean up old data (keep last 30 days)
            cutoff_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            old_dates = [date for date in self.daily_stats.keys() if date < cutoff_date]
            for date in old_dates:
                del self.daily_stats[date]
            
            self.logger.info(f"Processed analytics for {today}: {len(today_queries)} queries")
            
        except Exception as e:
            self.logger.error(f"Error processing analytics: {e}")
    
    def get_daily_stats(self, date: Optional[str] = None) -> Dict[str, Any]:
        """Get daily statistics"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        return self.daily_stats.get(date, {})
    
    def get_weekly_stats(self) -> Dict[str, Any]:
        """Get weekly statistics"""
        try:
            today = datetime.now()
            week_start = today - timedelta(days=today.weekday())
            
            weekly_queries = []
            for i in range(7):
                date = (week_start + timedelta(days=i)).strftime("%Y-%m-%d")
                daily_stats = self.daily_stats.get(date, {})
                weekly_queries.append(daily_stats.get("total_queries", 0))
            
            return {
                "week_start": week_start.strftime("%Y-%m-%d"),
                "week_end": (week_start + timedelta(days=6)).strftime("%Y-%m-%d"),
                "total_queries": sum(weekly_queries),
                "avg_daily_queries": sum(weekly_queries) / 7,
                "daily_breakdown": weekly_queries
            }
            
        except Exception as e:
            self.logger.error(f"Error getting weekly stats: {e}")
            return {}
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get analytics summary"""
        try:
            total_queries = len(self.queries)
            unique_users = len(self.users)
            
            if total_queries > 0:
                avg_response_time = sum(q.response_time for q in self.queries) / total_queries
                success_rate = sum(1 for q in self.queries if q.success) / total_queries * 100
            else:
                avg_response_time = 0.0
                success_rate = 0.0
            
            return {
                "total_queries": total_queries,
                "unique_users": unique_users,
                "avg_response_time": avg_response_time,
                "success_rate": success_rate,
                "top_query": self.query_counts.most_common(1)[0] if self.query_counts else None,
                "most_active_user": self.user_query_counts.most_common(1)[0] if self.user_query_counts else None,
                "daily_stats_count": len(self.daily_stats),
                "data_retention_days": 30
            }
            
        except Exception as e:
            self.logger.error(f"Error getting analytics summary: {e}")
            return {}
