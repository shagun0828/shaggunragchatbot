"""
User Profiler for Phase 5-6 Application
Manages user profiles and personalization features
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class UserProfile:
    """User profile data structure"""
    user_id: str
    created_at: float
    last_activity: float
    preferences: Dict[str, Any]
    query_history: List[Dict[str, Any]]
    interaction_patterns: Dict[str, Any]
    feedback_history: List[Dict[str, Any]]
    risk_tolerance: str
    investment_horizon: str
    preferred_fund_types: List[str]
    expertise_level: str
    session_count: int
    total_queries: int


@dataclass
class UserInteraction:
    """User interaction data"""
    user_id: str
    interaction_type: str  # query, feedback, rating, click
    content: str
    timestamp: float
    metadata: Dict[str, Any]
    context: Dict[str, Any]


class UserProfiler:
    """Advanced user profiler for personalization"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.profiles: Dict[str, UserProfile] = {}
        self.interactions: Dict[str, List[UserInteraction]] = defaultdict(list)
        
        # Personalization weights
        self.personalization_weights = {
            "query_history": 0.3,
            "feedback": 0.25,
            "interaction_patterns": 0.2,
            "explicit_preferences": 0.15,
            "demographic_factors": 0.1
        }
    
    async def get_profile(self, user_id: str) -> Optional[UserProfile]:
        """
        Get user profile
        """
        try:
            if user_id not in self.profiles:
                # Create new profile
                await self._create_profile(user_id)
            
            profile = self.profiles[user_id]
            
            # Update last activity
            profile.last_activity = asyncio.get_event_loop().time()
            
            return profile
            
        except Exception as e:
            self.logger.error(f"Error getting user profile: {e}")
            return None
    
    async def update_profile(
        self,
        user_id: str,
        query: str,
        retrieved_docs: List[Dict[str, Any]]
    ) -> bool:
        """
        Update user profile based on interaction
        """
        try:
            profile = await self.get_profile(user_id)
            if not profile:
                return False
            
            # Update query history
            profile.query_history.append({
                "query": query,
                "timestamp": asyncio.get_event_loop().time(),
                "doc_count": len(retrieved_docs),
                "doc_types": [doc.get("metadata", {}).get("type", "unknown") for doc in retrieved_docs]
            })
            
            # Limit query history
            if len(profile.query_history) > 100:
                profile.query_history = profile.query_history[-100:]
            
            # Update interaction patterns
            await self._update_interaction_patterns(profile, query, retrieved_docs)
            
            # Update preferences based on behavior
            await self._infer_preferences(profile, query, retrieved_docs)
            
            # Update counters
            profile.total_queries += 1
            
            self.logger.info(f"Updated profile for user {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating user profile: {e}")
            return False
    
    async def update_chat_activity(
        self,
        user_id: str,
        query: str,
        response: str,
        context_used: bool
    ) -> bool:
        """
        Update profile based on chat activity
        """
        try:
            profile = await self.get_profile(user_id)
            if not profile:
                return False
            
            # Record interaction
            interaction = UserInteraction(
                user_id=user_id,
                interaction_type="chat",
                content=query,
                timestamp=asyncio.get_event_loop().time(),
                metadata={
                    "response": response,
                    "context_used": context_used,
                    "response_length": len(response)
                },
                context={}
            )
            
            self.interactions[user_id].append(interaction)
            
            # Update chat-specific patterns
            if "chat_patterns" not in profile.interaction_patterns:
                profile.interaction_patterns["chat_patterns"] = {}
            
            chat_patterns = profile.interaction_patterns["chat_patterns"]
            chat_patterns["total_chats"] = chat_patterns.get("total_chats", 0) + 1
            chat_patterns["context_usage_rate"] = (
                chat_patterns.get("context_usage_count", 0) + (1 if context_used else 0)
            ) / chat_patterns["total_chats"]
            
            # Update expertise based on query complexity
            complexity = await self._analyze_query_complexity(query)
            profile.expertise_level = await self._update_expertise_level(profile.expertise_level, complexity)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating chat activity: {e}")
            return False
    
    async def add_feedback(
        self,
        user_id: str,
        feedback_data: Dict[str, Any]
    ) -> bool:
        """
        Add user feedback to profile
        """
        try:
            profile = await self.get_profile(user_id)
            if not profile:
                return False
            
            # Add feedback to history
            profile.feedback_history.append({
                **feedback_data,
                "timestamp": asyncio.get_event_loop().time()
            })
            
            # Limit feedback history
            if len(profile.feedback_history) > 50:
                profile.feedback_history = profile.feedback_history[-50:]
            
            # Update preferences based on feedback
            await self._process_feedback(profile, feedback_data)
            
            # Record interaction
            interaction = UserInteraction(
                user_id=user_id,
                interaction_type="feedback",
                content=feedback_data.get("query_id", ""),
                timestamp=asyncio.get_event_loop().time(),
                metadata=feedback_data,
                context={}
            )
            
            self.interactions[user_id].append(interaction)
            
            self.logger.info(f"Added feedback for user {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding feedback: {e}")
            return False
    
    async def personalize_context(
        self,
        context: List[str],
        user_profile: UserProfile
    ) -> List[str]:
        """
        Personalize context based on user profile
        """
        try:
            personalized_context = []
            
            # Reorder context based on user preferences
            if user_profile.preferred_fund_types:
                # Prioritize content about preferred fund types
                preferred_context = []
                other_context = []
                
                for content in context:
                    content_lower = content.lower()
                    if any(fund_type in content_lower for fund_type in user_profile.preferred_fund_types):
                        preferred_context.append(content)
                    else:
                        other_context.append(content)
                
                personalized_context = preferred_context + other_context
            else:
                personalized_context = context
            
            # Add personalization notes if user is novice
            if user_profile.expertise_level == "beginner":
                # Add explanatory notes
                personalized_context = await self._add_explanatory_notes(personalized_context)
            
            return personalized_context
            
        except Exception as e:
            self.logger.error(f"Error personalizing context: {e}")
            return context
    
    async def get_query_history(
        self,
        user_id: str,
        limit: int = 10,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get user query history
        """
        try:
            profile = await self.get_profile(user_id)
            if not profile:
                return []
            
            # Get paginated history
            history = profile.query_history[offset:offset + limit]
            
            return history
            
        except Exception as e:
            self.logger.error(f"Error getting query history: {e}")
            return []
    
    async def get_recommendations(
        self,
        user_id: str,
        recommendation_type: str = "general"
    ) -> List[Dict[str, Any]]:
        """
        Get personalized recommendations for user
        """
        try:
            profile = await self.get_profile(user_id)
            if not profile:
                return []
            
            recommendations = []
            
            if recommendation_type == "funds":
                recommendations = await self._recommend_funds(profile)
            elif recommendation_type == "topics":
                recommendations = await self._recommend_topics(profile)
            elif recommendation_type == "strategies":
                recommendations = await self._recommend_strategies(profile)
            else:
                recommendations = await self._recommend_general(profile)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error getting recommendations: {e}")
            return []
    
    async def _create_profile(self, user_id: str) -> UserProfile:
        """Create new user profile"""
        profile = UserProfile(
            user_id=user_id,
            created_at=asyncio.get_event_loop().time(),
            last_activity=asyncio.get_event_loop().time(),
            preferences={},
            query_history=[],
            interaction_patterns={},
            feedback_history=[],
            risk_tolerance="moderate",  # Default
            investment_horizon="medium",  # Default
            preferred_fund_types=[],
            expertise_level="beginner",  # Default
            session_count=1,
            total_queries=0
        )
        
        self.profiles[user_id] = profile
        self.logger.info(f"Created new profile for user {user_id}")
        
        return profile
    
    async def _update_interaction_patterns(
        self,
        profile: UserProfile,
        query: str,
        retrieved_docs: List[Dict[str, Any]]
    ) -> None:
        """Update user interaction patterns"""
        patterns = profile.interaction_patterns
        
        # Update query patterns
        if "query_patterns" not in patterns:
            patterns["query_patterns"] = {}
        
        query_patterns = patterns["query_patterns"]
        
        # Analyze query type
        query_lower = query.lower()
        if any(word in query_lower for word in ["performance", "returns", "nav"]):
            query_patterns["performance_queries"] = query_patterns.get("performance_queries", 0) + 1
        elif any(word in query_lower for word in ["risk", "safety", "conservative"]):
            query_patterns["risk_queries"] = query_patterns.get("risk_queries", 0) + 1
        elif any(word in query_lower for word in ["compare", "vs", "better"]):
            query_patterns["comparison_queries"] = query_patterns.get("comparison_queries", 0) + 1
        
        # Update document interaction patterns
        if "doc_interactions" not in patterns:
            patterns["doc_interactions"] = {}
        
        doc_interactions = patterns["doc_interactions"]
        for doc in retrieved_docs:
            doc_type = doc.get("metadata", {}).get("type", "unknown")
            doc_interactions[doc_type] = doc_interactions.get(doc_type, 0) + 1
    
    async def _infer_preferences(
        self,
        profile: UserProfile,
        query: str,
        retrieved_docs: List[Dict[str, Any]]
    ) -> None:
        """Infer user preferences from behavior"""
        query_lower = query.lower()
        
        # Infer risk tolerance
        if "low risk" in query_lower or "conservative" in query_lower:
            profile.risk_tolerance = "low"
        elif "high risk" in query_lower or "aggressive" in query_lower:
            profile.risk_tolerance = "high"
        
        # Infer investment horizon
        if "short term" in query_lower or "quick" in query_lower:
            profile.investment_horizon = "short"
        elif "long term" in query_lower or "retirement" in query_lower:
            profile.investment_horizon = "long"
        
        # Infer preferred fund types
        for doc in retrieved_docs:
            fund_type = doc.get("metadata", {}).get("fund_type")
            if fund_type and fund_type not in profile.preferred_fund_types:
                # Add to preferences if user interacts with this type frequently
                type_interactions = profile.interaction_patterns.get("doc_interactions", {})
                if type_interactions.get(fund_type, 0) >= 3:
                    profile.preferred_fund_types.append(fund_type)
    
    async def _process_feedback(
        self,
        profile: UserProfile,
        feedback_data: Dict[str, Any]
    ) -> None:
        """Process user feedback to update preferences"""
        rating = feedback_data.get("rating", 0)
        
        # Update preferences based on high ratings
        if rating >= 4:
            # Positive feedback - reinforce current preferences
            pass
        elif rating <= 2:
            # Negative feedback - adjust preferences
            pass
        
        # Update explicit preferences
        if "preferences" in feedback_data:
            profile.preferences.update(feedback_data["preferences"])
    
    async def _analyze_query_complexity(self, query: str) -> str:
        """Analyze query complexity"""
        words = query.split()
        
        if len(words) <= 5:
            return "simple"
        elif len(words) <= 10:
            return "moderate"
        else:
            return "complex"
    
    async def _update_expertise_level(
        self,
        current_level: str,
        query_complexity: str
    ) -> str:
        """Update user expertise level"""
        # Simple progression based on query complexity
        if current_level == "beginner":
            if query_complexity in ["moderate", "complex"]:
                return "intermediate"
        elif current_level == "intermediate":
            if query_complexity == "complex":
                return "advanced"
        
        return current_level
    
    async def _add_explanatory_notes(self, context: List[str]) -> List[str]:
        """Add explanatory notes for novice users"""
        enhanced_context = []
        
        for content in context:
            enhanced_content = content
            
            # Add explanations for financial terms
            if "nav" in content.lower():
                enhanced_content += " (NAV is the Net Asset Value per unit of the fund)"
            elif "sip" in content.lower():
                enhanced_content += " (SIP is Systematic Investment Plan for regular investments)"
            
            enhanced_context.append(enhanced_content)
        
        return enhanced_context
    
    async def _recommend_funds(self, profile: UserProfile) -> List[Dict[str, Any]]:
        """Recommend funds based on user profile"""
        recommendations = []
        
        # Mock recommendations based on preferences
        if profile.risk_tolerance == "low":
            recommendations.extend([
                {"type": "fund", "name": "Conservative Hybrid Fund", "reason": "Low risk tolerance"},
                {"type": "fund", "name": "Debt Fund", "reason": "Capital preservation"}
            ])
        elif profile.risk_tolerance == "high":
            recommendations.extend([
                {"type": "fund", "name": "Mid-Cap Equity Fund", "reason": "High growth potential"},
                {"type": "fund", "name": "Sector Fund", "reason": "Aggressive growth strategy"}
            ])
        
        # Add preferred fund types
        for fund_type in profile.preferred_fund_types:
            recommendations.append({
                "type": "fund",
                "name": f"{fund_type.title()} Fund",
                "reason": "Matches your preference"
            })
        
        return recommendations[:5]
    
    async def _recommend_topics(self, profile: UserProfile) -> List[Dict[str, Any]]:
        """Recommend topics based on user profile"""
        recommendations = []
        
        # Analyze query patterns
        patterns = profile.interaction_patterns.get("query_patterns", {})
        
        if patterns.get("performance_queries", 0) > 3:
            recommendations.append({
                "type": "topic",
                "name": "Fund Performance Analysis",
                "reason": "Based on your interest in performance"
            })
        
        if patterns.get("risk_queries", 0) > 2:
            recommendations.append({
                "type": "topic",
                "name": "Risk Management Strategies",
                "reason": "Based on your risk concerns"
            })
        
        # Add expertise-based recommendations
        if profile.expertise_level == "beginner":
            recommendations.append({
                "type": "topic",
                "name": "Mutual Fund Basics",
                "reason": "Great for beginners"
            })
        
        return recommendations[:5]
    
    async def _recommend_strategies(self, profile: UserProfile) -> List[Dict[str, Any]]:
        """Recommend investment strategies"""
        strategies = []
        
        if profile.investment_horizon == "short":
            strategies.append({
                "type": "strategy",
                "name": "Liquidity-focused Strategy",
                "reason": "Short investment horizon"
            })
        elif profile.investment_horizon == "long":
            strategies.append({
                "type": "strategy",
                "name": "Wealth Accumulation Strategy",
                "reason": "Long-term growth focus"
            })
        
        # Risk-based strategies
        if profile.risk_tolerance == "low":
            strategies.append({
                "type": "strategy",
                "name": "Conservative Portfolio Strategy",
                "reason": "Matches your risk tolerance"
            })
        
        return strategies[:3]
    
    async def _recommend_general(self, profile: UserProfile) -> List[Dict[str, Any]]:
        """General recommendations"""
        recommendations = []
        
        # Based on expertise level
        if profile.expertise_level == "beginner":
            recommendations.extend([
                {"type": "general", "name": "Investment Education", "reason": "Build your knowledge"},
                {"type": "general", "name": "Risk Assessment", "reason": "Understand your risk profile"}
            ])
        
        # Based on activity
        if profile.total_queries < 5:
            recommendations.append({
                "type": "general",
                "name": "Portfolio Diversification",
                "reason": "Essential for new investors"
            })
        
        return recommendations[:5]
    
    async def get_user_segments(self) -> Dict[str, List[str]]:
        """Segment users based on profiles"""
        segments = {
            "beginners": [],
            "intermediate": [],
            "advanced": [],
            "conservative": [],
            "aggressive": [],
            "active": []
        }
        
        for user_id, profile in self.profiles.items():
            # Expertise segmentation
            segments[profile.expertise_level].append(user_id)
            
            # Risk segmentation
            if profile.risk_tolerance == "low":
                segments["conservative"].append(user_id)
            elif profile.risk_tolerance == "high":
                segments["aggressive"].append(user_id)
            
            # Activity segmentation
            if profile.total_queries > 20:
                segments["active"].append(user_id)
        
        return segments
    
    async def get_profiler_statistics(self) -> Dict[str, Any]:
        """Get profiler statistics"""
        try:
            total_users = len(self.profiles)
            total_interactions = sum(len(interactions) for interactions in self.interactions.values())
            
            # Expertise distribution
            expertise_counts = defaultdict(int)
            risk_counts = defaultdict(int)
            
            for profile in self.profiles.values():
                expertise_counts[profile.expertise_level] += 1
                risk_counts[profile.risk_tolerance] += 1
            
            return {
                "total_users": total_users,
                "total_interactions": total_interactions,
                "expertise_distribution": dict(expertise_counts),
                "risk_tolerance_distribution": dict(risk_counts),
                "avg_queries_per_user": sum(p.total_queries for p in self.profiles.values()) / total_users if total_users > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error getting profiler statistics: {e}")
            return {}
    
    async def cleanup_inactive_profiles(self, days: int = 30) -> int:
        """Clean up inactive user profiles"""
        try:
            cutoff_time = asyncio.get_event_loop().time() - (days * 24 * 3600)
            inactive_users = []
            
            for user_id, profile in self.profiles.items():
                if profile.last_activity < cutoff_time:
                    inactive_users.append(user_id)
            
            # Remove inactive profiles
            for user_id in inactive_users:
                del self.profiles[user_id]
                if user_id in self.interactions:
                    del self.interactions[user_id]
            
            self.logger.info(f"Cleaned up {len(inactive_users)} inactive profiles")
            return len(inactive_users)
            
        except Exception as e:
            self.logger.error(f"Error cleaning up profiles: {e}")
            return 0
