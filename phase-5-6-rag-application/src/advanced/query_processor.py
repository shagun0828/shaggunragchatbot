"""
Advanced Query Processor for Phase 5-6 Application
Handles query preprocessing, expansion, and optimization
"""

import asyncio
import logging
import re
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import json


class QueryProcessor:
    """Advanced query processor with expansion and optimization"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Financial vocabulary for expansion
        self.financial_synonyms = {
            "mutual fund": ["mf", "mutualfund", "fund", "investment fund"],
            "nav": ["net asset value", "asset value", "unit price"],
            "returns": ["performance", "gain", "profit", "roi"],
            "expense ratio": ["management fee", "annual expense", "fee ratio"],
            "aum": ["assets under management", "fund size", "total assets"],
            "sip": ["systematic investment plan", "regular investment"],
            "risk": ["volatility", "uncertainty", "downside"],
            "portfolio": ["holdings", "investments", "asset allocation"],
            "benchmark": ["index", "reference", "comparison"]
        }
        
        # Query patterns for classification
        self.query_patterns = {
            "comparison": [
                r"compare.*vs.*",
                r"better.*than.*",
                r"difference.*between.*",
                r"which.*is.*better.*"
            ],
            "performance": [
                r"performance.*",
                r"returns.*",
                r"how.*did.*perform.*",
                r"past.*performance.*"
            ],
            "recommendation": [
                r"recommend.*",
                r"best.*fund.*",
                r"top.*fund.*",
                r"should.*invest.*"
            ],
            "explanation": [
                r"what.*is.*",
                r"explain.*",
                r"how.*does.*work.*",
                r"why.*is.*"
            ],
            "calculation": [
                r"calculate.*",
                r"how.*much.*",
                r"what.*is.*value.*",
                r"how.*to.*calculate.*"
            ]
        }
        
        # Stop words to filter out
        self.stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "is", "are", "was", "were", "be", "been", "have",
            "has", "had", "do", "does", "did", "will", "would", "could", "should"
        }
    
    async def process_query(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process and optimize user query
        """
        try:
            self.logger.info(f"Processing query: {query[:50]}...")
            
            # Step 1: Clean and normalize query
            cleaned_query = self._clean_query(query)
            
            # Step 2: Extract entities and intent
            entities = self._extract_entities(cleaned_query)
            intent = self._classify_intent(cleaned_query)
            
            # Step 3: Expand query with synonyms
            expanded_query = self._expand_query(cleaned_query, entities)
            
            # Step 4: Optimize for search
            optimized_query = self._optimize_for_search(expanded_query, intent)
            
            # Step 5: Generate filters based on entities
            filters = self._generate_filters(entities, intent)
            
            # Step 6: Personalize if user_id provided
            if user_id:
                personalized_query = await self._personalize_query(
                    optimized_query, user_id, session_id
                )
                optimized_query = personalized_query.get("query", optimized_query)
                filters.update(personalized_query.get("filters", {}))
            
            result = {
                "original_query": query,
                "cleaned_query": cleaned_query,
                "expanded_query": expanded_query,
                "optimized_query": optimized_query,
                "entities": entities,
                "intent": intent,
                "filters": filters,
                "expanded": len(expanded_query) > len(cleaned_query),
                "personalized": user_id is not None
            }
            
            self.logger.info(f"Query processed successfully: {intent}")
            return result
            
        except Exception as e:
            self.logger.error(f"Query processing failed: {e}")
            raise
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize query text"""
        # Convert to lowercase
        cleaned = query.lower()
        
        # Remove special characters except spaces
        cleaned = re.sub(r"[^\w\s]", " ", cleaned)
        
        # Remove extra spaces
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        
        return cleaned
    
    def _extract_entities(self, query: str) -> List[Dict[str, Any]]:
        """Extract financial entities from query"""
        entities = []
        
        # Extract fund names (mock implementation)
        fund_patterns = [
            r"(hdfc|icici|axis|sbi|uti|kotak)\s+(?:fund|mutual fund)",
            r"(mid\s*cap|large\s*cap|small\s*cap|hybrid)\s+fund",
            r"(equity|debt|balanced|elss)\s+fund"
        ]
        
        for pattern in fund_patterns:
            matches = re.finditer(pattern, query)
            for match in matches:
                entities.append({
                    "type": "fund_name",
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end()
                })
        
        # Extract numerical values
        number_patterns = [
            r"(\d+(?:\.\d+)?)\s*%",
            r"(\d+(?:\.\d+)?)\s*years?",
            r"(\d+(?:\.\d+)?)\s*months?",
            r"rs?\s*(\d+(?:,\d+)*)",
            r"(\d+(?:\.\d+)?)\s*cr"
        ]
        
        for pattern in number_patterns:
            matches = re.finditer(pattern, query)
            for match in matches:
                entities.append({
                    "type": "numerical_value",
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end()
                })
        
        # Extract financial terms
        for term in self.financial_synonyms.keys():
            if term in query:
                entities.append({
                    "type": "financial_term",
                    "value": term,
                    "start": query.find(term),
                    "end": query.find(term) + len(term)
                })
        
        return entities
    
    def _classify_intent(self, query: str) -> str:
        """Classify user query intent"""
        for intent, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return intent
        
        return "information_seeking"
    
    def _expand_query(self, query: str, entities: List[Dict[str, Any]]) -> str:
        """Expand query with synonyms and related terms"""
        expanded_terms = []
        
        # Add original query terms
        original_terms = query.split()
        expanded_terms.extend(original_terms)
        
        # Add synonyms for financial terms
        for term in original_terms:
            if term in self.financial_synonyms:
                expanded_terms.extend(self.financial_synonyms[term])
        
        # Add entity-based expansions
        for entity in entities:
            if entity["type"] == "fund_name":
                # Add related terms for fund queries
                expanded_terms.extend(["performance", "nav", "returns"])
            elif entity["type"] == "numerical_value":
                # Add context for numerical queries
                if "%" in entity["value"]:
                    expanded_terms.extend(["returns", "performance"])
        
        # Remove duplicates and stop words
        unique_terms = []
        for term in expanded_terms:
            if term not in unique_terms and term not in self.stop_words:
                unique_terms.append(term)
        
        return " ".join(unique_terms)
    
    def _optimize_for_search(self, query: str, intent: str) -> str:
        """Optimize query for better search results"""
        optimized_terms = query.split()
        
        # Add search-specific terms based on intent
        if intent == "performance":
            optimized_terms.extend(["historical", "performance", "returns"])
        elif intent == "recommendation":
            optimized_terms.extend(["top", "best", "recommended"])
        elif intent == "comparison":
            optimized_terms.extend(["compare", "versus", "difference"])
        elif intent == "explanation":
            optimized_terms.extend(["explain", "understand", "learn"])
        
        # Remove very short terms
        optimized_terms = [term for term in optimized_terms if len(term) > 2]
        
        return " ".join(optimized_terms)
    
    def _generate_filters(self, entities: List[Dict[str, Any]], intent: str) -> Dict[str, Any]:
        """Generate search filters based on entities and intent"""
        filters = {}
        
        # Filter by fund type
        fund_types = []
        for entity in entities:
            if entity["type"] == "fund_name":
                entity_value = entity["value"].lower()
                if "equity" in entity_value:
                    fund_types.append("equity")
                elif "debt" in entity_value:
                    fund_types.append("debt")
                elif "hybrid" in entity_value:
                    fund_types.append("hybrid")
                elif "elss" in entity_value:
                    fund_types.append("elss")
        
        if fund_types:
            filters["fund_type"] = fund_types
        
        # Filter by time period
        time_periods = []
        for entity in entities:
            if entity["type"] == "numerical_value":
                value = entity["value"].lower()
                if "year" in value:
                    if "1" in value:
                        time_periods.append("1_year")
                    elif "3" in value:
                        time_periods.append("3_years")
                    elif "5" in value:
                        time_periods.append("5_years")
        
        if time_periods:
            filters["time_period"] = time_periods
        
        # Add intent-based filters
        if intent == "recommendation":
            filters["minimum_rating"] = 4  # Only recommend highly rated funds
        elif intent == "performance":
            filters["has_performance_data"] = True
        
        return filters
    
    async def _personalize_query(
        self,
        query: str,
        user_id: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Personalize query based on user profile"""
        # Mock personalization - in production, would use actual user data
        user_profile = await self._get_user_profile(user_id)
        
        personalized_query = query
        additional_filters = {}
        
        # Add user preferences to query
        if user_profile.get("risk_preference") == "low":
            personalized_query += " low risk conservative"
            additional_filters["risk_level"] = ["low", "moderate"]
        elif user_profile.get("risk_preference") == "high":
            personalized_query += " high risk aggressive"
            additional_filters["risk_level"] = ["high", "moderate"]
        
        # Add user's preferred fund types
        preferred_types = user_profile.get("preferred_fund_types", [])
        if preferred_types:
            personalized_query += " " + " ".join(preferred_types)
            additional_filters["fund_type"] = preferred_types
        
        return {
            "query": personalized_query,
            "filters": additional_filters,
            "user_profile": user_profile
        }
    
    async def _get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user profile for personalization"""
        # Mock user profile - in production, would fetch from database
        return {
            "user_id": user_id,
            "risk_preference": "moderate",
            "preferred_fund_types": ["equity", "hybrid"],
            "investment_horizon": "5_years",
            "last_queries": ["mutual fund returns", "best sip plans"],
            "interaction_count": 15
        }
    
    async def analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """Analyze query complexity and characteristics"""
        complexity_score = 0
        characteristics = []
        
        # Length complexity
        word_count = len(query.split())
        if word_count > 10:
            complexity_score += 2
            characteristics.append("long_query")
        elif word_count > 5:
            complexity_score += 1
            characteristics.append("medium_query")
        else:
            characteristics.append("short_query")
        
        # Entity complexity
        entities = self._extract_entities(query)
        if len(entities) > 3:
            complexity_score += 2
            characteristics.append("entity_rich")
        elif len(entities) > 1:
            complexity_score += 1
            characteristics.append("entity_moderate")
        
        # Question complexity
        if "?" in query:
            complexity_score += 1
            characteristics.append("question")
        
        # Comparison complexity
        if any(word in query.lower() for word in ["vs", "versus", "compare", "better", "difference"]):
            complexity_score += 2
            characteristics.append("comparison")
        
        # Numerical complexity
        if any(char.isdigit() for char in query):
            complexity_score += 1
            characteristics.append("numerical")
        
        return {
            "complexity_score": complexity_score,
            "characteristics": characteristics,
            "word_count": word_count,
            "entity_count": len(entities)
        }
    
    async def suggest_query_improvements(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> List[str]:
        """Suggest improvements for better search results"""
        suggestions = []
        
        # Check if query is too short
        if len(query.split()) < 3:
            suggestions.append("Consider adding more specific terms to your query")
        
        # Check if no results found
        if not results:
            suggestions.append("Try using different keywords or check spelling")
            suggestions.append("Consider searching for broader terms")
        
        # Check if results are too many
        if len(results) > 50:
            suggestions.append("Add more specific filters to narrow down results")
        
        # Check if query lacks financial terms
        financial_terms = ["fund", "nav", "returns", "investment", "mutual"]
        if not any(term in query.lower() for term in financial_terms):
            suggestions.append("Include financial terms like 'fund', 'returns', or 'nav'")
        
        # Check for common misspellings
        common_misspellings = {
            "mutualfund": "mutual fund",
            "navs": "nav",
            "retuns": "returns",
            "expence": "expense"
        }
        
        for misspelling, correction in common_misspellings.items():
            if misspelling in query.lower():
                suggestions.append(f"Consider using '{correction}' instead of '{misspelling}'")
        
        return suggestions[:5]  # Return top 5 suggestions
    
    async def get_query_statistics(
        self,
        time_period: str = "7d"
    ) -> Dict[str, Any]:
        """Get query statistics for analysis"""
        # Mock statistics - in production, would fetch from analytics
        return {
            "total_queries": 1250,
            "unique_queries": 890,
            "avg_query_length": 6.8,
            "most_common_terms": [
                {"term": "fund", "count": 234},
                {"term": "returns", "count": 189},
                {"term": "nav", "count": 156},
                {"term": "investment", "count": 134},
                {"term": "mutual", "count": 112}
            ],
            "query_intents": {
                "information_seeking": 45,
                "performance": 28,
                "recommendation": 18,
                "comparison": 7,
                "explanation": 2
            },
            "time_period": time_period
        }


class QueryCache:
    """Cache for processed queries to improve performance"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = {}
        self.logger = logging.getLogger(__name__)
    
    def get(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached query processing result"""
        if query_hash in self.cache:
            result = self.cache[query_hash]
            result["cache_hit"] = True
            self.logger.debug(f"Cache hit for query hash: {query_hash}")
            return result
        return None
    
    def set(self, query_hash: str, result: Dict[str, Any]) -> None:
        """Cache query processing result"""
        # Remove oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        result["cache_hit"] = False
        result["cached_at"] = datetime.now().isoformat()
        self.cache[query_hash] = result
        self.logger.debug(f"Cached query result for hash: {query_hash}")
    
    def clear(self) -> None:
        """Clear cache"""
        self.cache.clear()
        self.logger.info("Query cache cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "utilization": len(self.cache) / self.max_size * 100
        }


class QueryOptimizer:
    """Optimizes queries for better search performance"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def optimize_query_batch(
        self,
        queries: List[str]
    ) -> List[Dict[str, Any]]:
        """Optimize a batch of queries"""
        optimized_results = []
        
        for query in queries:
            try:
                # Simple optimization - would be more sophisticated in production
                optimized_query = self._quick_optimize(query)
                optimized_results.append({
                    "original": query,
                    "optimized": optimized_query,
                    "improvement": len(optimized_query) > len(query)
                })
            except Exception as e:
                self.logger.error(f"Error optimizing query '{query}': {e}")
                optimized_results.append({
                    "original": query,
                    "optimized": query,
                    "improvement": False,
                    "error": str(e)
                })
        
        return optimized_results
    
    def _quick_optimize(self, query: str) -> str:
        """Quick query optimization"""
        # Remove common stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"}
        
        words = query.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        
        # Add common financial terms if missing
        financial_terms = ["fund", "mutual", "investment", "returns"]
        has_financial = any(term in " ".join(filtered_words).lower() for term in financial_terms)
        
        if not has_financial:
            filtered_words.append("fund")
        
        return " ".join(filtered_words)
