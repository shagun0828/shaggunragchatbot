"""
URL-based Model Router for Phase 4.3
Routes URLs to appropriate BGE models based on URL count and characteristics
"""

import logging
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlparse
import re


class ModelType(Enum):
    """Available BGE model types"""
    BGE_BASE = "bge-base"
    BGE_SMALL = "bge-small"


@dataclass
class URLInfo:
    """URL information for routing decisions"""
    url: str
    domain: str
    path: str
    complexity_score: float
    content_type: str
    priority: int


@dataclass
class RoutingDecision:
    """Model routing decision"""
    url: str
    model_type: ModelType
    reasoning: str
    confidence: float
    processing_group: str


class URLModelRouter:
    """Routes URLs to appropriate BGE models based on analysis"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        
        # URL patterns for different content types
        self.url_patterns = {
            'mutual_fund': [
                r'groww\.in/mutual-funds',
                r'moneycontrol\.com/mutual-funds',
                r'valueresearchonline\.com/funds',
                r'morningstar\.in/funds'
            ],
            'stock_market': [
                r'nseindia\.com',
                r'bseindia\.com',
                r'moneycontrol\.com/stock-price',
                r'economicstimes\.com/markets'
            ],
            'financial_news': [
                r'economictimes\.indiatimes\.com',
                r'livemint\.com',
                r'business-standard\.com',
                r'financial-express\.com'
            ],
            'company_data': [
                r'annualreports\.com',
                r'investor\.bseindia\.com',
                r'nseindia\.com/companies',
                r'rbi\.org\.in'
            ]
        }
        
        # Model selection rules
        self.model_rules = {
            ModelType.BGE_BASE: {
                'max_urls': 20,
                'min_complexity': 0.6,
                'preferred_content': ['mutual_fund', 'company_data'],
                'dimension': 768,
                'quality': 'high'
            },
            ModelType.BGE_SMALL: {
                'max_urls': 5,
                'min_complexity': 0.3,
                'preferred_content': ['financial_news', 'stock_market'],
                'dimension': 384,
                'quality': 'medium'
            }
        }
        
        # Routing cache
        self.routing_cache = {}
        self.url_groups = {}
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'enable_caching': True,
            'cache_size': 1000,
            'default_model': ModelType.BGE_SMALL,
            'complexity_thresholds': {
                'low': 0.3,
                'medium': 0.6,
                'high': 0.8
            }
        }
    
    def analyze_url(self, url: str) -> URLInfo:
        """Analyze URL to determine characteristics"""
        parsed = urlparse(url)
        
        # Determine content type
        content_type = self._determine_content_type(url)
        
        # Calculate complexity score
        complexity_score = self._calculate_complexity_score(url, content_type)
        
        # Determine priority
        priority = self._calculate_priority(url, content_type, complexity_score)
        
        return URLInfo(
            url=url,
            domain=parsed.netloc,
            path=parsed.path,
            complexity_score=complexity_score,
            content_type=content_type,
            priority=priority
        )
    
    def _determine_content_type(self, url: str) -> str:
        """Determine content type from URL"""
        for content_type, patterns in self.url_patterns.items():
            for pattern in patterns:
                if re.search(pattern, url, re.IGNORECASE):
                    return content_type
        return 'general'
    
    def _calculate_complexity_score(self, url: str, content_type: str) -> float:
        """Calculate complexity score for URL"""
        score = 0.0
        
        # Base score by content type
        content_scores = {
            'mutual_fund': 0.8,
            'company_data': 0.7,
            'stock_market': 0.5,
            'financial_news': 0.4,
            'general': 0.3
        }
        score += content_scores.get(content_type, 0.3)
        
        # URL depth complexity
        path_depth = len(url.split('/')) - 3  # Adjust for protocol and domain
        score += min(path_depth * 0.1, 0.2)
        
        # Parameter complexity
        if '?' in url:
            param_count = len(url.split('?')[1].split('&'))
            score += min(param_count * 0.05, 0.1)
        
        return min(score, 1.0)
    
    def _calculate_priority(self, url: str, content_type: str, complexity_score: float) -> int:
        """Calculate processing priority"""
        priority = 1
        
        # Higher priority for important content types
        if content_type in ['mutual_fund', 'company_data']:
            priority += 2
        elif content_type in ['stock_market']:
            priority += 1
        
        # Higher priority for complex content
        if complexity_score > 0.7:
            priority += 1
        
        return priority
    
    def route_urls(self, urls: List[str]) -> List[RoutingDecision]:
        """Route multiple URLs to appropriate models"""
        self.logger.info(f"Routing {len(urls)} URLs to BGE models")
        
        # Analyze all URLs
        url_infos = []
        for url in urls:
            if self.config['enable_caching'] and url in self.routing_cache:
                url_info = self.routing_cache[url]
            else:
                url_info = self.analyze_url(url)
                if self.config['enable_caching']:
                    self.routing_cache[url] = url_info
            url_infos.append(url_info)
        
        # Group URLs by characteristics
        self._group_urls(url_infos)
        
        # Make routing decisions
        routing_decisions = []
        
        # Process high-priority URLs first
        sorted_urls = sorted(url_infos, key=lambda x: x.priority, reverse=True)
        
        # Track model usage
        model_usage = {ModelType.BGE_BASE: 0, ModelType.BGE_SMALL: 0}
        
        for url_info in sorted_urls:
            decision = self._make_routing_decision(url_info, model_usage)
            routing_decisions.append(decision)
            model_usage[decision.model_type] += 1
        
        self.logger.info(f"Routing completed: {model_usage[ModelType.BGE_BASE]} URLs to BGE-base, "
                        f"{model_usage[ModelType.BGE_SMALL]} URLs to BGE-small")
        
        return routing_decisions
    
    def _group_urls(self, url_infos: List[URLInfo]) -> None:
        """Group URLs by characteristics for batch processing"""
        self.url_groups = {
            'high_complexity': [],
            'medium_complexity': [],
            'low_complexity': [],
            'mutual_funds': [],
            'financial_news': [],
            'stock_market': [],
            'company_data': []
        }
        
        for url_info in url_infos:
            # Group by complexity
            if url_info.complexity_score >= 0.7:
                self.url_groups['high_complexity'].append(url_info)
            elif url_info.complexity_score >= 0.4:
                self.url_groups['medium_complexity'].append(url_info)
            else:
                self.url_groups['low_complexity'].append(url_info)
            
            # Group by content type
            if url_info.content_type in self.url_groups:
                self.url_groups[url_info.content_type].append(url_info)
    
    def _make_routing_decision(self, url_info: URLInfo, model_usage: Dict[ModelType, int]) -> RoutingDecision:
        """Make routing decision for a single URL"""
        
        # Check model availability
        base_available = model_usage[ModelType.BGE_BASE] < self.model_rules[ModelType.BGE_BASE]['max_urls']
        small_available = model_usage[ModelType.BGE_SMALL] < self.model_rules[ModelType.BGE_SMALL]['max_urls']
        
        # Decision logic
        if url_info.complexity_score >= self.model_rules[ModelType.BGE_BASE]['min_complexity'] and base_available:
            # Use BGE-base for complex content
            model_type = ModelType.BGE_BASE
            reasoning = f"High complexity ({url_info.complexity_score:.2f}) and BGE-base available"
            confidence = 0.9
        elif url_info.content_type in self.model_rules[ModelType.BGE_BASE]['preferred_content'] and base_available:
            # Use BGE-base for preferred content
            model_type = ModelType.BGE_BASE
            reasoning = f"Preferred content type ({url_info.content_type}) and BGE-base available"
            confidence = 0.8
        elif small_available:
            # Use BGE-small
            model_type = ModelType.BGE_SMALL
            reasoning = f"BGE-small available for {url_info.content_type} content"
            confidence = 0.7
        else:
            # Fallback to default
            model_type = self.config['default_model']
            reasoning = "Fallback to default model due to model limits"
            confidence = 0.5
        
        # Determine processing group
        processing_group = self._determine_processing_group(url_info, model_type)
        
        return RoutingDecision(
            url=url_info.url,
            model_type=model_type,
            reasoning=reasoning,
            confidence=confidence,
            processing_group=processing_group
        )
    
    def _determine_processing_group(self, url_info: URLInfo, model_type: ModelType) -> str:
        """Determine processing group for batch optimization"""
        group_parts = [model_type.value, url_info.content_type]
        
        if url_info.complexity_score >= 0.7:
            group_parts.append('high_complexity')
        elif url_info.complexity_score >= 0.4:
            group_parts.append('medium_complexity')
        else:
            group_parts.append('low_complexity')
        
        return '_'.join(group_parts)
    
    def get_routing_statistics(self) -> Dict:
        """Get routing statistics"""
        return {
            'total_cached_urls': len(self.routing_cache),
            'url_groups': {group: len(urls) for group, urls in self.url_groups.items()},
            'model_rules': self.model_rules,
            'url_patterns': self.url_patterns
        }
    
    def clear_cache(self) -> None:
        """Clear routing cache"""
        self.routing_cache.clear()
        self.url_groups.clear()
        self.logger.info("Routing cache cleared")


class URLBatchProcessor:
    """Processes URLs in batches based on routing decisions"""
    
    def __init__(self, router: URLModelRouter):
        self.router = router
        self.logger = logging.getLogger(__name__)
    
    def create_processing_batches(self, routing_decisions: List[RoutingDecision]) -> Dict[str, List[RoutingDecision]]:
        """Create processing batches from routing decisions"""
        batches = {}
        
        for decision in routing_decisions:
            group = decision.processing_group
            if group not in batches:
                batches[group] = []
            batches[group].append(decision)
        
        self.logger.info(f"Created {len(batches)} processing batches")
        
        for group, decisions in batches.items():
            model_type = decisions[0].model_type.value
            self.logger.info(f"Batch '{group}': {len(decisions)} URLs with {model_type}")
        
        return batches
    
    def optimize_batch_order(self, batches: Dict[str, List[RoutingDecision]]) -> List[str]:
        """Optimize batch processing order"""
        # Sort batches by priority and model type
        batch_scores = {}
        
        for group, decisions in batches.items():
            # Calculate batch score
            avg_priority = np.mean([self.router.analyze_url(dec.url).priority for dec in decisions])
            model_priority = 2 if decisions[0].model_type == ModelType.BGE_BASE else 1
            batch_scores[group] = avg_priority * 10 + model_priority
        
        # Sort by score (highest first)
        sorted_groups = sorted(batch_scores.keys(), key=lambda x: batch_scores[x], reverse=True)
        
        self.logger.info(f"Optimized batch order: {sorted_groups}")
        
        return sorted_groups
