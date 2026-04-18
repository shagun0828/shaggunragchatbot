"""
Intelligent URL Router for Phase 4.3
Smart routing system that determines optimal model assignment
Routes URLs to BGE-base (20 URLs) or BGE-small (5 URLs) based on analysis
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlparse
import json
from collections import defaultdict
import numpy as np


class ModelType(Enum):
    """Available BGE model types"""
    BGE_BASE = "bge_base"
    BGE_SMALL = "bge_small"


class ContentType(Enum):
    """Content types for URL classification"""
    MUTUAL_FUND = "mutual_fund"
    FINANCIAL_NEWS = "financial_news"
    MARKET_DATA = "market_data"
    COMPANY_DATA = "company_data"
    GENERAL_FINANCIAL = "general_financial"
    NON_FINANCIAL = "non_financial"


class ComplexityLevel(Enum):
    """Content complexity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class URLAnalysis:
    """Comprehensive URL analysis results"""
    url: str
    domain: str
    path: str
    content_type: ContentType
    complexity_level: ComplexityLevel
    complexity_score: float
    financial_relevance: float
    processing_priority: int
    recommended_model: ModelType
    confidence: float
    reasoning: str
    metadata: Dict[str, Any]


@dataclass
class RoutingDecision:
    """Model routing decision"""
    url: str
    model_type: ModelType
    content_type: ContentType
    complexity_level: ComplexityLevel
    reasoning: str
    confidence: float
    processing_group: str
    priority: int


class IntelligentURLRouter:
    """Intelligent URL router with advanced analysis capabilities"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        
        # URL pattern definitions
        self.url_patterns = self._load_url_patterns()
        
        # Model selection rules
        self.model_rules = self._load_model_rules()
        
        # Complexity analysis parameters
        self.complexity_analyzers = self._initialize_complexity_analyzers()
        
        # Routing cache
        self.routing_cache = {}
        self.analysis_cache = {}
        
        # Statistics
        self.routing_stats = defaultdict(int)
        
        self.logger.info("Intelligent URL router initialized with advanced analysis capabilities")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'enable_caching': True,
            'cache_size': 1000,
            'default_model': ModelType.BGE_SMALL,
            'complexity_thresholds': {
                'low': 0.3,
                'medium': 0.6,
                'high': 0.8,
                'very_high': 0.9
            },
            'model_limits': {
                ModelType.BGE_BASE: 20,
                ModelType.BGE_SMALL: 5
            },
            'priority_weights': {
                'content_type': 0.4,
                'complexity': 0.3,
                'financial_relevance': 0.2,
                'domain_authority': 0.1
            }
        }
    
    def _load_url_patterns(self) -> Dict[ContentType, List[str]]:
        """Load comprehensive URL patterns for content classification"""
        return {
            ContentType.MUTUAL_FUND: [
                r'groww\.in/mutual-funds',
                r'moneycontrol\.com/mutual-funds',
                r'valueresearchonline\.com/funds',
                r'morningstar\.in/funds',
                r'fundsupermart\.com',
                r'etmoney\.com/mutual-funds',
                r'zerodha\.com/mutual-funds',
                r'icicidirect\.com/mutual-funds',
                r'hdfcsec\.com/mutual-funds',
                r'axisdirect\.in/mutual-funds'
            ],
            ContentType.FINANCIAL_NEWS: [
                r'economictimes\.indiatimes\.com',
                r'livemint\.com',
                r'business-standard\.com',
                r'financial-express\.com',
                r'moneycontrol\.com/news',
                r'ndtv\.com/business',
                r'thehindubusinessline\.com',
                r'bsiness\.in',
                r'vccircle\.com',
                r'yourstory\.com'
            ],
            ContentType.MARKET_DATA: [
                r'nseindia\.com',
                r'bseindia\.com',
                r'moneycontrol\.com/stock-price',
                r'economicstimes\.com/markets',
                'livemint\.com/market',
                r'ndtv\.com/business/markets',
                r'business-standard\.com/markets',
                r'financial-express\.com/market'
            ],
            ContentType.COMPANY_DATA: [
                r'annualreports\.com',
                r'investor\.bseindia\.com',
                r'nseindia\.com/companies',
                r'rbi\.org\.in',
                r'sebi\.gov\.in',
                r'mca\.gov\.in',
                r'bseindia\.com/corporates',
                r'nseindia\.com/corporate-actions'
            ],
            ContentType.GENERAL_FINANCIAL: [
                r'bankbazaar\.com',
                r'paisabazaar\.com',
                r'policybazaar\.com',
                r'coverfox\.com',
                r'acko\.com',
                r'digit\.insurance'
            ]
        }
    
    def _load_model_rules(self) -> Dict[ModelType, Dict[str, Any]]:
        """Load model selection rules"""
        return {
            ModelType.BGE_BASE: {
                'max_urls': 20,
                'preferred_content_types': [ContentType.MUTUAL_FUND, ContentType.COMPANY_DATA],
                'min_complexity': 0.6,
                'min_financial_relevance': 0.5,
                'dimension': 768,
                'quality': 'high',
                'use_cases': ['complex_analysis', 'detailed_research', 'comprehensive_data']
            },
            ModelType.BGE_SMALL: {
                'max_urls': 5,
                'preferred_content_types': [ContentType.FINANCIAL_NEWS, ContentType.MARKET_DATA],
                'min_complexity': 0.2,
                'min_financial_relevance': 0.3,
                'dimension': 384,
                'quality': 'good',
                'use_cases': ['quick_analysis', 'real_time_data', 'news_processing']
            }
        }
    
    def _initialize_complexity_analyzers(self) -> Dict[str, Any]:
        """Initialize complexity analysis parameters"""
        return {
            'url_depth_weight': 0.1,
            'parameter_weight': 0.05,
            'domain_weight': 0.2,
            'content_type_weight': 0.3,
            'path_complexity_weight': 0.15,
            'subdomain_weight': 0.1,
            'tld_weight': 0.1
        }
    
    def analyze_url(self, url: str) -> URLAnalysis:
        """Comprehensive URL analysis"""
        if self.config['enable_caching'] and url in self.analysis_cache:
            return self.analysis_cache[url]
        
        # Parse URL
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path.lower()
        
        # Determine content type
        content_type = self._determine_content_type(url, domain, path)
        
        # Calculate complexity score
        complexity_score = self._calculate_complexity_score(url, domain, path, content_type)
        complexity_level = self._determine_complexity_level(complexity_score)
        
        # Calculate financial relevance
        financial_relevance = self._calculate_financial_relevance(url, content_type, domain)
        
        # Calculate processing priority
        processing_priority = self._calculate_processing_priority(
            content_type, complexity_score, financial_relevance, domain
        )
        
        # Recommend model
        recommended_model, confidence, reasoning = self._recommend_model(
            content_type, complexity_score, financial_relevance, processing_priority
        )
        
        # Create analysis result
        analysis = URLAnalysis(
            url=url,
            domain=domain,
            path=path,
            content_type=content_type,
            complexity_level=complexity_level,
            complexity_score=complexity_score,
            financial_relevance=financial_relevance,
            processing_priority=processing_priority,
            recommended_model=recommended_model,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                'url_length': len(url),
                'path_segments': len([seg for seg in path.split('/') if seg]),
                'has_parameters': '?' in url,
                'subdomain_count': len(domain.split('.')) - 1,
                'tld': domain.split('.')[-1] if '.' in domain else ''
            }
        )
        
        # Cache result
        if self.config['enable_caching']:
            self.analysis_cache[url] = analysis
        
        return analysis
    
    def _determine_content_type(self, url: str, domain: str, path: str) -> ContentType:
        """Determine content type from URL analysis"""
        url_lower = url.lower()
        
        # Check against patterns
        for content_type, patterns in self.url_patterns.items():
            for pattern in patterns:
                if re.search(pattern, url_lower, re.IGNORECASE):
                    return content_type
        
        # Fallback analysis
        if any(keyword in url_lower for keyword in ['fund', 'mutual', 'sip', 'nav', 'aum']):
            return ContentType.MUTUAL_FUND
        elif any(keyword in url_lower for keyword in ['news', 'article', 'story', 'blog']):
            return ContentType.FINANCIAL_NEWS
        elif any(keyword in url_lower for keyword in ['market', 'stock', 'index', 'nse', 'bse']):
            return ContentType.MARKET_DATA
        elif any(keyword in url_lower for keyword in ['company', 'corporate', 'investor', 'annual']):
            return ContentType.COMPANY_DATA
        elif any(keyword in url_lower for keyword in ['bank', 'insurance', 'loan', 'emi']):
            return ContentType.GENERAL_FINANCIAL
        else:
            return ContentType.NON_FINANCIAL
    
    def _calculate_complexity_score(self, url: str, domain: str, path: str, content_type: ContentType) -> float:
        """Calculate comprehensive complexity score"""
        score = 0.0
        
        # Base score by content type
        content_type_scores = {
            ContentType.MUTUAL_FUND: 0.8,
            ContentType.COMPANY_DATA: 0.7,
            ContentType.MARKET_DATA: 0.5,
            ContentType.FINANCIAL_NEWS: 0.4,
            ContentType.GENERAL_FINANCIAL: 0.3,
            ContentType.NON_FINANCIAL: 0.1
        }
        score += content_type_scores.get(content_type, 0.1)
        
        # URL depth complexity
        path_segments = len([seg for seg in path.split('/') if seg])
        score += min(path_segments * self.complexity_analyzers['url_depth_weight'], 0.2)
        
        # Parameter complexity
        if '?' in url:
            param_count = len(url.split('?')[1].split('&'))
            score += min(param_count * self.complexity_analyzers['parameter_weight'], 0.15)
        
        # Domain complexity
        subdomain_count = len(domain.split('.')) - 1
        score += min(subdomain_count * self.complexity_analyzers['subdomain_weight'], 0.1)
        
        # Path complexity
        path_indicators = ['id', 'detail', 'view', 'show', 'data', 'report', 'analysis']
        path_complexity = sum(1 for indicator in path_indicators if indicator in path)
        score += min(path_complexity * self.complexity_analyzers['path_complexity_weight'], 0.2)
        
        # TLD complexity
        complex_tlds = ['com', 'org', 'gov', 'in', 'co.in']
        tld = domain.split('.')[-1] if '.' in domain else ''
        if tld in complex_tlds:
            score += self.complexity_analyzers['tld_weight']
        
        return min(score, 1.0)
    
    def _determine_complexity_level(self, score: float) -> ComplexityLevel:
        """Determine complexity level from score"""
        thresholds = self.config['complexity_thresholds']
        
        if score >= thresholds['very_high']:
            return ComplexityLevel.VERY_HIGH
        elif score >= thresholds['high']:
            return ComplexityLevel.HIGH
        elif score >= thresholds['medium']:
            return ComplexityLevel.MEDIUM
        else:
            return ComplexityLevel.LOW
    
    def _calculate_financial_relevance(self, url: str, content_type: ContentType, domain: str) -> float:
        """Calculate financial domain relevance score"""
        relevance = 0.0
        
        # Base relevance by content type
        content_type_relevance = {
            ContentType.MUTUAL_FUND: 0.9,
            ContentType.COMPANY_DATA: 0.8,
            ContentType.MARKET_DATA: 0.7,
            ContentType.FINANCIAL_NEWS: 0.6,
            ContentType.GENERAL_FINANCIAL: 0.5,
            ContentType.NON_FINANCIAL: 0.1
        }
        relevance += content_type_relevance.get(content_type, 0.1)
        
        # Domain authority boost
        high_authority_domains = [
            'nseindia.com', 'bseindia.com', 'sebi.gov.in', 'rbi.org.in',
            'valueresearchonline.com', 'morningstar.in', 'moneycontrol.com'
        ]
        if any(auth_domain in domain for auth_domain in high_authority_domains):
            relevance += 0.1
        
        # Financial keywords in URL
        financial_keywords = [
            'fund', 'mutual', 'nav', 'aum', 'return', 'risk', 'portfolio',
            'investment', 'sip', 'lumpsum', 'market', 'stock', 'equity', 'debt'
        ]
        keyword_count = sum(1 for keyword in financial_keywords if keyword in url.lower())
        relevance += min(keyword_count * 0.05, 0.2)
        
        return min(relevance, 1.0)
    
    def _calculate_processing_priority(self, content_type: ContentType, complexity_score: float, 
                                     financial_relevance: float, domain: str) -> int:
        """Calculate processing priority"""
        priority = 1
        
        # Priority by content type
        content_priorities = {
            ContentType.MUTUAL_FUND: 5,
            ContentType.COMPANY_DATA: 4,
            ContentType.MARKET_DATA: 3,
            ContentType.FINANCIAL_NEWS: 2,
            ContentType.GENERAL_FINANCIAL: 2,
            ContentType.NON_FINANCIAL: 1
        }
        priority += content_priorities.get(content_type, 1)
        
        # Complexity boost
        if complexity_score > 0.7:
            priority += 2
        elif complexity_score > 0.5:
            priority += 1
        
        # Financial relevance boost
        if financial_relevance > 0.8:
            priority += 1
        
        return priority
    
    def _recommend_model(self, content_type: ContentType, complexity_score: float, 
                        financial_relevance: float, priority: int) -> Tuple[ModelType, float, str]:
        """Recommend optimal model based on analysis"""
        
        # BGE-base recommendation logic
        base_rules = self.model_rules[ModelType.BGE_BASE]
        if (content_type in base_rules['preferred_content_types'] and
            complexity_score >= base_rules['min_complexity'] and
            financial_relevance >= base_rules['min_financial_relevance']):
            
            confidence = min(
                (complexity_score / 1.0) * 0.4 +
                (financial_relevance / 1.0) * 0.3 +
                (priority / 10.0) * 0.3,
                0.95
            )
            
            reasoning = f"High complexity ({complexity_score:.2f}) and financial relevance ({financial_relevance:.2f}) with {content_type.value} content"
            
            return ModelType.BGE_BASE, confidence, reasoning
        
        # BGE-small recommendation logic
        small_rules = self.model_rules[ModelType.BGE_SMALL]
        if (content_type in small_rules['preferred_content_types'] or
            complexity_score < base_rules['min_complexity'] or
            financial_relevance < base_rules['min_financial_relevance']):
            
            confidence = min(
                (1.0 - complexity_score) * 0.4 +
                (financial_relevance / 1.0) * 0.3 +
                (priority / 10.0) * 0.3,
                0.90
            )
            
            reasoning = f"Optimal for {content_type.value} content with complexity {complexity_score:.2f}"
            
            return ModelType.BGE_SMALL, confidence, reasoning
        
        # Default fallback
        default_model = self.config['default_model']
        reasoning = f"Default model selection for {content_type.value} content"
        
        return default_model, 0.5, reasoning
    
    def route_urls(self, urls: List[str]) -> List[RoutingDecision]:
        """Route multiple URLs to optimal models"""
        self.logger.info(f"Routing {len(urls)} URLs with intelligent analysis")
        
        # Analyze all URLs
        analyses = []
        for url in urls:
            analysis = self.analyze_url(url)
            analyses.append(analysis)
        
        # Sort by priority
        analyses.sort(key=lambda x: x.processing_priority, reverse=True)
        
        # Track model usage
        model_usage = {ModelType.BGE_BASE: 0, ModelType.BGE_SMALL: 0}
        
        # Make routing decisions with capacity constraints
        routing_decisions = []
        
        for analysis in analyses:
            # Check model availability
            recommended_model = analysis.recommended_model
            available_capacity = self.model_rules[recommended_model]['max_urls']
            current_usage = model_usage[recommended_model]
            
            if current_usage < available_capacity:
                # Use recommended model
                model_type = recommended_model
                model_usage[recommended_model] += 1
                confidence = analysis.confidence
                reasoning = analysis.reasoning
            else:
                # Find alternative model
                alternative_model = (ModelType.BGE_SMALL if recommended_model == ModelType.BGE_BASE 
                                  else ModelType.BGE_BASE)
                
                if model_usage[alternative_model] < self.model_rules[alternative_model]['max_urls']:
                    model_type = alternative_model
                    model_usage[alternative_model] += 1
                    confidence = analysis.confidence * 0.7  # Reduced confidence for fallback
                    reasoning = f"Fallback to {model_type.value} due to {recommended_model.value} capacity limit"
                else:
                    # Both models at capacity, use default
                    model_type = self.config['default_model']
                    confidence = 0.3
                    reasoning = "Both models at capacity, using default model"
            
            # Create routing decision
            decision = RoutingDecision(
                url=analysis.url,
                model_type=model_type,
                content_type=analysis.content_type,
                complexity_level=analysis.complexity_level,
                reasoning=reasoning,
                confidence=confidence,
                processing_group=self._determine_processing_group(analysis, model_type),
                priority=analysis.processing_priority
            )
            
            routing_decisions.append(decision)
            
            # Update statistics
            self.routing_stats[f"{model_type.value}_assigned"] += 1
            self.routing_stats[f"{analysis.content_type.value}_processed"] += 1
            self.routing_stats[f"{analysis.complexity_level.value}_complexity"] += 1
        
        self.logger.info(f"Routing completed: {model_usage[ModelType.BGE_BASE]} URLs to BGE-base, "
                        f"{model_usage[ModelType.BGE_SMALL]} URLs to BGE-small")
        
        return routing_decisions
    
    def _determine_processing_group(self, analysis: URLAnalysis, model_type: ModelType) -> str:
        """Determine processing group for batch optimization"""
        group_parts = [
            model_type.value,
            analysis.content_type.value,
            analysis.complexity_level.value
        ]
        
        return '_'.join(group_parts)
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics"""
        return {
            'routing_stats': dict(self.routing_stats),
            'cache_stats': {
                'analysis_cache_size': len(self.analysis_cache),
                'routing_cache_size': len(self.routing_cache)
            },
            'model_rules': self.model_rules,
            'url_patterns_count': {ct.value: len(patterns) for ct, patterns in self.url_patterns.items()},
            'complexity_analyzers': self.complexity_analyzers
        }
    
    def get_model_utilization(self, routing_decisions: List[RoutingDecision]) -> Dict[str, float]:
        """Calculate model utilization rates"""
        model_counts = defaultdict(int)
        for decision in routing_decisions:
            model_counts[decision.model_type.value] += 1
        
        utilization = {}
        for model_type, count in model_counts.items():
            max_urls = self.model_rules[ModelType(model_type)]['max_urls']
            utilization[model_type] = count / max_urls if max_urls > 0 else 0
        
        return utilization
    
    def clear_cache(self) -> None:
        """Clear routing and analysis cache"""
        self.routing_cache.clear()
        self.analysis_cache.clear()
        self.logger.info("Routing cache cleared")
    
    def analyze_routing_efficiency(self, routing_decisions: List[RoutingDecision]) -> Dict[str, Any]:
        """Analyze routing efficiency and quality"""
        if not routing_decisions:
            return {'status': 'no_decisions'}
        
        # Calculate metrics
        avg_confidence = np.mean([d.confidence for d in routing_decisions])
        confidence_distribution = self._calculate_confidence_distribution(routing_decisions)
        content_type_distribution = self._calculate_content_type_distribution(routing_decisions)
        complexity_distribution = self._calculate_complexity_distribution(routing_decisions)
        
        # Model distribution
        model_distribution = defaultdict(int)
        for decision in routing_decisions:
            model_distribution[decision.model_type.value] += 1
        
        return {
            'total_decisions': len(routing_decisions),
            'average_confidence': avg_confidence,
            'confidence_distribution': confidence_distribution,
            'content_type_distribution': content_type_distribution,
            'complexity_distribution': complexity_distribution,
            'model_distribution': dict(model_distribution),
            'utilization_rates': self.get_model_utilization(routing_decisions),
            'routing_quality': self._assess_routing_quality(routing_decisions)
        }
    
    def _calculate_confidence_distribution(self, decisions: List[RoutingDecision]) -> Dict[str, int]:
        """Calculate confidence score distribution"""
        distribution = {'high': 0, 'medium': 0, 'low': 0}
        
        for decision in decisions:
            if decision.confidence >= 0.8:
                distribution['high'] += 1
            elif decision.confidence >= 0.5:
                distribution['medium'] += 1
            else:
                distribution['low'] += 1
        
        return distribution
    
    def _calculate_content_type_distribution(self, decisions: List[RoutingDecision]) -> Dict[str, int]:
        """Calculate content type distribution"""
        distribution = defaultdict(int)
        
        for decision in decisions:
            distribution[decision.content_type.value] += 1
        
        return dict(distribution)
    
    def _calculate_complexity_distribution(self, decisions: List[RoutingDecision]) -> Dict[str, int]:
        """Calculate complexity level distribution"""
        distribution = defaultdict(int)
        
        for decision in decisions:
            distribution[decision.complexity_level.value] += 1
        
        return dict(distribution)
    
    def _assess_routing_quality(self, decisions: List[RoutingDecision]) -> Dict[str, Any]:
        """Assess overall routing quality"""
        avg_confidence = np.mean([d.confidence for d in decisions])
        
        # Check for optimal model assignments
        optimal_assignments = 0
        for decision in decisions:
            analysis = self.analyze_url(decision.url)
            if decision.model_type == analysis.recommended_model:
                optimal_assignments += 1
        
        optimal_rate = optimal_assignments / len(decisions) if decisions else 0
        
        return {
            'average_confidence': avg_confidence,
            'optimal_assignment_rate': optimal_rate,
            'routing_quality_score': avg_confidence * optimal_rate,
            'total_decisions': len(decisions)
        }
