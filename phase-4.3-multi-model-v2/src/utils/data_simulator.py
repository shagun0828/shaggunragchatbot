"""
Data simulator for Phase 4.3
Simulates realistic content for URLs during testing and demonstration
"""

import random
from typing import Dict, Any, List
from urllib.parse import urlparse


class DataSimulator:
    """Simulates content for URLs during testing"""
    
    def __init__(self):
        self.mutual_fund_templates = [
            "The fund has delivered impressive returns of {return}% in the last year. The current NAV stands at {nav} with AUM of {aum}. The fund manager {manager} has successfully navigated market volatility.",
            
            "Performance metrics show {year}-year returns of {returns}%. The expense ratio is {expense}% with an exit load of {load}%. The fund follows a {strategy} investment strategy.",
            
            "Top holdings include {holding1} ({holding1_pct}%), {holding2} ({holding2_pct}%), and {holding3} ({holding3_pct}%). Sector allocation shows {sector1}% in {sector1_name} and {sector2}% in {sector2_name}.",
            
            "Risk analysis indicates a {risk} risk rating with a beta of {beta}. The fund's standard deviation stands at {volatility}%, reflecting {market} market conditions.",
            
            "Investment objective focuses on long-term capital appreciation by investing in a diversified portfolio of {category} companies with {focus} potential."
        ]
        
        self.news_templates = [
            "The {market} market showed {trend} movement today with the {index} index {change} by {points} points. Investors are {sentiment} about the {sector} sector performance.",
            
            "{company} announced its quarterly results with {performance} performance. The company reported {metric} of {value}, {comparison} compared to the same period last year.",
            
            "Market analysts predict {outlook} for the {industry} industry in the coming quarter. Key factors include {factor1} and {factor2} affecting market sentiment.",
            
            "The {regulator} regulatory body announced new {policy} policies aimed at {goal}. This is expected to {impact} the {affected} sector significantly.",
            
            "Investors are closely watching {indicator} indicators as the economy shows {economic_trend} trends. The {central_bank} may {action} in response to {economic_condition}."
        ]
        
        self.fund_data = {
            'returns': ['24.5', '18.2', '16.8', '22.3', '19.7', '21.1', '17.9', '20.4', '15.6', '23.1'],
            'nav': ['â¹175.43', 'â¹145.67', 'â¹189.23', 'â¹167.89', 'â¹201.45', 'â¹156.78', 'â¹192.34', 'â¹178.90', 'â¹203.12', 'â¹187.56'],
            'aum': ['â¹28,432 Cr', 'â¹15,234 Cr', 'â¹32,156 Cr', 'â¹18,789 Cr', 'â¹25,678 Cr', 'â¹12,345 Cr', 'â¹21,987 Cr', 'â¹19,876 Cr', 'â¹14,321 Cr', 'â¹27,654 Cr'],
            'managers': ['Rashmi Joshi', 'Vijay Kuppa', 'Anupam Tiwari', 'Rajeev Thakkar', 'Nimish Shah', 'Saurabh Mukherjea', 'Prashant Jain', 'Neelesh Surana', 'Anil Kumar', 'Deepak Agrawal'],
            'expense': ['1.25', '1.15', '1.35', '1.10', '1.20', '1.30', '1.05', '1.40', '1.18', '1.22'],
            'load': ['1%', '0.5%', '0%', '1.5%', '0.75%', '2%', '1.25%', '0.8%', '1.1%', '0.9%'],
            'strategies': ['growth-oriented', 'value-oriented', 'balanced', 'aggressive', 'conservative', 'dynamic', 'systematic', 'opportunistic', 'quality-focused', 'diversified'],
            'categories': ['mid-cap', 'large-cap', 'small-cap', 'multi-cap', 'focused', 'sector', 'thematic', 'hybrid', 'arbitrage', 'balanced'],
            'risk': ['Very High', 'High', 'Moderately High', 'Moderate', 'Low', 'Very Low'],
            'beta': ['1.2', '1.1', '0.9', '1.3', '0.8', '1.4', '1.0', '1.15', '0.85', '1.25'],
            'volatility': ['18.5%', '15.2%', '12.8%', '21.3%', '14.7%', '19.8%', '16.9%', '13.4%', '17.2%', '20.1%'],
            'years': ['1', '3', '5'],
            'holding1': ['Reliance Industries', 'TCS', 'HDFC Bank', 'Infosys', 'ICICI Bank', 'HUL', 'ITC', 'Bharti Airtel', 'SBI', 'L&T'],
            'holding2': ['Kotak Mahindra Bank', 'HDFC', 'Maruti Suzuki', 'M&M', 'Asian Paints', 'Sun Pharma', 'Dr. Reddy\'s Labs', 'Wipro', 'Tech Mahindra', 'Nestle'],
            'holding3': ['Axis Bank', 'IndusInd Bank', 'Bajaj Finance', 'Bajaj Auto', 'Hero MotoCorp', 'Titan', 'Dabur', 'Britannia', 'Godrej Consumer', 'Colgate'],
            'holding1_pct': ['8.5%', '7.2%', '6.8%', '9.1%', '5.7%', '7.8%', '6.3%', '8.2%', '5.9%', '7.4%'],
            'holding2_pct': ['6.2%', '8.1%', '7.5%', '5.8%', '9.3%', '6.7%', '8.4%', '7.1%', '6.9%', '8.0%'],
            'holding3_pct': ['5.4%', '6.9%', '8.2%', '7.6%', '6.1%', '7.3%', '5.8%', '8.7%', '6.5%', '7.9%'],
            'sector1': ['25', '20', '15', '30', '18', '22', '28', '12', '35', '16'],
            'sector1_name': ['Financial Services', 'Technology', 'Healthcare', 'Energy', 'Consumer', 'Industrial', 'Infrastructure', 'Pharma', 'Banking', 'Auto'],
            'sector2': ['20', '15', '25', '18', '22', '25', '20', '15', '25', '18'],
            'sector2_name': ['Technology', 'Healthcare', 'Financial Services', 'Consumer', 'Industrial', 'Energy', 'Auto', 'Pharma', 'Banking', 'Infrastructure'],
            'market': ['stock', 'commodity', 'currency', 'bond', 'real estate'],
            'trend': ['upward', 'downward', 'sideways', 'volatile', 'stable'],
            'index': ['Nifty', 'Sensex', 'Bank Nifty', 'Nifty Midcap', 'Nifty Smallcap'],
            'change': ['gained', 'lost', 'flattened', 'surged', 'declined'],
            'points': ['150', '120', '80', '200', '95'],
            'sentiment': ['bullish', 'bearish', 'cautious', 'optimistic', 'pessimistic'],
            'company': ['TCS', 'Infosys', 'HDFC Bank', 'Reliance Industries', 'ICICI Bank', 'HUL', 'ITC', 'SBI'],
            'performance': ['strong', 'weak', 'mixed', 'better', 'worse'],
            'metric': ['revenue', 'profit', 'EBITDA', 'net income', 'operating margin'],
            'value': ['â¹1,234 crore', 'â¹2,456 crore', 'â¹3,789 crore', 'â¹4,567 crore', 'â¹5,678 crore'],
            'comparison': ['exceeding', 'falling short of', 'matching', 'improving upon', 'declining from'],
            'industry': ['technology', 'banking', 'pharmaceutical', 'automotive', 'retail', 'telecom'],
            'outlook': ['positive', 'negative', 'neutral', 'optimistic', 'pessimistic'],
            'factor1': ['inflation', 'interest rates', 'GDP growth', 'consumer spending', 'corporate earnings'],
            'factor2': ['global markets', 'oil prices', 'currency fluctuations', 'policy changes', 'seasonal trends'],
            'regulator': ['RBI', 'SEBI', 'IRDA', 'TRAI', 'FMC'],
            'policy': ['monetary', 'regulatory', 'fiscal', 'trade', 'investment'],
            'goal': ['stabilizing markets', 'protecting investors', 'boosting growth', 'controlling inflation', 'ensuring compliance'],
            'impact': ['boost', 'support', 'hinder', 'disrupt', 'stabilize'],
            'affected': ['banking', 'technology', 'pharma', 'auto', 'real estate'],
            'indicator': ['inflation', 'GDP', 'industrial production', 'consumer confidence', 'manufacturing PMI'],
            'economic_trend': ['expansionary', 'contractionary', 'stable', 'volatile', 'recovering'],
            'central_bank': ['RBI', 'Federal Reserve', 'ECB', 'Bank of England', 'Bank of Japan'],
            'action': ['raise rates', 'cut rates', 'hold rates', 'inject liquidity', 'tighten policy'],
            'economic_condition': ['inflation', 'recession', 'growth', 'stagnation', 'recovery']
        }
    
    def generate_content_for_url(self, url: str) -> str:
        """Generate realistic content for a given URL"""
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path.lower()
        
        # Determine content type based on URL
        if 'groww.in' in domain and 'mutual-funds' in path:
            return self._generate_mutual_fund_content(url)
        elif any(site in domain for site in ['economictimes.indiatimes.com', 'livemint.com', 'business-standard.com']):
            return self._generate_financial_news_content(url)
        elif any(site in domain for site in ['nseindia.com', 'bseindia.com']):
            return self._generate_market_data_content(url)
        else:
            return self._generate_general_financial_content(url)
    
    def _generate_mutual_fund_content(self, url: str) -> str:
        """Generate mutual fund specific content"""
        # Extract fund name from URL
        fund_name = self._extract_fund_name_from_url(url)
        
        content_parts = []
        
        # Generate 3-5 paragraphs
        num_paragraphs = random.randint(3, 5)
        
        for i in range(num_paragraphs):
            template = random.choice(self.mutual_fund_templates)
            content = self._fill_template(template, 'mutual_fund')
            
            # Add fund name if it's the first paragraph
            if i == 0 and fund_name:
                content = f"{fund_name} - {content}"
            
            content_parts.append(content)
        
        return ' '.join(content_parts)
    
    def _generate_financial_news_content(self, url: str) -> str:
        """Generate financial news content"""
        content_parts = []
        
        # Generate 2-4 paragraphs
        num_paragraphs = random.randint(2, 4)
        
        for i in range(num_paragraphs):
            template = random.choice(self.news_templates)
            content = self._fill_template(template, 'news')
            content_parts.append(content)
        
        return ' '.join(content_parts)
    
    def _generate_market_data_content(self, url: str) -> str:
        """Generate market data content"""
        return self._generate_financial_news_content(url)  # Similar structure
    
    def _generate_general_financial_content(self, url: str) -> str:
        """Generate general financial content"""
        return self._generate_financial_news_content(url)  # Similar structure
    
    def _extract_fund_name_from_url(self, url: str) -> str:
        """Extract fund name from URL"""
        # Simple extraction based on URL pattern
        path_parts = url.split('/')
        for part in path_parts:
            if 'fund' in part.lower():
                # Convert URL-friendly name to readable name
                fund_name = part.replace('-', ' ').replace('_', ' ').title()
                return fund_name
        return ""
    
    def _fill_template(self, template: str, content_type: str) -> str:
        """Fill template with random data"""
        filled = template
        
        # Replace placeholders with random data
        placeholders = self._get_placeholders(template)
        
        for placeholder in placeholders:
            if placeholder in self.fund_data:
                filled = filled.replace(f'{{{placeholder}}}', random.choice(self.fund_data[placeholder]))
        
        return filled
    
    def _get_placeholders(self, template: str) -> List[str]:
        """Extract placeholders from template"""
        import re
        return re.findall(r'\{(\w+)\}', template)
    
    def generate_batch_content(self, urls: List[str]) -> List[str]:
        """Generate content for multiple URLs"""
        contents = []
        for url in urls:
            content = self.generate_content_for_url(url)
            contents.append(content)
        return contents
    
    def get_content_stats(self, content: str) -> Dict[str, Any]:
        """Get statistics about generated content"""
        return {
            'character_count': len(content),
            'word_count': len(content.split()),
            'sentence_count': len(content.split('. ')),
            'financial_terms': self._count_financial_terms(content),
            'numeric_data': self._count_numeric_data(content)
        }
    
    def _count_financial_terms(self, content: str) -> int:
        """Count financial terms in content"""
        financial_terms = ['nav', 'return', 'fund', 'aum', '%', 'cr', 'lakh', 'crore', 'rs', 'â¹â¹', 'equity', 'debt', 'hybrid', 'risk']
        content_lower = content.lower()
        return sum(1 for term in financial_terms if term in content_lower)
    
    def _count_numeric_data(self, content: str) -> int:
        """Count numeric data points in content"""
        import re
        patterns = [
            r'\b\d+\.?\d*%\b',  # Percentages
            r'\b\d+\.?\d*\s*(?:cr|l|crore|lakh)\b',  # Indian currency
            r'\b\d+\.?\d*\s*xirr\b',  # XIRR values
            r'\b\d+\.?\d*\s*years?\b',  # Time periods
        ]
        
        count = 0
        for pattern in patterns:
            matches = re.findall(pattern, content.lower())
            count += len(matches)
        
        return count
