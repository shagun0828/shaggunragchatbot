"""
Mutual fund scraper for Groww.in URLs
Handles web scraping, data extraction, and initial validation
"""

import asyncio
import aiohttp
import logging
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
import json
from datetime import datetime

from utils.rate_limiter import RateLimiter
from utils.user_agents import UserAgentRotator


class MutualFundScraper:
    """Main scraper class for mutual fund data from Groww.in"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.rate_limiter = RateLimiter(requests_per_second=2)
        self.user_agent_rotator = UserAgentRotator()
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self._create_session()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def _create_session(self):
        """Create HTTP session with proper headers and timeout"""
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        headers = {
            'User-Agent': self.user_agent_rotator.get_random_user_agent(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            headers=headers,
            connector=aiohttp.TCPConnector(limit=10, limit_per_host=5)
        )
    
    async def scrape_all_funds(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Scrape data from all provided URLs"""
        if not self.session:
            await self._create_session()
            
        results = []
        
        async with self:
            tasks = [self.scrape_single_fund(url) for url in urls]
            scraped_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(scraped_results):
                if isinstance(result, Exception):
                    self.logger.error(f"Failed to scrape {urls[i]}: {str(result)}")
                    continue
                    
                if result:
                    results.append(result)
                    self.logger.info(f"Successfully scraped: {result.get('fund_name', 'Unknown')}")
        
        return results
    
    async def scrape_single_fund(self, url: str) -> Dict[str, Any]:
        """Scrape data from a single mutual fund URL"""
        try:
            await self.rate_limiter.wait()
            
            async with self.session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"HTTP {response.status}: {response.reason}")
                
                html_content = await response.text()
                return self._parse_fund_data(html_content, url)
                
        except Exception as e:
            self.logger.error(f"Error scraping {url}: {str(e)}")
            raise
    
    def _parse_fund_data(self, html_content: str, source_url: str) -> Dict[str, Any]:
        """Parse mutual fund data from HTML content"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract fund name from title or specific elements
        fund_name = self._extract_fund_name(soup)
        
        # Extract fund details
        fund_data = {
            'fund_name': fund_name,
            'source_url': source_url,
            'scraped_at': datetime.utcnow().isoformat(),
            'category': self._extract_category(soup),
            'risk_level': self._extract_risk_level(soup),
            'aum': self._extract_aum(soup),
            'expense_ratio': self._extract_expense_ratio(soup),
            'nav': self._extract_nav(soup),
            'returns': self._extract_returns(soup),
            'fund_manager': self._extract_fund_manager(soup),
            'inception_date': self._extract_inception_date(soup),
            'top_holdings': self._extract_top_holdings(soup),
            'sector_allocation': self._extract_sector_allocation(soup),
            'asset_allocation': self._extract_asset_allocation(soup),
            'description': self._extract_description(soup),
            'minimum_investment': self._extract_minimum_investment(soup),
            'exit_load': self._extract_exit_load(soup)
        }
        
        # Validate extracted data
        return self._validate_fund_data(fund_data)
    
    def _extract_fund_name(self, soup: BeautifulSoup) -> str:
        """Extract fund name from various possible locations"""
        selectors = [
            'h1[data-test-id="fundName"]',
            'h1.fund-name',
            'h1',
            '.fund-header h1',
            '[data-testid="fund-name"]'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text(strip=True)
        
        # Try to extract from title
        title = soup.find('title')
        if title:
            title_text = title.get_text(strip=True)
            # Remove common suffixes
            title_text = re.sub(r'\s*-\s*Groww.*$', '', title_text, flags=re.IGNORECASE)
            title_text = re.sub(r'\s*\|\s*.*$', '', title_text)
            return title_text
        
        return "Unknown Fund"
    
    def _extract_category(self, soup: BeautifulSoup) -> str:
        """Extract fund category"""
        selectors = [
            '[data-test-id="fundCategory"]',
            '.fund-category',
            '.category',
            '[data-testid="fund-category"]'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text(strip=True)
        
        return ""
    
    def _extract_risk_level(self, soup: BeautifulSoup) -> str:
        """Extract risk level"""
        selectors = [
            '[data-test-id="riskLevel"]',
            '.risk-level',
            '.risk-meter',
            '[data-testid="risk-level"]'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text(strip=True)
        
        return ""
    
    def _extract_aum(self, soup: BeautifulSoup) -> str:
        """Extract Assets Under Management"""
        selectors = [
            '[data-test-id="aum"]',
            '.aum-value',
            '.assets-under-management',
            '[data-testid="aum-value"]'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text(strip=True)
        
        return ""
    
    def _extract_expense_ratio(self, soup: BeautifulSoup) -> str:
        """Extract expense ratio"""
        selectors = [
            '[data-test-id="expenseRatio"]',
            '.expense-ratio',
            '.expense-ratio-value',
            '[data-testid="expense-ratio"]'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                text = element.get_text(strip=True)
                # Extract percentage
                match = re.search(r'(\d+\.?\d*)\s*%?', text)
                if match:
                    return match.group(1) + '%'
        
        return ""
    
    def _extract_nav(self, soup: BeautifulSoup) -> str:
        """Extract Net Asset Value"""
        selectors = [
            '[data-test-id="nav"]',
            '.nav-value',
            '.current-nav',
            '[data-testid="nav-value"]'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text(strip=True)
        
        return ""
    
    def _extract_returns(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract returns data for different periods"""
        returns = {}
        
        # Look for return periods in various formats
        return_patterns = {
            '1_year': [r'1\s*year\s*[:\s]+(\d+\.?\d*)\s*%?', r'1Y\s*[:\s]+(\d+\.?\d*)\s*%?'],
            '3_year': [r'3\s*years?\s*[:\s]+(\d+\.?\d*)\s*%?', r'3Y\s*[:\s]+(\d+\.?\d*)\s*%?'],
            '5_year': [r'5\s*years?\s*[:\s]+(\d+\.?\d*)\s*%?', r'5Y\s*[:\s]+(\d+\.?\d*)\s*%?'],
            'since_inception': [r'since\s*inception\s*[:\s]+(\d+\.?\d*)\s*%?', r'SI\s*[:\s]+(\d+\.?\d*)\s*%?']
        }
        
        page_text = soup.get_text()
        
        for period, patterns in return_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, page_text, re.IGNORECASE)
                if match:
                    returns[period] = match.group(1) + '%'
                    break
        
        return returns
    
    def _extract_fund_manager(self, soup: BeautifulSoup) -> str:
        """Extract fund manager name"""
        selectors = [
            '[data-test-id="fundManager"]',
            '.fund-manager',
            '.manager-name',
            '[data-testid="fund-manager"]'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text(strip=True)
        
        return ""
    
    def _extract_inception_date(self, soup: BeautifulSoup) -> str:
        """Extract fund inception date"""
        selectors = [
            '[data-test-id="inceptionDate"]',
            '.inception-date',
            '.launch-date',
            '[data-testid="inception-date"]'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text(strip=True)
        
        return ""
    
    def _extract_top_holdings(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract top holdings"""
        holdings = []
        
        # Look for holdings table
        holdings_table = soup.find('table', {'class': re.compile(r'holding', re.I)})
        if holdings_table:
            rows = holdings_table.find_all('tr')[1:]  # Skip header
            for row in rows[:10]:  # Top 10 holdings
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    holdings.append({
                        'name': cells[0].get_text(strip=True),
                        'percentage': cells[1].get_text(strip=True)
                    })
        
        return holdings
    
    def _extract_sector_allocation(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract sector allocation"""
        sectors = []
        
        # Look for sector allocation section
        sector_section = soup.find('div', {'class': re.compile(r'sector', re.I)})
        if sector_section:
            # Try to extract sector data
            sector_items = sector_section.find_all(['div', 'tr'])
            for item in sector_items[:10]:
                text = item.get_text(strip=True)
                # Look for percentage patterns
                match = re.search(r'(.+?)\s+(\d+\.?\d*)\s*%?', text)
                if match:
                    sectors.append({
                        'sector': match.group(1).strip(),
                        'allocation': match.group(2) + '%'
                    })
        
        return sectors
    
    def _extract_asset_allocation(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract asset allocation"""
        allocation = {}
        
        # Look for asset allocation data
        page_text = soup.get_text()
        
        # Common asset allocation patterns
        patterns = {
            'equity': r'equity\s*[:\s]+(\d+\.?\d*)\s*%?',
            'debt': r'debt\s*[:\s]+(\d+\.?\d*)\s*%?',
            'cash': r'cash\s*[:\s]+(\d+\.?\d*)\s*%?'
        }
        
        for asset_type, pattern in patterns.items():
            match = re.search(pattern, page_text, re.IGNORECASE)
            if match:
                allocation[asset_type] = match.group(1) + '%'
        
        return allocation
    
    def _extract_description(self, soup: BeautifulSoup) -> str:
        """Extract fund description"""
        selectors = [
            '[data-test-id="fundDescription"]',
            '.fund-description',
            '.description',
            '[data-testid="description"]'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text(strip=True)
        
        return ""
    
    def _extract_minimum_investment(self, soup: BeautifulSoup) -> str:
        """Extract minimum investment amount"""
        selectors = [
            '[data-test-id="minimumInvestment"]',
            '.minimum-investment',
            '.min-investment',
            '[data-testid="minimum-investment"]'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text(strip=True)
        
        return ""
    
    def _extract_exit_load(self, soup: BeautifulSoup) -> str:
        """Extract exit load information"""
        selectors = [
            '[data-test-id="exitLoad"]',
            '.exit-load',
            '.exit-load-info',
            '[data-testid="exit-load"]'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text(strip=True)
        
        return ""
    
    def _validate_fund_data(self, fund_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean extracted fund data"""
        # Ensure required fields exist
        if not fund_data.get('fund_name'):
            fund_data['fund_name'] = "Unknown Fund"
        
        # Clean numeric fields
        numeric_fields = ['aum', 'expense_ratio', 'nav']
        for field in numeric_fields:
            if fund_data.get(field):
                fund_data[field] = self._clean_numeric_field(fund_data[field])
        
        # Validate returns data
        returns = fund_data.get('returns', {})
        if isinstance(returns, dict):
            for period, value in returns.items():
                returns[period] = self._clean_numeric_field(value)
        
        # Add validation status
        fund_data['validation_status'] = 'validated' if fund_data.get('fund_name') != "Unknown Fund" else 'partial'
        
        return fund_data
    
    def _clean_numeric_field(self, value: str) -> str:
        """Clean numeric field by removing extra characters"""
        if not value:
            return ""
        
        # Remove currency symbols, commas, extra spaces
        cleaned = re.sub(r'[^\d.%]', '', str(value))
        
        # Ensure percentage format
        if '%' not in cleaned and cleaned.replace('.', '').isdigit():
            # Check if this looks like a percentage
            try:
                num_value = float(cleaned)
                if 0 <= num_value <= 100:
                    cleaned = f"{num_value}%"
            except ValueError:
                pass
        
        return cleaned
