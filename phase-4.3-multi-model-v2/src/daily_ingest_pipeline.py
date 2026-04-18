#!/usr/bin/env python3
"""
Daily Ingest Pipeline for Phase 4.3
Complete end-to-end pipeline: Scraping -> Chunking -> Embedding -> Chroma Cloud Upload
Triggered by GitHub Actions daily at 9:15 AM IST
"""

import asyncio
import logging
import sys
import time
import os
import json
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import aiohttp
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from simple_chroma_cloud import MultiModelChromaProcessor, MockDataSimulator
from utils.env_loader import env_loader


class PipelineStatus(Enum):
    """Pipeline execution status"""
    INITIALIZING = "initializing"
    SCRAPING = "scraping"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    UPLOADING = "uploading"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PipelineMetrics:
    """Comprehensive pipeline metrics"""
    start_time: float = 0.0
    end_time: float = 0.0
    total_duration: float = 0.0
    
    # Scraping metrics
    urls_scraped: int = 0
    scraping_time: float = 0.0
    scraping_errors: int = 0
    
    # Chunking metrics
    chunks_created: int = 0
    chunking_time: float = 0.0
    avg_chunk_size: float = 0.0
    
    # Embedding metrics
    embeddings_created: int = 0
    embedding_time: float = 0.0
    bge_base_embeddings: int = 0
    bge_small_embeddings: int = 0
    
    # Upload metrics
    documents_uploaded: int = 0
    upload_time: float = 0.0
    collections_updated: int = 0
    upload_errors: int = 0
    
    # Overall metrics
    total_urls_processed: int = 0
    pipeline_efficiency: float = 0.0
    error_count: int = 0


@dataclass
class PipelineConfig:
    """Pipeline configuration"""
    dry_run: bool = False
    test_mode: bool = False
    force_run: bool = False
    run_date: str = ""
    run_id: str = ""
    
    # Scraping configuration
    max_urls_per_source: int = 25
    scraping_timeout: int = 30
    retry_attempts: int = 3
    
    # Processing configuration
    chunk_size_target: int = 800
    max_chunk_size: int = 2000
    min_chunk_size: int = 100
    
    # Upload configuration
    batch_size: int = 50
    upload_timeout: int = 300


class FinancialDataScraper:
    """Advanced financial data scraper with multiple sources"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = self._setup_logger()
        self.session = None
        
        # Financial data sources
        self.mutual_fund_sources = [
            "https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth",
            "https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth",
            "https://groww.in/mutual-funds/hdfc-focused-fund-direct-growth",
            "https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth",
            "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
            "https://groww.in/mutual-funds/icici-prudential-technology-fund-direct-growth",
            "https://groww.in/mutual-funds/axis-bluechip-fund-direct-growth",
            "https://groww.in/mutual-funds/sbi-small-cap-fund-direct-plan-growth",
            "https://groww.in/mutual-funds/uti-nifty-index-fund-direct-growth",
            "https://groww.in/mutual-funds/mirae-asset-large-cap-fund-direct-growth",
            "https://groww.in/mutual-funds/tata-digital-india-fund-direct-growth",
            "https://groww.in/mutual-funds/kotak-emerging-equity-fund-direct-growth",
            "https://groww.in/mutual-funds/franklin-india-prima-fund-direct-growth",
            "https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth",
            "https://groww.in/mutual-funds/icici-prudential-balanced-advantage-fund-direct-growth",
            "https://groww.in/mutual-funds/axis-hybrid-equity-fund-direct-growth",
            "https://groww.in/mutual-funds/sbi-balanced-fund-direct-growth",
            "https://groww.in/mutual-funds/hdfc-arbitrage-fund-direct-growth",
            "https://groww.in/mutual-funds/icici-prudential-equity-arbitrage-fund-direct-growth",
            "https://groww.in/mutual-funds/nippon-india-growth-fund-direct-growth"
        ]
        
        self.financial_news_sources = [
            "https://www.economictimes.com/markets/stocks/news",
            "https://www.livemint.com/market/stock-market-news",
            "https://www.business-standard.com/markets",
            "https://www.financial-express.com/market",
            "https://moneycontrol.com/news"
        ]
    
    def _setup_logger(self):
        """Setup logger"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def _get_session(self):
        """Get HTTP session"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=self.config.scraping_timeout)
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            self.session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        return self.session
    
    async def scrape_url(self, url: str) -> Dict[str, Any]:
        """Scrape content from a single URL"""
        session = await self._get_session()
        
        for attempt in range(self.config.retry_attempts):
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        
                        # Extract basic information
                        title = self._extract_title(content)
                        date = self._extract_date(content)
                        
                        return {
                            'url': url,
                            'content': content,
                            'title': title,
                            'date': date,
                            'status': 'success',
                            'scraped_at': time.time(),
                            'content_length': len(content),
                            'source': self._get_source_type(url)
                        }
                    else:
                        self.logger.warning(f"HTTP {response.status} for {url}")
                        
            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return {
            'url': url,
            'content': '',
            'title': '',
            'date': '',
            'status': 'failed',
            'scraped_at': time.time(),
            'error': f"Failed after {self.config.retry_attempts} attempts",
            'source': self._get_source_type(url)
        }
    
    def _extract_title(self, content: str) -> str:
        """Extract title from HTML content"""
        import re
        
        # Try to extract title from title tag
        title_match = re.search(r'<title[^>]*>(.*?)</title>', content, re.IGNORECASE | re.DOTALL)
        if title_match:
            title = title_match.group(1).strip()
            # Clean up title
            title = re.sub(r'\s+', ' ', title)
            return title
        
        # Fallback to h1 tag
        h1_match = re.search(r'<h1[^>]*>(.*?)</h1>', content, re.IGNORECASE | re.DOTALL)
        if h1_match:
            return h1_match.group(1).strip()
        
        return "Unknown Title"
    
    def _extract_date(self, content: str) -> str:
        """Extract date from content"""
        import re
        from datetime import datetime
        
        # Try to find date in various formats
        date_patterns = [
            r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b',
            r'\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b',
            r'\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\b'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return datetime.now().strftime('%Y-%m-%d')
    
    def _get_source_type(self, url: str) -> str:
        """Determine source type from URL"""
        if 'mutual-funds' in url:
            return 'mutual_fund'
        elif 'economictimes' in url or 'livemint' in url or 'business-standard' in url:
            return 'financial_news'
        elif 'moneycontrol' in url:
            return 'market_data'
        else:
            return 'other'
    
    async def scrape_all_sources(self) -> List[Dict[str, Any]]:
        """Scrape all configured sources"""
        self.logger.info("Starting financial data scraping")
        
        all_sources = self.mutual_fund_sources + self.financial_news_sources
        
        if self.config.test_mode:
            # Limit to 3 sources for testing
            all_sources = all_sources[:3]
            self.logger.info("Test mode: Limited to 3 sources")
        
        self.logger.info(f"Scraping {len(all_sources)} URLs")
        
        # Scrape URLs concurrently
        tasks = []
        for url in all_sources:
            task = asyncio.create_task(self.scrape_url(url))
            tasks.append(task)
        
        # Wait for all scraping to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        scraped_data = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Scraping error: {result}")
                continue
            
            if result['status'] == 'success':
                scraped_data.append(result)
            else:
                self.logger.warning(f"Failed to scrape {result['url']}: {result.get('error', 'Unknown error')}")
        
        self.logger.info(f"Successfully scraped {len(scraped_data)} out of {len(all_sources)} URLs")
        
        return scraped_data
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()


class AdvancedChunker:
    """Advanced chunking system for financial content"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """Setup logger"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def create_chunks(self, scraped_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create chunks from scraped data"""
        self.logger.info(f"Creating chunks from {len(scraped_data)} documents")
        
        all_chunks = []
        
        for doc in scraped_data:
            if doc['status'] != 'success':
                continue
            
            # Clean content
            cleaned_content = self._clean_content(doc['content'])
            
            # Create chunks based on content type
            if doc['source'] == 'mutual_fund':
                chunks = self._chunk_mutual_fund_content(cleaned_content, doc)
            else:
                chunks = self._chunk_news_content(cleaned_content, doc)
            
            all_chunks.extend(chunks)
        
        self.logger.info(f"Created {len(all_chunks)} chunks")
        
        # Calculate average chunk size
        if all_chunks:
            avg_size = sum(len(chunk['text']) for chunk in all_chunks) / len(all_chunks)
            self.logger.info(f"Average chunk size: {avg_size:.1f} characters")
        
        return all_chunks
    
    def _clean_content(self, content: str) -> str:
        """Clean HTML content"""
        import re
        
        # Remove HTML tags
        content = re.sub(r'<[^>]+>', ' ', content)
        
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove special characters
        content = re.sub(r'[^\w\s.,;:!?%â¹â¹-]', ' ', content)
        
        # Clean up
        content = content.strip()
        
        return content
    
    def _chunk_mutual_fund_content(self, content: str, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create chunks for mutual fund content"""
        chunks = []
        
        # Split by paragraphs first
        paragraphs = content.split('\n')
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Check if adding this paragraph exceeds target size
            if len(current_chunk) + len(para) > self.config.chunk_size_target and current_chunk:
                # Create chunk
                chunk = self._create_chunk(current_chunk, doc, len(chunks))
                chunks.append(chunk)
                current_chunk = para
            else:
                current_chunk += " " + para if current_chunk else para
        
        # Add remaining content
        if current_chunk:
            chunk = self._create_chunk(current_chunk, doc, len(chunks))
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_news_content(self, content: str, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create chunks for news content"""
        chunks = []
        
        # Split by sentences for news content
        sentences = content.split('. ')
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if adding this sentence exceeds target size
            if len(current_chunk) + len(sentence) > self.config.chunk_size_target and current_chunk:
                # Create chunk
                chunk = self._create_chunk(current_chunk, doc, len(chunks))
                chunks.append(chunk)
                current_chunk = sentence
            else:
                current_chunk += ". " + sentence if current_chunk else sentence
        
        # Add remaining content
        if current_chunk:
            chunk = self._create_chunk(current_chunk, doc, len(chunks))
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(self, text: str, doc: Dict[str, Any], chunk_index: int) -> Dict[str, Any]:
        """Create a chunk object"""
        chunk_id = f"{doc['url']}_chunk_{chunk_index}_{int(time.time())}"
        
        return {
            'id': chunk_id,
            'text': text,
            'metadata': {
                'source_url': doc['url'],
                'source_title': doc['title'],
                'source_date': doc['date'],
                'source_type': doc['source'],
                'chunk_index': chunk_index,
                'created_at': time.time(),
                'text_length': len(text),
                'word_count': len(text.split()),
                'run_date': os.getenv('RUN_DATE', datetime.now().strftime('%Y-%m-%d')),
                'run_id': os.getenv('RUN_ID', 'unknown')
            }
        }


class DailyIngestPipeline:
    """Complete daily ingest pipeline orchestrator"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.metrics = PipelineMetrics()
        self.status = PipelineStatus.INITIALIZING
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize components
        self.scraper = FinancialDataScraper(self.config)
        self.chunker = AdvancedChunker(self.config)
        self.chroma_processor = MultiModelChromaProcessor()
        
        # Create directories
        self._create_directories()
    
    def _setup_logger(self):
        """Setup logger"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_config(self) -> PipelineConfig:
        """Load pipeline configuration"""
        return PipelineConfig(
            dry_run=os.getenv('DRY_RUN', 'false').lower() == 'true',
            test_mode=os.getenv('TEST_MODE', 'false').lower() == 'true',
            force_run=os.getenv('FORCE_RUN', 'false').lower() == 'true',
            run_date=os.getenv('RUN_DATE', datetime.now().strftime('%Y-%m-%d')),
            run_id=os.getenv('RUN_ID', 'unknown')
        )
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = ['logs', 'reports', 'temp']
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
    
    async def run(self) -> Dict[str, Any]:
        """Run the complete ingest pipeline"""
        self.metrics.start_time = time.time()
        
        try:
            self.logger.info("Starting Daily Ingest Pipeline")
            self.logger.info(f"Configuration: Dry run={self.config.dry_run}, Test mode={self.config.test_mode}")
            self.logger.info(f"Run date: {self.config.run_date}, Run ID: {self.config.run_id}")
            
            # Step 1: Scraping
            self.status = PipelineStatus.SCRAPING
            scraped_data = await self._run_scraping()
            
            # Step 2: Chunking
            self.status = PipelineStatus.CHUNKING
            chunks = self._run_chunking(scraped_data)
            
            # Step 3: Embedding + Upload
            self.status = PipelineStatus.EMBEDDING
            upload_results = await self._run_embedding_and_upload(chunks)
            
            # Step 4: Completion
            self.status = PipelineStatus.COMPLETED
            final_results = self._generate_final_results(scraped_data, chunks, upload_results)
            
            self.logger.info("Daily Ingest Pipeline completed successfully")
            
            return final_results
            
        except Exception as e:
            self.status = PipelineStatus.FAILED
            self.metrics.error_count += 1
            self.logger.error(f"Pipeline failed: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return error results
            return {
                'status': 'failed',
                'error': str(e),
                'traceback': traceback.format_exc(),
                'metrics': self._get_metrics_dict(),
                'run_date': self.config.run_date,
                'run_id': self.config.run_id
            }
        
        finally:
            self.metrics.end_time = time.time()
            self.metrics.total_duration = self.metrics.end_time - self.metrics.start_time
            
            # Cleanup
            await self._cleanup()
    
    async def _run_scraping(self) -> List[Dict[str, Any]]:
        """Run scraping step"""
        self.logger.info("Starting scraping step")
        start_time = time.time()
        
        try:
            scraped_data = await self.scraper.scrape_all_sources()
            
            self.metrics.urls_scraped = len(scraped_data)
            self.metrics.scraping_time = time.time() - start_time
            self.metrics.total_urls_processed = len(scraped_data)
            
            # Count errors
            self.metrics.scraping_errors = sum(1 for doc in scraped_data if doc.get('status') != 'success')
            
            self.logger.info(f"Scraping completed: {len(scraped_data)} URLs scraped in {self.metrics.scraping_time:.2f}s")
            
            return scraped_data
            
        except Exception as e:
            self.logger.error(f"Scraping failed: {e}")
            raise
    
    def _run_chunking(self, scraped_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run chunking step"""
        self.logger.info("Starting chunking step")
        start_time = time.time()
        
        try:
            chunks = self.chunker.create_chunks(scraped_data)
            
            self.metrics.chunks_created = len(chunks)
            self.metrics.chunking_time = time.time() - start_time
            
            # Calculate average chunk size
            if chunks:
                self.metrics.avg_chunk_size = sum(len(chunk['text']) for chunk in chunks) / len(chunks)
            
            self.logger.info(f"Chunking completed: {len(chunks)} chunks created in {self.metrics.chunking_time:.2f}s")
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Chunking failed: {e}")
            raise
    
    async def _run_embedding_and_upload(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run embedding and upload step"""
        self.logger.info("Starting embedding and upload step")
        start_time = time.time()
        
        try:
            # Convert chunks to URL data format for processor
            url_data = []
            for chunk in chunks:
                url_data.append({
                    'url': chunk['metadata']['source_url'],
                    'content': chunk['text'],
                    'metadata': chunk['metadata']
                })
            
            # Skip upload if dry run
            if self.config.dry_run:
                self.logger.info("DRY RUN: Skipping Chroma Cloud upload")
                return {
                    'status': 'skipped',
                    'reason': 'dry_run',
                    'documents_processed': len(url_data),
                    'chunks_processed': len(chunks)
                }
            
            # Process and upload
            upload_results = await self.chroma_processor.process_and_upload_urls(url_data)
            
            self.metrics.documents_uploaded = upload_results['cloud_results']['total_uploads']
            self.metrics.upload_time = time.time() - start_time
            self.metrics.collections_updated = len(upload_results['cloud_results']['upload_results'])
            
            # Count embeddings by model
            self.metrics.bge_base_embeddings = upload_results['model_distribution']['bge_base_chunks']
            self.metrics.bge_small_embeddings = upload_results['model_distribution']['bge_small_chunks']
            self.metrics.embeddings_created = self.metrics.bge_base_embeddings + self.metrics.bge_small_embeddings
            
            self.logger.info(f"Embedding and upload completed: {self.metrics.documents_uploaded} documents uploaded in {self.metrics.upload_time:.2f}s")
            
            return upload_results
            
        except Exception as e:
            self.logger.error(f"Embedding and upload failed: {e}")
            raise
    
    def _generate_final_results(self, scraped_data: List[Dict[str, Any]], 
                              chunks: List[Dict[str, Any]], 
                              upload_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final results summary"""
        
        # Calculate pipeline efficiency
        total_processing_time = self.metrics.scraping_time + self.metrics.chunking_time + self.metrics.upload_time
        self.metrics.pipeline_efficiency = len(scraped_data) / (total_processing_time + 0.001)
        
        return {
            'status': 'completed',
            'run_date': self.config.run_date,
            'run_id': self.config.run_id,
            'configuration': {
                'dry_run': self.config.dry_run,
                'test_mode': self.config.test_mode,
                'force_run': self.config.force_run
            },
            'scraping_results': {
                'urls_attempted': len(self.scraper.mutual_fund_sources) + len(self.scraper.financial_news_sources),
                'urls_scraped': self.metrics.urls_scraped,
                'scraping_time': self.metrics.scraping_time,
                'scraping_errors': self.metrics.scraping_errors
            },
            'chunking_results': {
                'chunks_created': self.metrics.chunks_created,
                'chunking_time': self.metrics.chunking_time,
                'avg_chunk_size': self.metrics.avg_chunk_size
            },
            'upload_results': upload_results,
            'metrics': self._get_metrics_dict(),
            'timestamp': time.time()
        }
    
    def _get_metrics_dict(self) -> Dict[str, Any]:
        """Get metrics as dictionary"""
        return {
            'start_time': self.metrics.start_time,
            'end_time': self.metrics.end_time,
            'total_duration': self.metrics.total_duration,
            'urls_scraped': self.metrics.urls_scraped,
            'scraping_time': self.metrics.scraping_time,
            'scraping_errors': self.metrics.scraping_errors,
            'chunks_created': self.metrics.chunks_created,
            'chunking_time': self.metrics.chunking_time,
            'avg_chunk_size': self.metrics.avg_chunk_size,
            'embeddings_created': self.metrics.embeddings_created,
            'embedding_time': self.metrics.upload_time,
            'bge_base_embeddings': self.metrics.bge_base_embeddings,
            'bge_small_embeddings': self.metrics.bge_small_embeddings,
            'documents_uploaded': self.metrics.documents_uploaded,
            'upload_time': self.metrics.upload_time,
            'collections_updated': self.metrics.collections_updated,
            'upload_errors': self.metrics.upload_errors,
            'total_urls_processed': self.metrics.total_urls_processed,
            'pipeline_efficiency': self.metrics.pipeline_efficiency,
            'error_count': self.metrics.error_count
        }
    
    async def _cleanup(self):
        """Cleanup resources"""
        try:
            await self.scraper.close()
        except Exception as e:
            self.logger.warning(f"Cleanup error: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status"""
        return {
            'status': self.status.value,
            'metrics': self._get_metrics_dict(),
            'configuration': {
                'dry_run': self.config.dry_run,
                'test_mode': self.config.test_mode,
                'run_date': self.config.run_date,
                'run_id': self.config.run_id
            }
        }


async def main():
    """Main entry point"""
    print("Daily Ingest Pipeline for Phase 4.3")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = DailyIngestPipeline()
    
    try:
        # Run pipeline
        results = await pipeline.run()
        
        # Display results
        print(f"\nPipeline Status: {results['status']}")
        print(f"Run Date: {results['run_date']}")
        print(f"Run ID: {results['run_id']}")
        
        if results['status'] == 'completed':
            print(f"\nScraping Results:")
            print(f"  URLs Scraped: {results['scraping_results']['urls_scraped']}")
            print(f"  Scraping Time: {results['scraping_results']['scraping_time']:.2f}s")
            print(f"  Scraping Errors: {results['scraping_results']['scraping_errors']}")
            
            print(f"\nChunking Results:")
            print(f"  Chunks Created: {results['chunking_results']['chunks_created']}")
            print(f"  Chunking Time: {results['chunking_results']['chunking_time']:.2f}s")
            print(f"  Avg Chunk Size: {results['chunking_results']['avg_chunk_size']:.1f} chars")
            
            print(f"\nUpload Results:")
            print(f"  Documents Uploaded: {results['upload_results']['cloud_results']['total_uploads']}")
            print(f"  Upload Time: {results['upload_results']['processing_time']:.2f}s")
            print(f"  Collections Updated: {len(results['upload_results']['cloud_results']['upload_results'])}")
            
            print(f"\nOverall Metrics:")
            print(f"  Total Duration: {results['metrics']['total_duration']:.2f}s")
            print(f"  Pipeline Efficiency: {results['metrics']['pipeline_efficiency']:.2f} URLs/s")
            print(f"  BGE-base Embeddings: {results['metrics']['bge_base_embeddings']}")
            print(f"  BGE-small Embeddings: {results['metrics']['bge_small_embeddings']}")
            
            # Save results to file
            results_file = f"reports/daily_ingest_{pipeline.config.run_date}.json"
            Path('reports').mkdir(exist_ok=True)
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"\nResults saved to: {results_file}")
            
        elif results['status'] == 'failed':
            print(f"Pipeline failed: {results['error']}")
            print(f"Error details: {results.get('traceback', 'No traceback available')}")
        
        elif results['status'] == 'skipped':
            print(f"Pipeline skipped: {results.get('reason', 'Unknown reason')}")
        
        print("\n" + "=" * 50)
        print("Daily Ingest Pipeline completed")
        
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        pipeline.status = PipelineStatus.FAILED
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        pipeline.status = PipelineStatus.FAILED
        raise


if __name__ == "__main__":
    asyncio.run(main())
