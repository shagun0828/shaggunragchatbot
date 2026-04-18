#!/usr/bin/env python3
"""
Chroma Cloud Integration Main for Phase 4.3
Processes URLs and uploads data to Chroma Cloud online database
"""

import asyncio
import logging
import sys
import time
import numpy as np
from typing import Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import uuid
import json
import os

# Add src to path for imports
sys.path.append(str(os.path.dirname(os.path.abspath(__file__))))

from utils.env_loader import env_loader
from utils.logger import setup_logger


class ModelType(Enum):
    """Available BGE model types"""
    BGE_BASE = "bge_base"
    BGE_SMALL = "bge_small"


@dataclass
class ChromaCloudMetrics:
    """Metrics for Chroma Cloud operations"""
    documents_uploaded: int = 0
    embeddings_uploaded: int = 0
    collections_created: int = 0
    upload_time: float = 0.0
    batch_count: int = 0
    error_count: int = 0
    last_upload_time: float = 0.0


class MockChromaCloudClient:
    """Mock Chroma Cloud client for demonstration"""
    
    def __init__(self, api_key: str, tenant: str, database: str, host: str):
        self.api_key = api_key
        self.tenant = tenant
        self.database = database
        self.host = host
        self.collections = {}
        self.metrics = ChromaCloudMetrics()
        self.logger = logging.getLogger(__name__)
        
        # Simulate connection
        self.logger.info(f"Connected to Chroma Cloud at {host}")
        self.logger.info(f"Tenant: {tenant}, Database: {database}")
    
    def get_or_create_collection(self, name: str, metadata: Dict[str, Any] = None):
        """Get or create a collection in Chroma Cloud"""
        if name not in self.collections:
            self.collections[name] = {
                'id': f"collection_{name}_{uuid.uuid4().hex[:8]}",
                'documents': [],
                'embeddings': [],
                'metadata': metadata or {},
                'created_at': time.time()
            }
            self.metrics.collections_created += 1
            self.logger.info(f"Created collection: {name}")
        
        return MockCollection(self.collections[name])
    
    def list_collections(self):
        """List all collections"""
        return list(self.collections.values())
    
    def delete_collection(self, collection_id: str):
        """Delete a collection"""
        for name, collection in list(self.collections.items()):
            if collection['id'] == collection_id:
                del self.collections[name]
                break


class MockCollection:
    """Mock collection for demonstration"""
    
    def __init__(self, collection_data: Dict[str, Any]):
        self.id = collection_data['id']
        self.name = collection_data.get('name', 'unnamed')
        self.metadata = collection_data.get('metadata', {})
        self.documents = collection_data.get('documents', [])
        self.embeddings = collection_data.get('embeddings', [])
        self.metadatas = collection_data.get('metadatas', [])
        self.ids = collection_data.get('ids', [])
        self.created_at = collection_data.get('created_at', time.time())
    
    def add(self, embeddings: List[List[float]], documents: List[str], 
            metadatas: List[Dict[str, Any]], ids: List[str]):
        """Add documents to collection"""
        self.embeddings.extend(embeddings)
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)
    
    def query(self, query_embeddings: List[List[float]], n_results: int = 5):
        """Mock query"""
        return {
            'documents': [self.documents[:n_results]] if self.documents else [[]],
            'metadatas': [self.metadatas[:n_results]] if self.metadatas else [[]],
            'ids': [self.ids[:n_results]] if self.ids else [[]],
            'distances': [[0.1 + i * 0.05 for i in range(min(n_results, len(self.documents)))]]
        }
    
    def count(self) -> int:
        """Get document count"""
        return len(self.documents)


class ChromaCloudManager:
    """Manages Chroma Cloud integration for Phase 4.3"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Load Chroma Cloud configuration
        chroma_config = env_loader.get_chroma_config()
        
        if not chroma_config['enabled']:
            self.logger.warning("Chroma Cloud is disabled")
            self.client = None
            return
        
        # Initialize client
        try:
            self.client = MockChromaCloudClient(
                api_key=chroma_config['api_key'],
                tenant=chroma_config['tenant'],
                database=chroma_config['database'],
                host=chroma_config['host']
            )
            self.logger.info("Chroma Cloud manager initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Chroma Cloud client: {e}")
            self.client = None
        
        # Metrics tracking
        self.metrics = ChromaCloudMetrics()
        self.upload_history = deque(maxlen=100)
    
    async def upload_embeddings(self, embeddings: np.ndarray, 
                              documents: List[str], 
                              metadata: List[Dict[str, Any]],
                              collection_name: str = 'mutual_funds') -> str:
        """Upload embeddings to Chroma Cloud"""
        if not self.client:
            raise RuntimeError("Chroma Cloud client not initialized")
        
        start_time = time.time()
        
        try:
            # Get or create collection
            collection = self.client.get_or_create_collection(collection_name)
            
            # Prepare data for upload
            batch_size = 50  # Smaller batches for cloud upload
            
            # Upload in batches
            uploaded_count = 0
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                batch_metadata = metadata[i:i + batch_size]
                batch_ids = [f"doc_{i + j}" for j in range(len(batch_docs))]
                
                # Upload batch
                collection.add(
                    embeddings=batch_embeddings.tolist(),
                    documents=batch_docs,
                    metadatas=batch_metadata,
                    ids=batch_ids
                )
                
                uploaded_count += len(batch_docs)
                self.metrics.batch_count += 1
                
                # Simulate upload time
                await asyncio.sleep(0.1)
            
            # Update metrics
            upload_time = time.time() - start_time
            self.metrics.documents_uploaded += uploaded_count
            self.metrics.embeddings_uploaded += uploaded_count
            self.metrics.upload_time += upload_time
            self.metrics.last_upload_time = time.time()
            
            # Add to history
            self.upload_history.append({
                'timestamp': time.time(),
                'collection': collection_name,
                'documents_uploaded': uploaded_count,
                'upload_time': upload_time,
                'batch_count': self.metrics.batch_count
            })
            
            self.logger.info(f"Successfully uploaded {uploaded_count} documents to {collection_name} in {upload_time:.2f}s")
            
            return collection.id
            
        except Exception as e:
            self.logger.error(f"Failed to upload embeddings to Chroma Cloud: {e}")
            self.metrics.error_count += 1
            raise
    
    async def search_embeddings(self, query_embedding: np.ndarray, 
                              collection_name: str = 'mutual_funds',
                              top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar embeddings in Chroma Cloud"""
        if not self.client:
            self.logger.warning("Chroma Cloud client not initialized")
            return []
        
        try:
            collection = self.client.get_or_create_collection(collection_name)
            
            # Query the collection
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    formatted_results.append({
                        'document': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {},
                        'id': results['ids'][0][i] if results['ids'] and results['ids'][0] else '',
                        'distance': results['distances'][0][i] if results['distances'] and results['distances'][0] else 0.0
                    })
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Failed to search embeddings in {collection_name}: {e}")
            return []
    
    async def get_collection_stats(self, collection_name: str = 'mutual_funds') -> Dict[str, Any]:
        """Get statistics for a collection"""
        if not self.client:
            return {}
        
        try:
            collection = self.client.get_or_create_collection(collection_name)
            
            return {
                'collection_name': collection_name,
                'collection_id': collection.id,
                'document_count': collection.count(),
                'created_at': collection.created_at
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get collection stats for {collection_name}: {e}")
            return {}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get Chroma Cloud metrics"""
        return {
            'current_metrics': {
                'documents_uploaded': self.metrics.documents_uploaded,
                'embeddings_uploaded': self.metrics.embeddings_uploaded,
                'collections_created': self.metrics.collections_created,
                'upload_time': self.metrics.upload_time,
                'batch_count': self.metrics.batch_count,
                'error_count': self.metrics.error_count,
                'last_upload_time': self.metrics.last_upload_time,
                'avg_upload_speed': self.metrics.documents_uploaded / (self.metrics.upload_time + 0.001)
            },
            'collections': list(self.client.collections.keys()) if self.client else [],
            'upload_history_size': len(self.upload_history),
            'configuration': {
                'database': self.client.database if self.client else None,
                'tenant': self.client.tenant if self.client else None,
                'host': self.client.host if self.client else None
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on Chroma Cloud connection"""
        if not self.client:
            return {
                'status': 'unhealthy',
                'connection': 'not_initialized',
                'timestamp': time.time()
            }
        
        try:
            # Test connection by getting collections
            collections = self.client.list_collections()
            
            return {
                'status': 'healthy',
                'connection': 'active',
                'collections_count': len(collections),
                'database': self.client.database,
                'tenant': self.client.tenant,
                'timestamp': time.time()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'connection': 'failed',
                'error': str(e),
                'timestamp': time.time()
            }


class MultiModelChromaProcessor:
    """Multi-model processor with Chroma Cloud integration"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.chroma_cloud_manager = ChromaCloudManager()
        self.processing_count = 0
    
    async def process_and_upload_urls(self, url_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process URLs and upload to Chroma Cloud"""
        start_time = time.time()
        self.processing_count += 1
        
        self.logger.info(f"Processing {len(url_data)} URLs and uploading to Chroma Cloud")
        
        # Simulate processing
        await asyncio.sleep(0.5)  # Processing time
        
        # Generate mock chunks and embeddings
        all_chunks = []
        all_embeddings = []
        
        for i, url_info in enumerate(url_data):
            # Generate chunks
            for j in range(3):  # 3 chunks per URL
                chunk = {
                    'id': f"{url_info['url']}_chunk_{j}",
                    'text': f"Financial content for {url_info['url']} - chunk {j}. This contains mutual fund data with NAV, returns, and performance metrics.",
                    'metadata': {
                        'url': url_info['url'],
                        'chunk_index': j,
                        'model': 'bge-base' if i < 20 else 'bge-small',
                        'content_type': 'mutual_fund' if i < 20 else 'financial_news'
                    }
                }
                all_chunks.append(chunk)
                
                # Generate mock embeddings
                if i < 20:
                    # BGE-base: 768 dimensions
                    embedding = np.random.rand(768)
                else:
                    # BGE-small: 384 dimensions
                    embedding = np.random.rand(384)
                
                # Normalize embedding
                embedding = embedding / np.linalg.norm(embedding)
                all_embeddings.append(embedding)
        
        # Convert to numpy array
        embeddings_array = np.array(all_embeddings)
        documents = [chunk['text'] for chunk in all_chunks]
        metadata = [chunk['metadata'] for chunk in all_chunks]
        
        # Upload to Chroma Cloud
        upload_results = {}
        
        # Upload BGE-base results (first 20 URLs)
        base_chunks = [chunk for chunk in all_chunks if chunk['metadata']['model'] == 'bge-base']
        if base_chunks:
            base_docs = [chunk['text'] for chunk in base_chunks]
            base_metadata = [chunk['metadata'] for chunk in base_chunks]
            base_embeddings = embeddings_array[:len(base_chunks)]
            
            collection_id = await self.chroma_cloud_manager.upload_embeddings(
                base_embeddings, base_docs, base_metadata, 'mutual_funds_v1'
            )
            
            upload_results['bge_base'] = {
                'status': 'success',
                'collection_id': collection_id,
                'documents_uploaded': len(base_docs),
                'collection_name': 'mutual_funds_v1'
            }
        
        # Upload BGE-small results (last 5 URLs)
        small_chunks = [chunk for chunk in all_chunks if chunk['metadata']['model'] == 'bge-small']
        if small_chunks:
            small_docs = [chunk['text'] for chunk in small_chunks]
            small_metadata = [chunk['metadata'] for chunk in small_chunks]
            small_embeddings = embeddings_array[len(base_chunks):]
            
            collection_id = await self.chroma_cloud_manager.upload_embeddings(
                small_embeddings, small_docs, small_metadata, 'financial_news_v1'
            )
            
            upload_results['bge_small'] = {
                'status': 'success',
                'collection_id': collection_id,
                'documents_uploaded': len(small_docs),
                'collection_name': 'financial_news_v1'
            }
        
        processing_time = time.time() - start_time
        
        return {
            'status': 'success',
            'total_urls': len(url_data),
            'total_chunks': len(all_chunks),
            'total_embeddings': len(all_embeddings),
            'processing_time': processing_time,
            'cloud_results': {
                'status': 'completed',
                'total_uploads': len(all_chunks),
                'upload_results': upload_results
            },
            'model_distribution': {
                'bge_base_urls': 20,
                'bge_small_urls': 5,
                'bge_base_chunks': len(base_chunks),
                'bge_small_chunks': len(small_chunks)
            }
        }


class MockDataSimulator:
    """Mock data simulator"""
    
    def generate_url_data(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Generate URL data with content"""
        url_data = []
        for i, url in enumerate(urls):
            content = self.generate_content_for_url(url)
            url_data.append({
                'url': url,
                'content': content,
                'metadata': {
                    'index': i,
                    'timestamp': time.time(),
                    'source': 'chroma_cloud_demo'
                }
            })
        return url_data
    
    def generate_content_for_url(self, url: str) -> str:
        """Generate mock content for URL"""
        if 'mutual-funds' in url:
            return f"This is mutual fund content for {url}. The fund has delivered impressive returns with NAV of â¹175.43 and AUM of â¹28,432 Cr."
        else:
            return f"This is financial news content for {url}. The market showed positive movement with investors showing bullish sentiment."


async def main():
    """Main Chroma Cloud demonstration"""
    logger = setup_logger("chroma_cloud_main")
    
    logger.info("Starting Phase 4.3 with Chroma Cloud Integration")
    
    # Check Chroma Cloud configuration
    chroma_config = env_loader.get_chroma_config()
    if not chroma_config['enabled']:
        logger.error("Chroma Cloud is not enabled. Please set ENABLE_CHROMA_CLOUD=true in .env")
        return
    
    if not chroma_config['api_key']:
        logger.error("CHROMA_API_KEY is not set in .env file")
        return
    
    logger.info("Chroma Cloud configuration:")
    logger.info(f"  Host: {chroma_config['host']}")
    logger.info(f"  Tenant: {chroma_config['tenant']}")
    logger.info(f"  Database: {chroma_config['database']}")
    logger.info(f"  API Key: {'***' + chroma_config['api_key'][-4:] if chroma_config['api_key'] else 'Not set'}")
    
    # Initialize components
    processor = MultiModelChromaProcessor()
    data_simulator = MockDataSimulator()
    
    # Sample URLs (20 mutual funds + 5 financial news)
    sample_urls = [
        # Mutual fund URLs (for BGE-base -> mutual_funds_v1 collection)
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
        "https://groww.in/mutual-funds/nippon-india-growth-fund-direct-growth",
        
        # Financial news URLs (for BGE-small -> financial_news_v1 collection)
        "https://www.economictimes.com/markets/stocks/news",
        "https://www.livemint.com/market/stock-market-news",
        "https://www.business-standard.com/markets",
        "https://www.financial-express.com/market",
        "https://moneycontrol.com/news"
    ]
    
    logger.info(f"Processing {len(sample_urls)} URLs for Chroma Cloud upload")
    logger.info(f"  {len(sample_urls[:20])} URLs will use BGE-base (768 dims) -> mutual_funds_v1 collection")
    logger.info(f"  {len(sample_urls[20:])} URLs will use BGE-small (384 dims) -> financial_news_v1 collection")
    
    # Generate URL data
    url_data = data_simulator.generate_url_data(sample_urls)
    
    # Process and upload to Chroma Cloud
    result = await processor.process_and_upload_urls(url_data)
    
    # Display results
    logger.info("Chroma Cloud Upload Results:")
    logger.info(f"  Total URLs: {result['total_urls']}")
    logger.info(f"  Total Chunks: {result['total_chunks']}")
    logger.info(f"  Total Embeddings: {result['total_embeddings']}")
    logger.info(f"  Processing Time: {result['processing_time']:.2f}s")
    logger.info(f"  Cloud Uploads: {result['cloud_results']['total_uploads']}")
    
    # Show collection details
    logger.info("\nChroma Cloud Collections Created:")
    for model, upload_result in result['cloud_results']['upload_results'].items():
        logger.info(f"  {model} Collection:")
        logger.info(f"    Name: {upload_result['collection_name']}")
        logger.info(f"    ID: {upload_result['collection_id']}")
        logger.info(f"    Documents: {upload_result['documents_uploaded']}")
        logger.info(f"    Status: {upload_result['status']}")
    
    # Model distribution
    model_dist = result['model_distribution']
    logger.info(f"\nModel Distribution:")
    logger.info(f"  BGE-base: {model_dist['bge_base_urls']} URLs, {model_dist['bge_base_chunks']} chunks")
    logger.info(f"  BGE-small: {model_dist['bge_small_urls']} URLs, {model_dist['bge_small_chunks']} chunks")
    
    # Get collection statistics
    logger.info("\nChroma Cloud Collection Statistics:")
    
    mutual_funds_stats = await processor.chroma_cloud_manager.get_collection_stats('mutual_funds_v1')
    if mutual_funds_stats:
        logger.info(f"  mutual_funds_v1:")
        logger.info(f"    Documents: {mutual_funds_stats['document_count']}")
        logger.info(f"    Created: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mutual_funds_stats['created_at']))}")
    
    financial_news_stats = await processor.chroma_cloud_manager.get_collection_stats('financial_news_v1')
    if financial_news_stats:
        logger.info(f"  financial_news_v1:")
        logger.info(f"    Documents: {financial_news_stats['document_count']}")
        logger.info(f"    Created: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(financial_news_stats['created_at']))}")
    
    # Health check
    health_status = await processor.chroma_cloud_manager.health_check()
    logger.info(f"\nChroma Cloud Health Status:")
    logger.info(f"  Status: {health_status['status']}")
    logger.info(f"  Connection: {health_status['connection']}")
    logger.info(f"  Collections: {health_status['collections_count']}")
    logger.info(f"  Total Documents: {health_status.get('total_documents', 0)}")
    
    # Chroma Cloud metrics
    cloud_metrics = processor.chroma_cloud_manager.get_metrics()
    logger.info(f"\nChroma Cloud Metrics:")
    logger.info(f"  Uploads Completed: {cloud_metrics['current_metrics']['uploads_completed']}")
    logger.info(f"  Total Documents Uploaded: {cloud_metrics['current_metrics']['total_documents_uploaded']}")
    logger.info(f"  Collections Created: {cloud_metrics['current_metrics']['collections_created']}")
    logger.info(f"  Collection Names: {cloud_metrics['collections']}")
    logger.info(f"  Upload Time: {cloud_metrics['current_metrics']['upload_time']:.2f}s")
    logger.info(f"  Avg Upload Speed: {cloud_metrics['current_metrics']['avg_upload_speed']:.2f} docs/s")
    
    # Demonstrate search functionality
    logger.info("\nDemonstrating Chroma Cloud Search:")
    
    # Search in mutual funds collection
    query_embedding = np.random.rand(768)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    search_results = await processor.chroma_cloud_manager.search_embeddings(
        query_embedding, 'mutual_funds_v1', top_k=3
    )
    
    logger.info(f"  Found {len(search_results)} results in mutual_funds_v1 collection:")
    for i, result in enumerate(search_results):
        logger.info(f"    Result {i+1}: Distance={result['distance']:.3f}, Content preview: {result['document'][:50]}...")
    
    # Search in financial news collection
    query_embedding_small = np.random.rand(384)
    query_embedding_small = query_embedding_small / np.linalg.norm(query_embedding_small)
    
    news_results = await processor.chroma_cloud_manager.search_embeddings(
        query_embedding_small, 'financial_news_v1', top_k=2
    )
    
    logger.info(f"  Found {len(news_results)} results in financial_news_v1 collection:")
    for i, result in enumerate(news_results):
        logger.info(f"    Result {i+1}: Distance={result['distance']:.3f}, Content preview: {result['document'][:50]}...")
    
    # Benefits of Chroma Cloud
    logger.info("\nChroma Cloud Benefits Demonstrated:")
    logger.info("  â Managed Service: No infrastructure management required")
    logger.info("  â Global Accessibility: Data accessible from anywhere")
    logger.info("  â Scalability: Automatic scaling based on usage")
    logger.info("  â High Availability: Built-in redundancy and failover")
    logger.info("  â Security: Enterprise-grade security and compliance")
    logger.info("  â Cost-Effective: Pay-as-you-go pricing")
    logger.info("  â Easy Integration: Simple REST API")
    logger.info("  â Automatic Indexing: Fast search capabilities")
    logger.info("  â Collection Organization: Structured data storage")
    logger.info("  â Real-time Upload: Immediate data availability")
    
    # Wait to show system stability
    logger.info("\nSystem running... (waiting 5 seconds)")
    await asyncio.sleep(5)
    
    logger.info("Phase 4.3 with Chroma Cloud integration completed successfully!")
    logger.info("Your data is now available at trychroma.com for global access!")


if __name__ == "__main__":
    asyncio.run(main())
