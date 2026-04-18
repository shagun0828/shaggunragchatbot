"""
Chroma Cloud Manager for Phase 4.3
Handles integration with Chroma Cloud at trychroma.com
"""

import asyncio
import logging
import time
import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from collections import deque
import json

# Mock ChromaDB for demonstration (in production, use actual chromadb)
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logging.warning("ChromaDB not available, using mock implementation")


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


class ChromaCloudManager:
    """Manages Chroma Cloud integration for vector storage"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        
        # Chroma Cloud connection settings
        self.api_key = os.getenv('CHROMA_API_KEY', self.config.get('api_key', ''))
        self.tenant = self.config.get('tenant', 'default')
        self.database = self.config.get('database', 'mutual-funds-db')
        self.host = self.config.get('host', 'https://api.trychroma.com')
        
        # Metrics tracking
        self.metrics = ChromaCloudMetrics()
        self.upload_history = deque(maxlen=100)
        
        # Collections
        self.collections = {}
        self.client = None
        
        # Initialize client
        self._initialize_client()
        
        self.logger.info(f"Chroma Cloud manager initialized for database: {self.database}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'api_key': '',
            'tenant': 'default',
            'database': 'mutual-funds-db',
            'host': 'https://api.trychroma.com',
            'collections': {
                'mutual_funds': 'mutual_funds_v1',
                'financial_news': 'financial_news_v1',
                'market_data': 'market_data_v1'
            },
            'upload_settings': {
                'batch_size': 100,
                'max_retries': 3,
                'timeout': 30.0,
                'retry_delay': 1.0
            },
            'indexing': {
                'index_type': 'hnsw',
                'ef_construction': 200,
                'ef_search': 50,
                'm': 16
            }
        }
    
    def _initialize_client(self) -> None:
        """Initialize Chroma Cloud client"""
        try:
            if CHROMA_AVAILABLE and self.api_key:
                self.client = chromadb.HttpClient(
                    host=self.host,
                    settings=Settings(
                        chroma_auth="chromadb.auth.TokenAuth",
                        chroma_token=self.api_key,
                        chroma_tenant=self.tenant,
                        chroma_database=self.database
                    )
                )
                self.logger.info("Chroma Cloud client initialized successfully")
            else:
                self.logger.warning("Chroma Cloud client not available, using mock implementation")
                self.client = MockChromaClient()
        except Exception as e:
            self.logger.error(f"Failed to initialize Chroma Cloud client: {e}")
            self.client = MockChromaClient()
    
    async def upload_embeddings(self, embeddings: np.ndarray, 
                              documents: List[str], 
                              metadata: List[Dict[str, Any]],
                              collection_name: str = 'mutual_funds') -> str:
        """Upload embeddings to Chroma Cloud"""
        start_time = time.time()
        
        try:
            # Get or create collection
            collection = await self._get_or_create_collection(collection_name)
            
            # Prepare data for upload
            batch_size = self.config['upload_settings']['batch_size']
            
            # Upload in batches
            uploaded_count = 0
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                batch_metadata = metadata[i:i + batch_size]
                batch_ids = [f"doc_{i + j}" for j in range(len(batch_docs))]
                
                # Upload batch
                await self._upload_batch(
                    collection, batch_embeddings, batch_docs, batch_metadata, batch_ids
                )
                
                uploaded_count += len(batch_docs)
                self.metrics.batch_count += 1
            
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
    
    async def _get_or_create_collection(self, collection_name: str):
        """Get or create a collection in Chroma Cloud"""
        try:
            if collection_name in self.collections:
                return self.collections[collection_name]
            
            # Get collection name from config
            config_name = self.config['collections'].get(collection_name, collection_name)
            
            # Create collection
            collection = self.client.get_or_create_collection(
                name=config_name,
                metadata={
                    'created_at': time.time(),
                    'index_type': self.config['indexing']['index_type'],
                    'ef_construction': self.config['indexing']['ef_construction'],
                    'ef_search': self.config['indexing']['ef_search'],
                    'm': self.config['indexing']['m']
                }
            )
            
            self.collections[collection_name] = collection
            self.metrics.collections_created += 1
            
            self.logger.info(f"Collection '{config_name}' created/retrieved successfully")
            
            return collection
            
        except Exception as e:
            self.logger.error(f"Failed to get/create collection {collection_name}: {e}")
            raise
    
    async def _upload_batch(self, collection, embeddings: np.ndarray, 
                          documents: List[str], metadata: List[Dict[str, Any]], 
                          ids: List[str]) -> None:
        """Upload a batch of data to Chroma Cloud"""
        max_retries = self.config['upload_settings']['max_retries']
        retry_delay = self.config['upload_settings']['retry_delay']
        
        for attempt in range(max_retries):
            try:
                # Add documents to collection
                collection.add(
                    embeddings=embeddings.tolist(),
                    documents=documents,
                    metadatas=metadata,
                    ids=ids
                )
                
                return  # Success
                
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Upload attempt {attempt + 1} failed, retrying in {retry_delay}s: {e}")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    self.logger.error(f"Upload failed after {max_retries} attempts: {e}")
                    raise
    
    async def search_embeddings(self, query_embedding: np.ndarray, 
                              collection_name: str = 'mutual_funds',
                              top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar embeddings in Chroma Cloud"""
        try:
            collection = await self._get_or_create_collection(collection_name)
            
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
        try:
            collection = await self._get_or_create_collection(collection_name)
            
            # Get collection info
            stats = {
                'collection_name': collection_name,
                'collection_id': collection.id,
                'document_count': collection.count(),
                'metadata': collection.metadata,
                'created_at': collection.metadata.get('created_at', 0) if collection.metadata else 0
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get collection stats for {collection_name}: {e}")
            return {}
    
    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection from Chroma Cloud"""
        try:
            if collection_name in self.collections:
                collection = self.collections[collection_name]
                self.client.delete_collection(collection.id)
                del self.collections[collection_name]
                
                self.logger.info(f"Collection '{collection_name}' deleted successfully")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to delete collection {collection_name}: {e}")
            return False
    
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
            'collections': list(self.collections.keys()),
            'upload_history_size': len(self.upload_history),
            'configuration': {
                'database': self.database,
                'tenant': self.tenant,
                'host': self.host,
                'batch_size': self.config['upload_settings']['batch_size']
            }
        }
    
    def reset_metrics(self) -> None:
        """Reset all metrics"""
        self.metrics = ChromaCloudMetrics()
        self.upload_history.clear()
        self.logger.info("Chroma Cloud metrics reset")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on Chroma Cloud connection"""
        try:
            # Test connection by getting collections
            collections = self.client.list_collections()
            
            return {
                'status': 'healthy',
                'connection': 'active',
                'collections_count': len(collections),
                'database': self.database,
                'tenant': self.tenant,
                'timestamp': time.time()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'connection': 'failed',
                'error': str(e),
                'timestamp': time.time()
            }


class MockChromaClient:
    """Mock Chroma client for demonstration purposes"""
    
    def __init__(self):
        self.collections = {}
        self.logger = logging.getLogger(__name__)
    
    def get_or_create_collection(self, name: str, metadata: Dict[str, Any] = None):
        """Mock get or create collection"""
        if name not in self.collections:
            self.collections[name] = MockCollection(name, metadata or {})
        return self.collections[name]
    
    def list_collections(self):
        """Mock list collections"""
        return list(self.collections.values())
    
    def delete_collection(self, collection_id: str):
        """Mock delete collection"""
        # Find and delete collection by ID
        for name, collection in list(self.collections.items()):
            if collection.id == collection_id:
                del self.collections[name]
                break


class MockCollection:
    """Mock collection for demonstration"""
    
    def __init__(self, name: str, metadata: Dict[str, Any]):
        self.name = name
        self.id = f"collection_{name}_{hash(name)}"
        self.metadata = metadata
        self.documents = []
        self.embeddings = []
        self.metadatas = []
        self.ids = []
    
    def add(self, embeddings: List[List[float]], documents: List[str], 
            metadatas: List[Dict[str, Any]], ids: List[str]):
        """Mock add documents"""
        self.embeddings.extend(embeddings)
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)
    
    def query(self, query_embeddings: List[List[float]], n_results: int = 5):
        """Mock query"""
        # Return mock results
        return {
            'documents': [self.documents[:n_results]] if self.documents else [[]],
            'metadatas': [self.metadatas[:n_results]] if self.metadatas else [[]],
            'ids': [self.ids[:n_results]] if self.ids else [[]],
            'distances': [[0.1] * min(n_results, len(self.documents))] if self.documents else [[]]
        }
    
    def count(self) -> int:
        """Mock count"""
        return len(self.documents)
