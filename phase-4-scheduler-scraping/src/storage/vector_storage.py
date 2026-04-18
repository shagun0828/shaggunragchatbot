"""
Vector storage manager for mutual fund embeddings
Handles embedding generation and storage in vector database
"""

import asyncio
import logging
import os
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime

from embedders.sentence_transformer_embedder import SentenceTransformerEmbedder
from embedders.openai_embedder import OpenAIEmbedder
from embedders.financial_embedder import FinancialEmbedder
from models.chunk import Chunk


class VectorStorage:
    """Vector storage manager for mutual fund data"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.vector_db_url = os.getenv('VECTOR_DB_URL')
        self.vector_db_api_key = os.getenv('VECTOR_DB_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        # Initialize embedders
        self.sentence_transformer_embedder = SentenceTransformerEmbedder()
        self.financial_embedder = FinancialEmbedder()
        
        # Initialize OpenAI embedder if API key is available
        self.openai_embedder = None
        if self.openai_api_key:
            try:
                self.openai_embedder = OpenAIEmbedder()
            except Exception as e:
                self.logger.warning(f"Failed to initialize OpenAI embedder: {e}")
        
        # Vector database client (placeholder - would be actual implementation)
        self.vector_client = None
        
        if self.vector_db_url and self.vector_db_api_key:
            self._initialize_vector_client()
    
    def _initialize_vector_client(self):
        """Initialize vector database client"""
        try:
            # This would be actual implementation based on chosen vector DB
            # For now, we'll use a placeholder
            self.logger.info(f"Vector database client initialized for {self.vector_db_url}")
        except Exception as e:
            self.logger.error(f"Failed to initialize vector database client: {e}")
    
    async def store_embeddings(self, processed_data: List[Dict[str, Any]]) -> bool:
        """Generate and store embeddings for processed fund data"""
        self.logger.info(f"Generating embeddings for {len(processed_data)} funds")
        
        try:
            # Collect all chunks from all funds
            all_chunks = []
            for fund_data in processed_data:
                chunks = fund_data.get('chunks', [])
                all_chunks.extend(chunks)
            
            if not all_chunks:
                self.logger.warning("No chunks found to generate embeddings")
                return True
            
            # Generate embeddings
            embeddings = await self._generate_embeddings(all_chunks)
            
            # Store embeddings if vector client is available
            if self.vector_client:
                success = await self._store_in_vector_db(all_chunks, embeddings)
                if success:
                    self.logger.info(f"Successfully stored {len(all_chunks)} embeddings in vector database")
                else:
                    self.logger.error("Failed to store embeddings in vector database")
                    return False
            else:
                # Save embeddings locally
                await self._save_embeddings_locally(all_chunks, embeddings)
                self.logger.info(f"Saved {len(all_chunks)} embeddings locally")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing embeddings: {str(e)}")
            return False
    
    async def _generate_embeddings(self, chunks: List[Chunk]) -> np.ndarray:
        """Generate embeddings for chunks"""
        self.logger.info(f"Generating embeddings for {len(chunks)} chunks")
        
        # Choose embedder based on availability and configuration
        embedder = self._choose_embedder()
        
        # Generate embeddings
        embeddings = embedder.generate_embeddings(chunks)
        
        # Enhance for financial domain if using sentence transformers
        if isinstance(embedder, (SentenceTransformerEmbedder, FinancialEmbedder)):
            embeddings = self.financial_embedder.enhance_financial_embeddings(chunks, embeddings)
        
        # Validate embeddings
        await self._validate_embeddings(embeddings, chunks)
        
        return embeddings
    
    def _choose_embedder(self):
        """Choose the best available embedder"""
        # Prefer OpenAI if available and API key is set
        if self.openai_embedder and self.openai_api_key:
            self.logger.info("Using OpenAI embedder")
            return self.openai_embedder
        
        # Use sentence transformers as fallback
        self.logger.info("Using Sentence Transformer embedder")
        return self.sentence_transformer_embedder
    
    async def _validate_embeddings(self, embeddings: np.ndarray, chunks: List[Chunk]) -> None:
        """Validate generated embeddings"""
        if len(embeddings) != len(chunks):
            raise ValueError(f"Embeddings count ({len(embeddings)}) doesn't match chunks count ({len(chunks)})")
        
        # Check for NaN or infinite values
        if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
            raise ValueError("Embeddings contain NaN or infinite values")
        
        # Check embedding dimensions
        expected_dim = self._get_expected_dimension()
        if embeddings.shape[1] != expected_dim:
            self.logger.warning(f"Embedding dimension ({embeddings.shape[1]) differs from expected ({expected_dim})")
        
        self.logger.info(f"Validated {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
    
    def _get_expected_dimension(self) -> int:
        """Get expected embedding dimension based on available embedders"""
        if self.openai_embedder:
            return self.openai_embedder.dimension
        return self.sentence_transformer_embedder.dimension
    
    async def _store_in_vector_db(self, chunks: List[Chunk], embeddings: np.ndarray) -> bool:
        """Store embeddings in vector database"""
        try:
            # Prepare vectors for insertion
            vectors = self._prepare_vectors(chunks, embeddings)
            
            # Batch insert
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch_vectors = vectors[i:i + batch_size]
                
                # This would be actual vector DB insertion
                # For now, we'll simulate the insertion
                await self._simulate_vector_insertion(batch_vectors)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing in vector database: {str(e)}")
            return False
    
    async def _simulate_vector_insertion(self, vectors: List[Dict]) -> None:
        """Simulate vector database insertion (placeholder)"""
        await asyncio.sleep(0.1)  # Simulate network delay
        self.logger.debug(f"Simulated insertion of {len(vectors)} vectors")
    
    def _prepare_vectors(self, chunks: List[Chunk], embeddings: np.ndarray) -> List[Dict]:
        """Prepare vectors for database insertion"""
        vectors = []
        
        for chunk, embedding in zip(chunks, embeddings):
            vector = {
                'id': chunk.id,
                'values': embedding.tolist(),
                'metadata': {
                    'text': chunk.text,
                    'fund_name': chunk.metadata.get('fund_name'),
                    'section_type': chunk.metadata.get('section_type'),
                    'chunk_type': chunk.metadata.get('chunk_type'),
                    'created_at': datetime.utcnow().isoformat(),
                    'source': 'groww_mutual_funds'
                }
            }
            vectors.append(vector)
        
        return vectors
    
    async def _save_embeddings_locally(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        """Save embeddings locally when vector DB is not available"""
        from pathlib import Path
        import json
        
        embeddings_dir = Path("data/embeddings")
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save embeddings as numpy array
        embeddings_file = embeddings_dir / f"embeddings_{timestamp}.npy"
        np.save(embeddings_file, embeddings)
        
        # Save chunk metadata
        metadata_file = embeddings_dir / f"metadata_{timestamp}.json"
        metadata = []
        
        for chunk in chunks:
            metadata.append({
                'id': chunk.id,
                'text': chunk.text,
                'metadata': chunk.metadata
            })
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved embeddings locally to {embeddings_file}")
    
    async def search_similar_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks (placeholder implementation)"""
        if not self.vector_client:
            self.logger.warning("Vector database not available, returning empty results")
            return []
        
        try:
            # Generate query embedding
            embedder = self._choose_embedder()
            query_embedding = embedder.generate_embeddings([query])
            
            # Search in vector database (placeholder)
            # This would be actual vector search implementation
            results = await self._simulate_vector_search(query_embedding, top_k)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching similar chunks: {str(e)}")
            return []
    
    async def _simulate_vector_search(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Simulate vector search (placeholder)"""
        await asyncio.sleep(0.2)  # Simulate search delay
        
        # Return empty results for now
        # In real implementation, this would return actual search results
        return []
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        stats = {
            'vector_db_connected': bool(self.vector_client),
            'total_vectors': 0,
            'last_updated': None,
            'embedder_type': 'sentence_transformers' if not self.openai_embedder else 'openai'
        }
        
        # Get local embeddings stats if vector DB is not available
        if not self.vector_client:
            from pathlib import Path
            import glob
            
            embeddings_dir = Path("data/embeddings")
            if embeddings_dir.exists():
                embedding_files = glob.glob(str(embeddings_dir / "embeddings_*.npy"))
                stats['total_vectors'] = len(embedding_files)
                
                if embedding_files:
                    latest_file = max(embedding_files, key=lambda x: Path(x).stat().st_mtime)
                    stats['last_updated'] = Path(latest_file).stat().st_mtime
        
        return stats
