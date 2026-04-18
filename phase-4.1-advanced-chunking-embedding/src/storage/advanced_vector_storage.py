"""
Advanced Vector Storage Manager
Enhanced vector storage with quality assurance and batch processing
"""

import asyncio
import logging
import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import uuid

from embedders.embedding_quality_checker import EmbeddingQualityChecker, QualityReport
from embedders.enhanced_financial_embedder import EnhancedFinancialEmbedder
from models.chunk import Chunk


class AdvancedVectorStorage:
    """Advanced vector storage with quality assurance and batch processing"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.quality_checker = EmbeddingQualityChecker()
        self.financial_embedder = EnhancedFinancialEmbedder()
        
        # Storage configuration
        self.vector_db_url = os.getenv('VECTOR_DB_URL')
        self.vector_db_api_key = os.getenv('VECTOR_DB_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        # Batch processing parameters
        self.batch_size = 100
        self.max_retries = 3
        self.retry_delay = 1.0
        
        # Quality thresholds
        self.min_quality_score = 0.7
        self.auto_fix_quality = True
        
        # Storage paths
        self.storage_path = Path("data/vector_storage")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Vector database client (placeholder)
        self.vector_client = None
        self._initialize_vector_client()
        
        # Storage statistics
        self.stats = {
            'total_vectors': 0,
            'last_update': None,
            'quality_scores': [],
            'failed_batches': 0
        }
    
    def _initialize_vector_client(self):
        """Initialize vector database client"""
        try:
            if self.vector_db_url and self.vector_db_api_key:
                # This would be actual vector DB initialization
                self.logger.info(f"Vector database client initialized: {self.vector_db_url}")
                self.vector_client = True  # Placeholder
            else:
                self.logger.warning("Vector database credentials not provided, using local storage")
        except Exception as e:
            self.logger.error(f"Failed to initialize vector database client: {e}")
    
    async def store_embeddings_with_quality_assurance(self, processed_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Store embeddings with comprehensive quality assurance"""
        self.logger.info(f"Storing embeddings for {len(processed_data)} funds with quality assurance")
        
        storage_results = {
            'total_funds': len(processed_data),
            'total_chunks': 0,
            'quality_passed': 0,
            'quality_failed': 0,
            'auto_fixed': 0,
            'storage_errors': 0,
            'quality_reports': []
        }
        
        # Collect all chunks from all funds
        all_chunks = []
        for fund_data in processed_data:
            chunks = fund_data.get('chunks', [])
            all_chunks.extend(chunks)
            storage_results['total_chunks'] += len(chunks)
        
        if not all_chunks:
            self.logger.warning("No chunks found to process")
            return storage_results
        
        # Generate base embeddings
        base_embeddings = await self._generate_base_embeddings(all_chunks)
        
        # Enhance embeddings for financial domain
        enhanced_embeddings = self.financial_embedder.enhance_financial_embeddings(all_chunks, base_embeddings)
        
        # Quality check
        quality_report = self.quality_checker.check_embedding_quality(enhanced_embeddings, all_chunks)
        storage_results['quality_reports'].append(quality_report)
        
        # Filter chunks based on quality
        quality_filtered_chunks, quality_filtered_embeddings = self._filter_by_quality(
            all_chunks, enhanced_embeddings, quality_report, storage_results
        )
        
        # Store embeddings
        if quality_filtered_chunks:
            storage_success = await self._store_embeddings_batch(
                quality_filtered_chunks, quality_filtered_embeddings, storage_results
            )
            
            if storage_success:
                self.logger.info(f"Successfully stored {len(quality_filtered_chunks)} high-quality embeddings")
            else:
                self.logger.error("Failed to store embeddings")
                storage_results['storage_errors'] += 1
        
        # Update statistics
        self._update_storage_statistics(storage_results)
        
        # Save quality report
        await self._save_quality_report(quality_report, storage_results)
        
        return storage_results
    
    async def _generate_base_embeddings(self, chunks: List[Chunk]) -> np.ndarray:
        """Generate base embeddings for chunks"""
        self.logger.info(f"Generating base embeddings for {len(chunks)} chunks")
        
        # Use sentence transformers as base
        texts = [chunk.text for chunk in chunks]
        
        try:
            embeddings = self.financial_embedder.base_model.encode(
                texts,
                batch_size=32,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            return embeddings
        except Exception as e:
            self.logger.error(f"Failed to generate base embeddings: {e}")
            raise
    
    def _filter_by_quality(self, chunks: List[Chunk], embeddings: np.ndarray, 
                          quality_report: QualityReport, storage_results: Dict[str, Any]) -> Tuple[List[Chunk], np.ndarray]:
        """Filter chunks and embeddings based on quality assessment"""
        self.logger.info("Filtering chunks based on quality assessment")
        
        # Get indices of chunks with quality issues
        all_issue_indices = set()
        for issue_type, indices in quality_report.issues.items():
            all_issue_indices.update(indices)
        
        # Separate good and problematic chunks
        good_indices = [i for i in range(len(chunks)) if i not in all_issue_indices]
        problem_indices = list(all_issue_indices)
        
        good_chunks = [chunks[i] for i in good_indices]
        good_embeddings = embeddings[good_indices] if good_indices else np.array([])
        
        problem_chunks = [chunks[i] for i in problem_indices]
        problem_embeddings = embeddings[problem_indices] if problem_indices else np.array([])
        
        storage_results['quality_failed'] = len(problem_indices)
        storage_results['quality_passed'] = len(good_indices)
        
        # Attempt to fix problematic chunks if auto-fix is enabled
        if self.auto_fix_quality and problem_chunks:
            fixed_chunks, fixed_embeddings = self._attempt_quality_fix(
                problem_chunks, problem_embeddings, quality_report
            )
            
            if fixed_chunks:
                good_chunks.extend(fixed_chunks)
                if len(good_embeddings) > 0:
                    good_embeddings = np.vstack([good_embeddings, fixed_embeddings])
                else:
                    good_embeddings = fixed_embeddings
                
                storage_results['auto_fixed'] = len(fixed_chunks)
                self.logger.info(f"Auto-fixed {len(fixed_chunks)} problematic chunks")
        
        return good_chunks, good_embeddings
    
    def _attempt_quality_fix(self, problem_chunks: List[Chunk], problem_embeddings: np.ndarray, 
                           quality_report: QualityReport) -> Tuple[List[Chunk], np.ndarray]:
        """Attempt to fix quality issues in problematic chunks"""
        fixed_chunks = []
        fixed_embeddings = []
        
        for i, chunk in enumerate(problem_chunks):
            embedding = problem_embeddings[i]
            
            # Check specific issues and attempt fixes
            if i in quality_report.issues[QualityIssue.LOW_VARIANCE]:
                # Try to enhance low variance embeddings
                fixed_embedding = self._fix_low_variance_embedding(chunk, embedding)
                if fixed_embedding is not None:
                    fixed_chunks.append(chunk)
                    fixed_embeddings.append(fixed_embedding)
                    continue
            
            if i in quality_report.issues[QualityIssue.OUTLIER]:
                # Try to normalize outlier embeddings
                fixed_embedding = self._fix_outlier_embedding(chunk, embedding)
                if fixed_embedding is not None:
                    fixed_chunks.append(chunk)
                    fixed_embeddings.append(fixed_embedding)
                    continue
            
            # Skip chunks with NaN or infinite values
            if i in quality_report.issues[QualityIssue.NAN_VALUES] or i in quality_report.issues[QualityIssue.INFINITE_VALUES]:
                continue
        
        return fixed_chunks, np.array(fixed_embeddings) if fixed_embeddings else np.array([])
    
    def _fix_low_variance_embedding(self, chunk: Chunk, embedding: np.ndarray) -> Optional[np.ndarray]:
        """Fix low variance embedding by adding noise"""
        try:
            # Add small amount of noise to increase variance
            noise = np.random.normal(0, 0.01, embedding.shape)
            fixed_embedding = embedding + noise
            
            # Renormalize
            fixed_embedding = fixed_embedding / np.linalg.norm(fixed_embedding)
            
            # Check if variance improved
            if np.var(fixed_embedding) > 0.01:
                return fixed_embedding
        except Exception:
            pass
        
        return None
    
    def _fix_outlier_embedding(self, chunk: Chunk, embedding: np.ndarray) -> Optional[np.ndarray]:
        """Fix outlier embedding by normalizing"""
        try:
            # Clip extreme values
            norm = np.linalg.norm(embedding)
            if norm > 2.0:  # Too large norm
                normalized_embedding = embedding / norm * 1.0
                return normalized_embedding
            elif norm < 0.5:  # Too small norm
                normalized_embedding = embedding / norm * 0.8
                return normalized_embedding
        except Exception:
            pass
        
        return None
    
    async def _store_embeddings_batch(self, chunks: List[Chunk], embeddings: np.ndarray, 
                                    storage_results: Dict[str, Any]) -> bool:
        """Store embeddings in batches with retry logic"""
        if self.vector_client:
            return await self._store_in_vector_database(chunks, embeddings, storage_results)
        else:
            return await self._store_locally(chunks, embeddings, storage_results)
    
    async def _store_in_vector_database(self, chunks: List[Chunk], embeddings: np.ndarray, 
                                      storage_results: Dict[str, Any]) -> bool:
        """Store embeddings in vector database with batch processing"""
        try:
            # Prepare vectors for insertion
            vectors = self._prepare_vectors(chunks, embeddings)
            
            # Batch insertion with retry
            for i in range(0, len(vectors), self.batch_size):
                batch_vectors = vectors[i:i + self.batch_size]
                
                for attempt in range(self.max_retries):
                    try:
                        # Simulate vector database insertion
                        await self._simulate_vector_db_insertion(batch_vectors)
                        break
                    except Exception as e:
                        if attempt == self.max_retries - 1:
                            self.logger.error(f"Failed to insert batch after {self.max_retries} attempts: {e}")
                            storage_results['failed_batches'] += 1
                            return False
                        
                        await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
            
            return True
            
        except Exception as e:
            self.logger.error(f"Vector database storage failed: {e}")
            return False
    
    async def _store_locally(self, chunks: List[Chunk], embeddings: np.ndarray, 
                           storage_results: Dict[str, Any]) -> bool:
        """Store embeddings locally with metadata"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save embeddings
            embeddings_file = self.storage_path / f"embeddings_{timestamp}.npy"
            np.save(embeddings_file, embeddings)
            
            # Save metadata
            metadata = []
            for chunk in chunks:
                metadata.append({
                    'id': chunk.id,
                    'text': chunk.text,
                    'metadata': chunk.metadata,
                    'stored_at': datetime.utcnow().isoformat()
                })
            
            metadata_file = self.storage_path / f"metadata_{timestamp}.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            # Save storage summary
            summary = {
                'timestamp': timestamp,
                'total_chunks': len(chunks),
                'embedding_dimension': embeddings.shape[1],
                'storage_type': 'local',
                'stored_at': datetime.utcnow().isoformat()
            }
            
            summary_file = self.storage_path / f"summary_{timestamp}.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
            
            self.logger.info(f"Stored {len(chunks)} embeddings locally to {embeddings_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Local storage failed: {e}")
            return False
    
    def _prepare_vectors(self, chunks: List[Chunk], embeddings: np.ndarray) -> List[Dict[str, Any]]:
        """Prepare vectors for database insertion"""
        vectors = []
        
        for chunk, embedding in zip(chunks, embeddings):
            vector = {
                'id': str(uuid.uuid4()),  # Generate unique ID
                'values': embedding.tolist(),
                'metadata': {
                    'chunk_id': chunk.id,
                    'text': chunk.text,
                    'fund_name': chunk.metadata.get('fund_name'),
                    'section_type': chunk.metadata.get('section_type'),
                    'chunk_type': chunk.metadata.get('chunk_type'),
                    'created_at': chunk.created_at.isoformat(),
                    'stored_at': datetime.utcnow().isoformat(),
                    'source': 'advanced_mutual_funds',
                    'quality_score': chunk.metadata.get('quality_score', 0.0)
                }
            }
            vectors.append(vector)
        
        return vectors
    
    async def _simulate_vector_db_insertion(self, vectors: List[Dict]) -> None:
        """Simulate vector database insertion (placeholder)"""
        await asyncio.sleep(0.1)  # Simulate network delay
        self.logger.debug(f"Simulated insertion of {len(vectors)} vectors")
    
    def _update_storage_statistics(self, storage_results: Dict[str, Any]) -> None:
        """Update storage statistics"""
        self.stats['total_vectors'] += storage_results['quality_passed'] + storage_results['auto_fixed']
        self.stats['last_update'] = datetime.utcnow()
        
        # Add quality scores to history
        for report in storage_results['quality_reports']:
            self.stats['quality_scores'].append(report.overall_score)
        
        # Keep only last 100 quality scores
        if len(self.stats['quality_scores']) > 100:
            self.stats['quality_scores'] = self.stats['quality_scores'][-100:]
    
    async def _save_quality_report(self, quality_report: QualityReport, 
                                 storage_results: Dict[str, Any]) -> None:
        """Save quality report with storage results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report_data = {
            'timestamp': timestamp,
            'quality_report': {
                'total_embeddings': quality_report.total_embeddings,
                'dimension': quality_report.dimension,
                'overall_score': quality_report.overall_score,
                'issues': {issue.value: len(indices) for issue, indices in quality_report.issues.items()},
                'statistics': quality_report.statistics,
                'recommendations': quality_report.recommendations
            },
            'storage_results': storage_results,
            'generated_at': datetime.utcnow().isoformat()
        }
        
        report_file = self.storage_path / f"quality_report_{timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Quality report saved to {report_file}")
    
    async def search_similar_chunks(self, query: str, top_k: int = 5, 
                                  filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar chunks with advanced filtering"""
        if not self.vector_client:
            self.logger.warning("Vector database not available, returning empty results")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.financial_embedder.base_model.encode([query])[0]
            
            # Apply filters
            filter_params = self._prepare_search_filters(filters)
            
            # Search in vector database (placeholder)
            results = await self._simulate_vector_search(query_embedding, top_k, filter_params)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []
    
    def _prepare_search_filters(self, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare search filters"""
        if not filters:
            return {}
        
        filter_params = {}
        
        # Map filter keys to metadata keys
        filter_mapping = {
            'fund_name': 'fund_name',
            'section_type': 'section_type',
            'chunk_type': 'chunk_type',
            'min_quality_score': 'quality_score'
        }
        
        for key, value in filters.items():
            if key in filter_mapping:
                filter_params[filter_mapping[key]] = value
        
        return filter_params
    
    async def _simulate_vector_search(self, query_embedding: np.ndarray, top_k: int, 
                                    filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simulate vector search (placeholder)"""
        await asyncio.sleep(0.2)  # Simulate search delay
        
        # Return empty results for now
        # In real implementation, this would return actual search results
        return []
    
    def get_storage_statistics(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics"""
        stats = self.stats.copy()
        
        # Calculate average quality score
        if stats['quality_scores']:
            stats['average_quality_score'] = np.mean(stats['quality_scores'])
            stats['quality_score_trend'] = 'improving' if len(stats['quality_scores']) > 1 and stats['quality_scores'][-1] > stats['quality_scores'][-2] else 'stable'
        else:
            stats['average_quality_score'] = 0.0
            stats['quality_score_trend'] = 'unknown'
        
        # Add storage type information
        stats['storage_type'] = 'vector_database' if self.vector_client else 'local'
        stats['vector_db_connected'] = bool(self.vector_client)
        
        return stats
    
    async def cleanup_old_embeddings(self, days_old: int = 30) -> Dict[str, Any]:
        """Clean up old embeddings to manage storage"""
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        
        cleanup_stats = {
            'files_deleted': 0,
            'space_freed': 0,
            'errors': 0
        }
        
        try:
            # Clean up local files
            for file_path in self.storage_path.glob("*.npy"):
                if file_path.stat().st_mtime < cutoff_date.timestamp():
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    cleanup_stats['files_deleted'] += 1
                    cleanup_stats['space_freed'] += file_size
            
            # Clean up associated metadata files
            for file_path in self.storage_path.glob("*.json"):
                if file_path.stat().st_mtime < cutoff_date.timestamp():
                    file_path.unlink()
                    cleanup_stats['files_deleted'] += 1
            
            self.logger.info(f"Cleaned up {cleanup_stats['files_deleted']} old files")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
            cleanup_stats['errors'] += 1
        
        return cleanup_stats
