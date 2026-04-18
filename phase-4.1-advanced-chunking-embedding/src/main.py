#!/usr/bin/env python3
"""
Main entry point for Phase 4.1: Advanced Chunking and Embedding
Enhanced chunking and embedding with quality assurance
"""

import asyncio
import logging
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from chunkers.enhanced_semantic_chunker import EnhancedSemanticChunker
from chunkers.recursive_character_splitter import RecursiveCharacterSplitter
from chunkers.mutual_fund_chunker_v2 import MutualFundChunkerV2
from embedders.enhanced_financial_embedder import EnhancedFinancialEmbedder
from embedders.embedding_quality_checker import EmbeddingQualityChecker
from storage.advanced_vector_storage import AdvancedVectorStorage
from utils.logger import setup_logger
from utils.config_loader import ConfigLoader
from utils.notifications import NotificationManager


class AdvancedChunkingEmbeddingProcessor:
    """Main processor for advanced chunking and embedding"""
    
    def __init__(self, config_path: str = "config/chunking_config.yaml"):
        self.logger = setup_logger("advanced_chunking_embedding")
        self.config = ConfigLoader.load_config(config_path)
        
        # Initialize components
        self.semantic_chunker = EnhancedSemanticChunker(
            model_name=self.config['chunking']['semantic_chunker']['model_name']
        )
        self.recursive_splitter = RecursiveCharacterSplitter()
        self.mutual_fund_chunker = MutualFundChunkerV2()
        self.financial_embedder = EnhancedFinancialEmbedder(
            base_model=self.config['embedding']['financial_embedder']['base_model']
        )
        self.quality_checker = EmbeddingQualityChecker(
            similarity_threshold=self.config['embedding']['quality_checker']['similarity_threshold'],
            variance_threshold=self.config['embedding']['quality_checker']['variance_threshold'],
            outlier_threshold=self.config['embedding']['quality_checker']['outlier_threshold']
        )
        self.vector_storage = AdvancedVectorStorage()
        self.notification_manager = NotificationManager()
        
        # Processing statistics
        self.stats = {
            'funds_processed': 0,
            'chunks_created': 0,
            'embeddings_generated': 0,
            'quality_issues_fixed': 0,
            'processing_time': 0
        }
    
    async def process_fund_data(self, fund_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process fund data with advanced chunking and embedding"""
        self.logger.info(f"Starting advanced processing for {len(fund_data_list)} funds")
        start_time = asyncio.get_event_loop().time()
        
        processing_results = {
            'total_funds': len(fund_data_list),
            'processed_funds': 0,
            'total_chunks': 0,
            'quality_reports': [],
            'storage_results': {},
            'errors': []
        }
        
        try:
            # Step 1: Advanced chunking for each fund
            all_chunks = []
            for fund_data in fund_data_list:
                try:
                    fund_chunks = await self._process_fund_chunking(fund_data)
                    all_chunks.extend(fund_chunks)
                    processing_results['processed_funds'] += 1
                except Exception as e:
                    self.logger.error(f"Failed to chunk fund {fund_data.get('fund_name', 'Unknown')}: {e}")
                    processing_results['errors'].append(f"Chunking failed for {fund_data.get('fund_name', 'Unknown')}: {e}")
                    continue
            
            processing_results['total_chunks'] = len(all_chunks)
            
            if not all_chunks:
                self.logger.warning("No chunks created, returning empty results")
                return processing_results
            
            # Step 2: Generate enhanced embeddings
            base_embeddings = await self._generate_base_embeddings(all_chunks)
            enhanced_embeddings = self.financial_embedder.enhance_financial_embeddings(all_chunks, base_embeddings)
            
            # Step 3: Quality assurance
            quality_report = self.quality_checker.check_embedding_quality(enhanced_embeddings, all_chunks)
            processing_results['quality_reports'].append(quality_report)
            
            # Step 4: Store with quality assurance
            processed_data = [{'chunks': all_chunks}]  # Format for storage
            storage_results = await self.vector_storage.store_embeddings_with_quality_assurance(processed_data)
            processing_results['storage_results'] = storage_results
            
            # Step 5: Update statistics
            end_time = asyncio.get_event_loop().time()
            self.stats['processing_time'] = end_time - start_time
            self.stats['funds_processed'] = processing_results['processed_funds']
            self.stats['chunks_created'] = processing_results['total_chunks']
            self.stats['embeddings_generated'] = len(all_chunks)
            self.stats['quality_issues_fixed'] = storage_results.get('auto_fixed', 0)
            
            # Step 6: Send notifications
            await self._send_processing_notifications(processing_results, quality_report)
            
            self.logger.info(f"Advanced processing completed: {processing_results['processed_funds']} funds, {processing_results['total_chunks']} chunks")
            
        except Exception as e:
            self.logger.error(f"Advanced processing failed: {e}")
            processing_results['errors'].append(f"Processing failed: {e}")
            await self.notification_manager.send_failure_notification(f"Advanced processing failed: {e}")
            raise
        
        return processing_results
    
    async def _process_fund_chunking(self, fund_data: Dict[str, Any]) -> List:
        """Process chunking for a single fund using advanced strategies"""
        from models.chunk import Chunk
        
        fund_name = fund_data.get('fund_name', 'Unknown Fund')
        self.logger.info(f"Advanced chunking for fund: {fund_name}")
        
        # Use mutual fund chunker v2 for domain-aware chunking
        chunks = self.mutual_fund_chunker.chunk_fund_data_advanced(fund_data)
        
        # Validate chunks
        valid_chunks = []
        for chunk in chunks:
            if self._validate_chunk(chunk):
                valid_chunks.append(chunk)
            else:
                self.logger.warning(f"Invalid chunk filtered out: {chunk.id}")
        
        self.logger.info(f"Created {len(valid_chunks)} valid chunks for {fund_name}")
        return valid_chunks
    
    async def _generate_base_embeddings(self, chunks: List) -> Any:
        """Generate base embeddings for chunks"""
        self.logger.info(f"Generating base embeddings for {len(chunks)} chunks")
        
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
    
    def _validate_chunk(self, chunk) -> bool:
        """Validate chunk quality"""
        # Length validation
        if len(chunk.text) < 30 or len(chunk.text) > 2000:
            return False
        
        # Content validation
        if not self._has_meaningful_content(chunk.text):
            return False
        
        # Required metadata validation
        required_fields = ['fund_name', 'section_type']
        if not all(field in chunk.metadata for field in required_fields):
            return False
        
        return True
    
    def _has_meaningful_content(self, text: str) -> bool:
        """Check if chunk contains meaningful content"""
        text_lower = text.lower()
        meaningless_patterns = ['click here', 'read more', 'view details', 'n/a', 'tbd']
        
        return not any(pattern in text_lower for pattern in meaningless_patterns)
    
    async def _send_processing_notifications(self, results: Dict[str, Any], quality_report) -> None:
        """Send processing notifications"""
        success_rate = results['processed_funds'] / results['total_funds'] if results['total_funds'] > 0 else 0
        
        if success_rate >= 0.9 and quality_report.overall_score >= 0.7:
            message = (
                f"Advanced chunking and embedding completed successfully!\n"
                f"Processed: {results['processed_funds']}/{results['total_funds']} funds\n"
                f"Chunks created: {results['total_chunks']}\n"
                f"Quality score: {quality_report.overall_score:.3f}\n"
                f"Processing time: {self.stats['processing_time']:.2f}s"
            )
            await self.notification_manager.send_success_notification(message)
        else:
            message = (
                f"Advanced processing completed with issues:\n"
                f"Success rate: {success_rate:.1%}\n"
                f"Quality score: {quality_report.overall_score:.3f}\n"
                f"Errors: {len(results['errors'])}"
            )
            await self.notification_manager.send_warning_notification(message)
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        stats = self.stats.copy()
        stats.update(self.vector_storage.get_storage_statistics())
        return stats


async def main():
    """Main entry point"""
    logger = setup_logger("main")
    logger.info("Starting Phase 4.1: Advanced Chunking and Embedding")
    
    try:
        # Initialize processor
        processor = AdvancedChunkingEmbeddingProcessor()
        
        # Load sample data (in production, this would come from Phase 4.0)
        sample_fund_data = [
            {
                'fund_name': 'HDFC Mid-Cap Fund Direct Growth',
                'category': 'Mid Cap',
                'risk_level': 'Very High',
                'aum': 'â¹28,432 Cr',
                'nav': 'â¹175.43',
                'returns': {'1_year': '24.5%', '3_year': '18.2%', '5_year': '16.8%'},
                'expense_ratio': '1.25%',
                'fund_manager': 'Rashmi Joshi',
                'inception_date': '01-Jan-2010',
                'description': 'The scheme aims to generate long-term capital appreciation by investing in mid-cap companies.',
                'investment_objective': 'To provide long-term capital growth through mid-cap investments.',
                'top_holdings': [
                    {'name': 'Reliance Industries', 'percentage': '8.5%'},
                    {'name': 'TCS', 'percentage': '7.2%'},
                    {'name': 'HDFC Bank', 'percentage': '6.8%'}
                ],
                'sector_allocation': [
                    {'sector': 'Financial Services', 'allocation': '25%'},
                    {'sector': 'Technology', 'allocation': '20%'},
                    {'sector': 'Healthcare', 'allocation': '15%'}
                ],
                'asset_allocation': {'equity': '95%', 'debt': '3%', 'cash': '2%'}
            }
        ]
        
        # Process data
        results = await processor.process_fund_data(sample_fund_data)
        
        # Print results
        logger.info(f"Processing completed: {results}")
        
        # Print statistics
        stats = processor.get_processing_statistics()
        logger.info(f"Processing statistics: {stats}")
        
    except Exception as e:
        logger.error(f"Main processing failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
