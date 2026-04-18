"""
Data processor for mutual fund data
Handles chunking, validation, and preparation for embedding generation
"""

import asyncio
import json
import logging
from typing import List, Dict, Any
from pathlib import Path
import re
from datetime import datetime

from chunkers.semantic_chunker import SemanticChunker
from chunkers.mutual_fund_chunker import MutualFundChunker
from chunkers.fixed_size_chunker import FixedSizeChunker
from models.chunk import Chunk


class DataProcessor:
    """Main data processor for mutual fund scraping results"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.semantic_chunker = SemanticChunker()
        self.mutual_fund_chunker = MutualFundChunker()
        self.fixed_size_chunker = FixedSizeChunker()
        
        # Data directories
        self.raw_data_dir = Path("data/raw")
        self.processed_data_dir = Path("data/processed")
        self.embeddings_dir = Path("data/embeddings")
        
        # Ensure directories exist
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    async def process_scraped_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process scraped mutual fund data"""
        self.logger.info(f"Processing {len(raw_data)} raw fund records")
        
        processed_funds = []
        
        for fund_data in raw_data:
            try:
                # Validate and clean fund data
                cleaned_fund = await self._clean_fund_data(fund_data)
                
                # Create structured chunks for the fund
                chunks = await self._create_fund_chunks(cleaned_fund)
                
                # Validate chunks
                valid_chunks = await self._validate_chunks(chunks)
                
                # Add chunks to fund data
                cleaned_fund['chunks'] = valid_chunks
                cleaned_fund['chunk_count'] = len(valid_chunks)
                
                processed_funds.append(cleaned_fund)
                
                self.logger.info(f"Processed fund: {cleaned_fund.get('fund_name', 'Unknown')} with {len(valid_chunks)} chunks")
                
            except Exception as e:
                self.logger.error(f"Error processing fund {fund_data.get('fund_name', 'Unknown')}: {str(e)}")
                continue
        
        self.logger.info(f"Successfully processed {len(processed_funds)} funds")
        return processed_funds
    
    async def _clean_fund_data(self, fund_data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and standardize fund data"""
        cleaned = fund_data.copy()
        
        # Standardize numeric fields
        numeric_fields = {
            'expense_ratio': 'percentage',
            'nav': 'currency',
            'aum': 'currency'
        }
        
        for field, field_type in numeric_fields.items():
            if cleaned.get(field):
                cleaned[field] = self._standardize_numeric_field(cleaned[field], field_type)
        
        # Clean returns data
        returns = cleaned.get('returns', {})
        if isinstance(returns, dict):
            cleaned_returns = {}
            for period, value in returns.items():
                cleaned_returns[period] = self._standardize_numeric_field(value, 'percentage')
            cleaned['returns'] = cleaned_returns
        
        # Clean holdings data
        holdings = cleaned.get('top_holdings', [])
        if isinstance(holdings, list):
            cleaned_holdings = []
            for holding in holdings:
                if isinstance(holding, dict):
                    cleaned_holding = {
                        'name': holding.get('name', '').strip(),
                        'percentage': self._standardize_numeric_field(holding.get('percentage', ''), 'percentage')
                    }
                    if cleaned_holding['name']:
                        cleaned_holdings.append(cleaned_holding)
            cleaned['top_holdings'] = cleaned_holdings
        
        # Clean sector allocation
        sectors = cleaned.get('sector_allocation', [])
        if isinstance(sectors, list):
            cleaned_sectors = []
            for sector in sectors:
                if isinstance(sector, dict):
                    cleaned_sector = {
                        'sector': sector.get('sector', '').strip(),
                        'allocation': self._standardize_numeric_field(sector.get('allocation', ''), 'percentage')
                    }
                    if cleaned_sector['sector']:
                        cleaned_sectors.append(cleaned_sector)
            cleaned['sector_allocation'] = cleaned_sectors
        
        # Clean asset allocation
        asset_allocation = cleaned.get('asset_allocation', {})
        if isinstance(asset_allocation, dict):
            cleaned_allocation = {}
            for asset_type, value in asset_allocation.items():
                cleaned_allocation[asset_type] = self._standardize_numeric_field(value, 'percentage')
            cleaned['asset_allocation'] = cleaned_allocation
        
        # Add processing metadata
        cleaned['processed_at'] = datetime.utcnow().isoformat()
        cleaned['data_quality_score'] = self._calculate_data_quality_score(cleaned)
        
        return cleaned
    
    def _standardize_numeric_field(self, value: str, field_type: str) -> str:
        """Standardize numeric fields"""
        if not value:
            return ""
        
        # Remove currency symbols, commas, extra spaces
        cleaned = re.sub(r'[^\d.%-]', '', str(value))
        
        if not cleaned:
            return ""
        
        try:
            if field_type == 'percentage':
                # Ensure percentage format
                if '%' not in cleaned:
                    # Check if it's already a percentage (0-100)
                    num_value = float(cleaned)
                    if 0 <= num_value <= 100:
                        cleaned = f"{num_value:.2f}%"
                    else:
                        # Might be in decimal form (0.15 = 15%)
                        if num_value < 1:
                            cleaned = f"{num_value * 100:.2f}%"
                        else:
                            cleaned = f"{num_value}%"
                else:
                    # Clean up percentage format
                    num_value = float(cleaned.replace('%', ''))
                    cleaned = f"{num_value:.2f}%"
            
            elif field_type == 'currency':
                # Remove % if present (sometimes mistakenly included)
                cleaned = cleaned.replace('%', '')
                num_value = float(cleaned)
                # Format as currency (assuming INR)
                if num_value >= 10000000:  # Crores
                    cleaned = f"â¹{num_value/10000000:.2f} Cr"
                elif num_value >= 100000:  # Lakhs
                    cleaned = f"â¹{num_value/100000:.2f} L"
                else:
                    cleaned = f"â¹{num_value:.2f}"
            
        except (ValueError, TypeError):
            self.logger.warning(f"Could not standardize numeric field: {value}")
            return value
        
        return cleaned
    
    async def _create_fund_chunks(self, fund_data: Dict[str, Any]) -> List[Chunk]:
        """Create chunks for mutual fund data"""
        chunks = []
        
        # Use mutual fund specific chunker first
        fund_chunks = self.mutual_fund_chunker.chunk_fund_data(fund_data)
        chunks.extend(fund_chunks)
        
        # Create narrative chunks for description
        description = fund_data.get('description', '')
        if description and len(description) > 200:
            semantic_chunks = self.semantic_chunker.chunk_semantic(description)
            for i, chunk in enumerate(semantic_chunks):
                chunk.metadata.update({
                    'fund_name': fund_data.get('fund_name'),
                    'section_type': 'description',
                    'chunk_index': i,
                    'chunk_type': 'narrative'
                })
                chunks.append(chunk)
        
        return chunks
    
    async def _validate_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Validate chunks and filter out invalid ones"""
        valid_chunks = []
        
        for chunk in chunks:
            if self._is_valid_chunk(chunk):
                valid_chunks.append(chunk)
            else:
                self.logger.warning(f"Invalid chunk filtered out: {chunk.id}")
        
        return valid_chunks
    
    def _is_valid_chunk(self, chunk: Chunk) -> bool:
        """Check if a chunk is valid"""
        # Length validation
        if len(chunk.text) < 50 or len(chunk.text) > 2000:
            return False
        
        # Content validation
        if not self._has_meaningful_content(chunk.text):
            return False
        
        # Required metadata
        required_fields = ['fund_name', 'section_type']
        if not all(field in chunk.metadata for field in required_fields):
            return False
        
        return True
    
    def _has_meaningful_content(self, text: str) -> bool:
        """Check if chunk contains meaningful content"""
        text_lower = text.lower()
        meaningless_patterns = [
            'click here', 'read more', 'view details', 'n/a', 
            'not available', 'data not available', 'tbd'
        ]
        
        return not any(pattern in text_lower for pattern in meaningless_patterns)
    
    def _calculate_data_quality_score(self, fund_data: Dict[str, Any]) -> float:
        """Calculate data quality score (0-1)"""
        score = 0.0
        max_score = 10.0
        
        # Fund name (essential)
        if fund_data.get('fund_name') and fund_data['fund_name'] != "Unknown Fund":
            score += 2.0
        
        # Key financial metrics
        key_fields = ['nav', 'expense_ratio', 'returns']
        for field in key_fields:
            if fund_data.get(field):
                score += 1.5
        
        # Additional valuable data
        additional_fields = ['fund_manager', 'inception_date', 'top_holdings', 'sector_allocation']
        for field in additional_fields:
            if fund_data.get(field):
                score += 0.5
        
        # Description
        if fund_data.get('description') and len(fund_data['description']) > 100:
            score += 1.0
        
        return min(score / max_score, 1.0)
    
    async def save_processed_data(self, processed_data: List[Dict[str, Any]]) -> None:
        """Save processed data to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save each fund as separate file
        for fund_data in processed_data:
            fund_name = fund_data.get('fund_name', 'unknown').lower()
            fund_name = re.sub(r'[^a-z0-9]', '_', fund_name)
            
            filename = f"{fund_name}_{timestamp}.json"
            filepath = self.processed_data_dir / filename
            
            # Remove chunks from main data for cleaner JSON
            fund_copy = fund_data.copy()
            chunks = fund_copy.pop('chunks', [])
            
            # Save fund data
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(fund_copy, f, indent=2, ensure_ascii=False)
            
            # Save chunks separately
            if chunks:
                chunks_filename = f"{fund_name}_chunks_{timestamp}.json"
                chunks_filepath = self.processed_data_dir / chunks_filename
                
                chunks_data = []
                for chunk in chunks:
                    chunk_dict = {
                        'id': chunk.id,
                        'text': chunk.text,
                        'metadata': chunk.metadata
                    }
                    chunks_data.append(chunk_dict)
                
                with open(chunks_filepath, 'w', encoding='utf-8') as f:
                    json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        # Save summary
        summary = {
            'timestamp': timestamp,
            'total_funds': len(processed_data),
            'funds': [
                {
                    'fund_name': fund.get('fund_name'),
                    'chunk_count': fund.get('chunk_count', 0),
                    'data_quality_score': fund.get('data_quality_score', 0),
                    'validation_status': fund.get('validation_status', 'unknown')
                }
                for fund in processed_data
            ]
        }
        
        summary_filename = f"summary_{timestamp}.json"
        summary_filepath = self.processed_data_dir / summary_filename
        
        with open(summary_filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved processed data with timestamp {timestamp}")
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        stats = {
            'processed_files': 0,
            'total_chunks': 0,
            'last_processed': None
        }
        
        if self.processed_data_dir.exists():
            json_files = list(self.processed_data_dir.glob('*.json'))
            stats['processed_files'] = len(json_files)
            
            # Find last processed file
            summary_files = list(self.processed_data_dir.glob('summary_*.json'))
            if summary_files:
                latest_summary = max(summary_files, key=lambda x: x.stat().st_mtime)
                stats['last_processed'] = latest_summary.stat().st_mtime
                
                # Read summary for chunk count
                try:
                    with open(latest_summary) as f:
                        summary_data = json.load(f)
                        stats['total_chunks'] = sum(fund.get('chunk_count', 0) for fund in summary_data.get('funds', []))
                except Exception:
                    pass
        
        return stats
