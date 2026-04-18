"""
Mutual fund specific chunker
Chunks mutual fund data with domain awareness
"""

from typing import List, Dict, Any

from chunkers.semantic_chunker import SemanticChunker
from chunkers.fixed_size_chunker import FixedSizeChunker
from models.chunk import Chunk


class MutualFundChunker:
    """Chunks mutual fund data with domain-specific logic"""
    
    def __init__(self):
        self.semantic_chunker = SemanticChunker()
        self.fixed_size_chunker = FixedSizeChunker()
        
        # Define fund data sections
        self.fund_sections = {
            'basic_info': ['fund_name', 'category', 'risk_level', 'aum'],
            'performance': ['returns_1y', 'returns_3y', 'returns_5y', 'nav'],
            'allocation': ['equity', 'debt', 'cash', 'sector_allocation'],
            'holdings': ['top_holdings', 'portfolio_composition'],
            'metadata': ['expense_ratio', 'fund_manager', 'inception_date']
        }
    
    def chunk_fund_data(self, fund_data: Dict[str, Any]) -> List[Chunk]:
        """Chunk mutual fund data with domain awareness"""
        chunks = []
        fund_name = fund_data.get('fund_name', 'Unknown Fund')
        
        # Create structured chunks for each section
        for section, fields in self.fund_sections.items():
            section_text = self._extract_section_text(fund_data, section, fields)
            if section_text:
                chunk = self._create_structured_chunk(
                    text=section_text,
                    section=section,
                    fund_name=fund_name,
                    metadata={'section_type': section, 'fields': fields}
                )
                chunks.append(chunk)
        
        # Create narrative chunks for descriptive content
        description = fund_data.get('description', '')
        if description and len(description) > 200:
            semantic_chunks = self.semantic_chunker.chunk_semantic(description)
            for chunk in semantic_chunks:
                chunk.metadata.update({
                    'fund_name': fund_name,
                    'section_type': 'description',
                    'chunk_type': 'narrative'
                })
                chunks.append(chunk)
        
        return chunks
    
    def _extract_section_text(self, fund_data: Dict[str, Any], section: str, fields: List[str]) -> str:
        """Extract text for a specific section"""
        section_parts = []
        
        for field in fields:
            value = fund_data.get(field)
            if value:
                if isinstance(value, dict):
                    # Handle nested data like returns, asset allocation
                    formatted_value = self._format_nested_data(value, field)
                    if formatted_value:
                        section_parts.append(f"{field.replace('_', ' ').title()}: {formatted_value}")
                elif isinstance(value, list):
                    # Handle list data like holdings, sector allocation
                    formatted_value = self._format_list_data(value, field)
                    if formatted_value:
                        section_parts.append(f"{field.replace('_', ' ').title()}: {formatted_value}")
                else:
                    # Handle simple values
                    section_parts.append(f"{field.replace('_', ' ').title()}: {value}")
        
        return " | ".join(section_parts)
    
    def _format_nested_data(self, data: Dict[str, Any], field_name: str) -> str:
        """Format nested dictionary data"""
        if not data:
            return ""
        
        parts = []
        for key, value in data.items():
            if value:
                parts.append(f"{key}: {value}")
        
        return "; ".join(parts) if parts else ""
    
    def _format_list_data(self, data: List[Dict[str, Any]], field_name: str) -> str:
        """Format list data"""
        if not data:
            return ""
        
        # Limit to top items for readability
        top_items = data[:5] if field_name == 'top_holdings' else data[:3]
        
        parts = []
        for item in top_items:
            if isinstance(item, dict):
                # Format as "name: value"
                if 'name' in item and 'percentage' in item:
                    parts.append(f"{item['name']} ({item['percentage']})")
                elif 'name' in item:
                    parts.append(item['name'])
                else:
                    # Generic formatting
                    item_parts = [f"{k}: {v}" for k, v in item.items() if v]
                    parts.append(" | ".join(item_parts))
        
        return "; ".join(parts) if parts else ""
    
    def _create_structured_chunk(self, text: str, section: str, fund_name: str, metadata: Dict[str, Any]) -> Chunk:
        """Create a structured chunk for fund data"""
        chunk_metadata = {
            'fund_name': fund_name,
            'section_type': section,
            'chunk_type': 'structured',
            'text_length': len(text),
            **metadata
        }
        return Chunk(text=text, metadata=chunk_metadata)
