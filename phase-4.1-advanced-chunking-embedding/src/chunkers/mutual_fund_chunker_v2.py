"""
Mutual Fund Chunker v2.0
Advanced mutual fund-specific chunking with enhanced domain awareness
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from chunkers.enhanced_semantic_chunker import EnhancedSemanticChunker
from chunkers.recursive_character_splitter import RecursiveCharacterSplitter, SplittingRule
from models.chunk import Chunk


class FundSection(Enum):
    """Mutual fund data sections"""
    BASIC_INFO = "basic_info"
    PERFORMANCE = "performance"
    HOLDINGS = "holdings"
    ALLOCATION = "allocation"
    RISK_ANALYSIS = "risk_analysis"
    TAX_INFO = "tax_info"
    DESCRIPTION = "description"


@dataclass
class FundSectionConfig:
    """Configuration for fund section chunking"""
    section: FundSection
    fields: List[str]
    chunking_strategy: str
    priority: int
    max_chunk_size: int = 800
    min_chunk_size: int = 50


class MutualFundChunkerV2:
    """Advanced mutual fund chunker with enhanced domain awareness"""
    
    def __init__(self):
        self.semantic_chunker = EnhancedSemanticChunker()
        self.recursive_splitter = RecursiveCharacterSplitter()
        
        # Enhanced fund section configurations
        self.section_configs = self._get_section_configs()
        
        # Financial-specific splitting rules
        self.financial_rules = self._get_financial_splitting_rules()
        
        # Initialize recursive splitter with financial rules
        self.recursive_splitter.add_custom_rules(self.financial_rules)
    
    def chunk_fund_data_advanced(self, fund_data: Dict[str, Any]) -> List[Chunk]:
        """Advanced chunking of mutual fund data with multiple strategies"""
        chunks = []
        fund_name = fund_data.get('fund_name', 'Unknown Fund')
        
        # Create structured chunks for each section
        for config in self.section_configs:
            section_chunks = self._chunk_section(fund_data, config, fund_name)
            chunks.extend(section_chunks)
        
        # Create narrative chunks for descriptive content
        description_chunks = self._chunk_description(fund_data, fund_name)
        chunks.extend(description_chunks)
        
        # Create comparison chunks
        comparison_chunks = self._create_comparison_chunks(fund_data, fund_name)
        chunks.extend(comparison_chunks)
        
        # Post-process and validate all chunks
        chunks = self._post_process_fund_chunks(chunks)
        
        return chunks
    
    def _get_section_configs(self) -> List[FundSectionConfig]:
        """Get configurations for different fund sections"""
        return [
            FundSectionConfig(
                section=FundSection.BASIC_INFO,
                fields=['fund_name', 'category', 'risk_level', 'aum', 'fund_manager', 'inception_date'],
                chunking_strategy='structured',
                priority=1
            ),
            FundSectionConfig(
                section=FundSection.PERFORMANCE,
                fields=['nav', 'returns', 'expense_ratio', 'benchmark_performance'],
                chunking_strategy='structured_numeric',
                priority=2
            ),
            FundSectionConfig(
                section=FundSection.HOLDINGS,
                fields=['top_holdings', 'portfolio_composition', 'holding_analysis'],
                chunking_strategy='tabular',
                priority=3
            ),
            FundSectionConfig(
                section=FundSection.ALLOCATION,
                fields=['sector_allocation', 'asset_allocation', 'geographic_allocation'],
                chunking_strategy='allocation',
                priority=4
            ),
            FundSectionConfig(
                section=FundSection.RISK_ANALYSIS,
                fields=['risk_metrics', 'volatility', 'drawdown', 'risk_adjusted_returns'],
                chunking_strategy='analytical',
                priority=5
            ),
            FundSectionConfig(
                section=FundSection.TAX_INFO,
                fields=['tax_implications', 'exit_load', 'tax_saving_benefits'],
                chunking_strategy='tax',
                priority=6
            ),
            FundSectionConfig(
                section=FundSection.DESCRIPTION,
                fields=['description', 'investment_objective', 'strategy'],
                chunking_strategy='semantic',
                priority=7
            )
        ]
    
    def _get_financial_splitting_rules(self) -> List[SplittingRule]:
        """Get financial-specific splitting rules"""
        return [
            SplittingRule(r'(?<=\d+\.?\d*%)\s', 1.5, "After percentage values", is_regex=True),
            SplittingRule(r'(?<=\d+\.\d+ Cr)\s', 1.5, "After crore amounts", is_regex=True),
            SplittingRule(r'(?<=\d+\.\d+ L)\s', 1.5, "After lakh amounts", is_regex=True),
            SplittingRule(r'(?<=Rs\.\s*\d+\.?\d*)\s', 1.5, "After rupee amounts", is_regex=True),
            SplittingRule(r'(?<=XIRR)\s', 2, "After XIRR values", is_regex=True),
            SplittingRule(r'(?<=Sharpe)\s', 2, "After Sharpe ratio", is_regex=True),
            SplittingRule(r'(?<=Beta)\s', 2, "After Beta values", is_regex=True),
            SplittingRule(r'(?<=Alpha)\s', 2, "After Alpha values", is_regex=True),
            SplittingRule(r'(?<=Standard Deviation)\s', 2, "After Std Dev", is_regex=True),
        ]
    
    def _chunk_section(self, fund_data: Dict[str, Any], config: FundSectionConfig, 
                      fund_name: str) -> List[Chunk]:
        """Chunk a specific fund section"""
        section_data = self._extract_section_data(fund_data, config)
        
        if not section_data:
            return []
        
        # Apply section-specific chunking strategy
        if config.chunking_strategy == 'structured':
            return self._chunk_structured_section(section_data, config, fund_name)
        elif config.chunking_strategy == 'structured_numeric':
            return self._chunk_numeric_section(section_data, config, fund_name)
        elif config.chunking_strategy == 'tabular':
            return self._chunk_tabular_section(section_data, config, fund_name)
        elif config.chunking_strategy == 'allocation':
            return self._chunk_allocation_section(section_data, config, fund_name)
        elif config.chunking_strategy == 'analytical':
            return self._chunk_analytical_section(section_data, config, fund_name)
        elif config.chunking_strategy == 'tax':
            return self._chunk_tax_section(section_data, config, fund_name)
        elif config.chunking_strategy == 'semantic':
            return self._chunk_semantic_section(section_data, config, fund_name)
        
        return []
    
    def _extract_section_data(self, fund_data: Dict[str, Any], config: FundSectionConfig) -> Dict[str, Any]:
        """Extract data for a specific section"""
        section_data = {}
        
        for field in config.fields:
            value = fund_data.get(field)
            if value:
                section_data[field] = value
        
        return section_data
    
    def _chunk_structured_section(self, section_data: Dict[str, Any], config: FundSectionConfig, 
                                 fund_name: str) -> List[Chunk]:
        """Chunk structured fund information"""
        chunk_text = self._format_structured_data(section_data)
        
        metadata = {
            'fund_name': fund_name,
            'section_type': config.section.value,
            'chunk_type': 'structured',
            'fields': list(section_data.keys()),
            'strategy': config.chunking_strategy
        }
        
        return [self._create_fund_chunk(chunk_text, metadata)]
    
    def _chunk_numeric_section(self, section_data: Dict[str, Any], config: FundSectionConfig, 
                              fund_name: str) -> List[Chunk]:
        """Chunk numeric performance data"""
        chunks = []
        
        # Group related numeric data
        performance_groups = self._group_numeric_data(section_data)
        
        for group_name, group_data in performance_groups.items():
            chunk_text = self._format_numeric_group(group_name, group_data)
            
            metadata = {
                'fund_name': fund_name,
                'section_type': config.section.value,
                'chunk_type': 'numeric',
                'group': group_name,
                'data_points': len(group_data),
                'strategy': config.chunking_strategy
            }
            
            chunks.append(self._create_fund_chunk(chunk_text, metadata))
        
        return chunks
    
    def _chunk_tabular_section(self, section_data: Dict[str, Any], config: FundSectionConfig, 
                              fund_name: str) -> List[Chunk]:
        """Chunk tabular holdings data"""
        chunks = []
        
        for field, data in section_data.items():
            if isinstance(data, list) and data:
                # Format as table-like structure
                chunk_text = self._format_tabular_data(field, data)
                
                metadata = {
                    'fund_name': fund_name,
                    'section_type': config.section.value,
                    'chunk_type': 'tabular',
                    'table_field': field,
                    'row_count': len(data),
                    'strategy': config.chunking_strategy
                }
                
                chunks.append(self._create_fund_chunk(chunk_text, metadata))
        
        return chunks
    
    def _chunk_allocation_section(self, section_data: Dict[str, Any], config: FundSectionConfig, 
                                 fund_name: str) -> List[Chunk]:
        """Chunk allocation data with percentage analysis"""
        chunks = []
        
        for allocation_type, allocation_data in section_data.items():
            if isinstance(allocation_data, (dict, list)):
                chunk_text = self._format_allocation_data(allocation_type, allocation_data)
                
                # Calculate allocation statistics
                stats = self._calculate_allocation_stats(allocation_data)
                
                metadata = {
                    'fund_name': fund_name,
                    'section_type': config.section.value,
                    'chunk_type': 'allocation',
                    'allocation_type': allocation_type,
                    'allocation_stats': stats,
                    'strategy': config.chunking_strategy
                }
                
                chunks.append(self._create_fund_chunk(chunk_text, metadata))
        
        return chunks
    
    def _chunk_analytical_section(self, section_data: Dict[str, Any], config: FundSectionConfig, 
                                 fund_name: str) -> List[Chunk]:
        """Chunk analytical risk data"""
        chunk_text = self._format_analytical_data(section_data)
        
        metadata = {
            'fund_name': fund_name,
            'section_type': config.section.value,
            'chunk_type': 'analytical',
            'risk_metrics': list(section_data.keys()),
            'strategy': config.chunking_strategy
        }
        
        return [self._create_fund_chunk(chunk_text, metadata)]
    
    def _chunk_tax_section(self, section_data: Dict[str, Any], config: FundSectionConfig, 
                          fund_name: str) -> List[Chunk]:
        """Chunk tax-related information"""
        chunk_text = self._format_tax_data(section_data)
        
        metadata = {
            'fund_name': fund_name,
            'section_type': config.section.value,
            'chunk_type': 'tax',
            'tax_fields': list(section_data.keys()),
            'strategy': config.chunking_strategy
        }
        
        return [self._create_fund_chunk(chunk_text, metadata)]
    
    def _chunk_semantic_section(self, section_data: Dict[str, Any], config: FundSectionConfig, 
                               fund_name: str) -> List[Chunk]:
        """Chunk descriptive content using semantic chunking"""
        all_text = " ".join([str(v) for v in section_data.values() if v])
        
        base_metadata = {
            'fund_name': fund_name,
            'section_type': config.section.value,
            'strategy': config.chunking_strategy
        }
        
        return self.semantic_chunker.chunk_semantic_enhanced(all_text, base_metadata)
    
    def _chunk_description(self, fund_data: Dict[str, Any], fund_name: str) -> List[Chunk]:
        """Chunk fund description with enhanced semantic analysis"""
        description = fund_data.get('description', '')
        investment_objective = fund_data.get('investment_objective', '')
        strategy = fund_data.get('strategy', '')
        
        # Combine all descriptive text
        all_descriptive_text = " ".join(filter(None, [description, investment_objective, strategy]))
        
        if len(all_descriptive_text) > 200:
            metadata = {
                'fund_name': fund_name,
                'section_type': 'description',
                'chunk_type': 'narrative',
                'sources': ['description', 'investment_objective', 'strategy']
            }
            
            return self.semantic_chunker.chunk_semantic_enhanced(all_descriptive_text, metadata)
        
        return []
    
    def _create_comparison_chunks(self, fund_data: Dict[str, Any], fund_name: str) -> List[Chunk]:
        """Create comparison chunks for benchmark analysis"""
        chunks = []
        
        # Performance comparison
        returns = fund_data.get('returns', {})
        benchmark = fund_data.get('benchmark_performance', {})
        
        if returns and benchmark:
            comparison_text = self._create_performance_comparison(returns, benchmark)
            
            metadata = {
                'fund_name': fund_name,
                'section_type': 'comparison',
                'chunk_type': 'performance_comparison',
                'comparison_type': 'returns_vs_benchmark'
            }
            
            chunks.append(self._create_fund_chunk(comparison_text, metadata))
        
        return chunks
    
    def _format_structured_data(self, data: Dict[str, Any]) -> str:
        """Format structured data as readable text"""
        parts = []
        for key, value in data.items():
            if value:
                formatted_key = key.replace('_', ' ').title()
                if isinstance(value, dict):
                    nested_parts = [f"{k}: {v}" for k, v in value.items() if v]
                    parts.append(f"{formatted_key}: {'; '.join(nested_parts)}")
                else:
                    parts.append(f"{formatted_key}: {value}")
        
        return " | ".join(parts)
    
    def _group_numeric_data(self, section_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Group related numeric data"""
        groups = {
            'returns': {},
            'ratios': {},
            'values': {}
        }
        
        for key, value in section_data.items():
            if 'return' in key.lower():
                groups['returns'][key] = value
            elif any(ratio in key.lower() for ratio in ['ratio', 'sharpe', 'alpha', 'beta']):
                groups['ratios'][key] = value
            else:
                groups['values'][key] = value
        
        # Remove empty groups
        return {k: v for k, v in groups.items() if v}
    
    def _format_numeric_group(self, group_name: str, group_data: Dict[str, Any]) -> str:
        """Format numeric group data"""
        parts = [f"{group_name.title()} Metrics:"]
        for key, value in group_data.items():
            formatted_key = key.replace('_', ' ').title()
            parts.append(f"{formatted_key}: {value}")
        
        return " | ".join(parts)
    
    def _format_tabular_data(self, field_name: str, data: List[Dict[str, Any]]) -> str:
        """Format tabular data"""
        if not data:
            return ""
        
        parts = [f"{field_name.replace('_', ' ').title()}:"]
        
        for item in data[:10]:  # Limit to top 10
            if isinstance(item, dict):
                if 'name' in item and 'percentage' in item:
                    parts.append(f"{item['name']} ({item['percentage']})")
                elif 'name' in item:
                    parts.append(item['name'])
                else:
                    # Generic formatting
                    item_parts = [f"{k}: {v}" for k, v in item.items() if v]
                    parts.append(" | ".join(item_parts))
        
        return " | ".join(parts)
    
    def _format_allocation_data(self, allocation_type: str, allocation_data: Any) -> str:
        """Format allocation data"""
        parts = [f"{allocation_type.replace('_', ' ').title()} Allocation:"]
        
        if isinstance(allocation_data, dict):
            for key, value in allocation_data.items():
                if value:
                    parts.append(f"{key}: {value}")
        elif isinstance(allocation_data, list):
            for item in allocation_data:
                if isinstance(item, dict) and 'sector' in item and 'allocation' in item:
                    parts.append(f"{item['sector']} ({item['allocation']})")
        
        return " | ".join(parts)
    
    def _calculate_allocation_stats(self, allocation_data: Any) -> Dict[str, Any]:
        """Calculate allocation statistics"""
        stats = {
            'total_items': 0,
            'has_percentages': False
        }
        
        if isinstance(allocation_data, list):
            stats['total_items'] = len(allocation_data)
            stats['has_percentages'] = any(
                isinstance(item, dict) and 'percentage' in item 
                for item in allocation_data
            )
        elif isinstance(allocation_data, dict):
            stats['total_items'] = len(allocation_data)
        
        return stats
    
    def _format_analytical_data(self, data: Dict[str, Any]) -> str:
        """Format analytical/risk data"""
        parts = ["Risk Analysis Metrics:"]
        for key, value in data.items():
            formatted_key = key.replace('_', ' ').title()
            parts.append(f"{formatted_key}: {value}")
        
        return " | ".join(parts)
    
    def _format_tax_data(self, data: Dict[str, Any]) -> str:
        """Format tax-related data"""
        parts = ["Tax Information:"]
        for key, value in data.items():
            formatted_key = key.replace('_', ' ').title()
            parts.append(f"{formatted_key}: {value}")
        
        return " | ".join(parts)
    
    def _create_performance_comparison(self, returns: Dict[str, Any], 
                                     benchmark: Dict[str, Any]) -> str:
        """Create performance comparison text"""
        parts = ["Performance vs Benchmark:"]
        
        for period in ['1_year', '3_year', '5_year']:
            fund_return = returns.get(period)
            benchmark_return = benchmark.get(period)
            
            if fund_return and benchmark_return:
                try:
                    fund_val = float(fund_return.replace('%', ''))
                    bench_val = float(benchmark_return.replace('%', ''))
                    outperformance = fund_val - bench_val
                    
                    parts.append(f"{period.replace('_', ' ').title()}: Fund {fund_return} vs Benchmark {benchmark_return} (Outperformance: {outperformance:+.2f}%)")
                except (ValueError, AttributeError):
                    parts.append(f"{period.replace('_', ' ').title()}: Fund {fund_return} vs Benchmark {benchmark_return}")
        
        return " | ".join(parts)
    
    def _post_process_fund_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Post-process fund chunks"""
        processed_chunks = []
        
        for chunk in chunks:
            # Validate chunk content
            if self._is_valid_fund_chunk(chunk):
                # Add additional metadata
                chunk.metadata['processed_by'] = 'mutual_fund_chunker_v2'
                chunk.metadata['quality_score'] = self._calculate_chunk_quality(chunk)
                processed_chunks.append(chunk)
        
        return processed_chunks
    
    def _is_valid_fund_chunk(self, chunk: Chunk) -> bool:
        """Validate fund chunk"""
        if not chunk.text or len(chunk.text) < 30:
            return False
        
        # Check for financial content
        financial_keywords = ['fund', 'return', 'nav', 'allocation', 'holding', 'risk', 'performance']
        text_lower = chunk.text.lower()
        
        return any(keyword in text_lower for keyword in financial_keywords)
    
    def _calculate_chunk_quality(self, chunk: Chunk) -> float:
        """Calculate chunk quality score"""
        score = 0.0
        
        # Length score
        length = len(chunk.text)
        if 50 <= length <= 800:
            score += 0.3
        elif 800 < length <= 1200:
            score += 0.2
        
        # Content score
        financial_keywords = ['%', 'nav', 'return', 'cr', 'l', 'rs', 'fund', 'allocation']
        keyword_count = sum(1 for keyword in financial_keywords if keyword in chunk.text.lower())
        score += min(keyword_count * 0.1, 0.4)
        
        # Structure score
        if chunk.metadata.get('chunk_type') in ['structured', 'numeric', 'tabular']:
            score += 0.2
        elif chunk.metadata.get('chunk_type') == 'narrative':
            score += 0.1
        
        return min(score, 1.0)
    
    def _create_fund_chunk(self, text: str, metadata: Dict[str, Any]) -> Chunk:
        """Create a fund-specific chunk"""
        chunk_metadata = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'created_by': 'mutual_fund_chunker_v2',
            **metadata
        }
        
        return Chunk(text=text, metadata=chunk_metadata)
