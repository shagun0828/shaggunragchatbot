"""
Configuration loader for Phase 4.1
Loads and validates YAML configuration files
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """Configuration loader and validator"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        logger = logging.getLogger(__name__)
        
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                logger.warning(f"Config file {config_path} not found, using defaults")
                return ConfigLoader._get_default_config()
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Validate configuration
            validated_config = ConfigLoader._validate_config(config)
            
            logger.info(f"Loaded configuration from {config_path}")
            return validated_config
            
        except Exception as e:
            logger.error(f"Failed to load config {config_path}: {e}")
            logger.info("Using default configuration")
            return ConfigLoader._get_default_config()
    
    @staticmethod
    def _validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration structure and values"""
        # Ensure required sections exist
        required_sections = ['chunking', 'embedding', 'storage', 'processing']
        
        for section in required_sections:
            if section not in config:
                config[section] = {}
        
        # Validate chunking configuration
        chunking_config = config['chunking']
        if 'semantic_chunker' not in chunking_config:
            chunking_config['semantic_chunker'] = {}
        
        if 'mutual_fund_chunker' not in chunking_config:
            chunking_config['mutual_fund_chunker'] = {}
        
        # Validate embedding configuration
        embedding_config = config['embedding']
        if 'financial_embedder' not in embedding_config:
            embedding_config['financial_embedder'] = {}
        
        if 'quality_checker' not in embedding_config:
            embedding_config['quality_checker'] = {}
        
        # Validate storage configuration
        storage_config = config['storage']
        if 'vector_storage' not in storage_config:
            storage_config['vector_storage'] = {}
        
        # Validate processing configuration
        processing_config = config['processing']
        if 'pipeline' not in processing_config:
            processing_config['pipeline'] = {}
        
        return config
    
    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'chunking': {
                'semantic_chunker': {
                    'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
                    'base_similarity_threshold': 0.7,
                    'max_chunk_size': 1000,
                    'min_chunk_size': 100,
                    'adaptive_threshold': True
                },
                'mutual_fund_chunker': {
                    'section_configs': {}
                }
            },
            'embedding': {
                'financial_embedder': {
                    'base_model': 'sentence-transformers/all-MiniLM-L6-v2',
                    'enhancement_strength': 0.3,
                    'context_awareness': True
                },
                'quality_checker': {
                    'similarity_threshold': 0.95,
                    'variance_threshold': 0.01,
                    'outlier_threshold': 3.0
                }
            },
            'storage': {
                'vector_storage': {
                    'batch_size': 100,
                    'min_quality_score': 0.7,
                    'auto_fix_quality': True
                }
            },
            'processing': {
                'pipeline': {
                    'enable_quality_assurance': True,
                    'enable_financial_enhancement': True
                }
            }
        }
