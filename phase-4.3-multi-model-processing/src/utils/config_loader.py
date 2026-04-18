"""
Configuration loader for Phase 4.3
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
        required_sections = ['multi_model', 'bge_base', 'bge_small']
        
        for section in required_sections:
            if section not in config:
                config[section] = {}
        
        # Validate multi-model configuration
        multi_model = config.get('multi_model', {})
        if 'enable_parallel_processing' not in multi_model:
            multi_model['enable_parallel_processing'] = True
        if 'max_concurrent_models' not in multi_model:
            multi_model['max_concurrent_models'] = 2
        
        # Validate BGE-base configuration
        bge_base = config.get('bge_base', {})
        if 'max_urls' not in bge_base:
            bge_base['max_urls'] = 20
        if 'batch_size' not in bge_base:
            bge_base['batch_size'] = 16
        
        # Validate BGE-small configuration
        bge_small = config.get('bge_small', {})
        if 'max_urls' not in bge_small:
            bge_small['max_urls'] = 5
        if 'batch_size' not in bge_small:
            bge_small['batch_size'] = 32
        
        config['multi_model'] = multi_model
        config['bge_base'] = bge_base
        config['bge_small'] = bge_small
        
        return config
    
    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'multi_model': {
                'enable_parallel_processing': True,
                'max_concurrent_models': 2,
                'quality_threshold': 0.7,
                'enable_model_comparison': True,
                'optimize_batch_order': True
            },
            'bge_base': {
                'max_urls': 20,
                'batch_size': 16,
                'quality_threshold': 0.8,
                'enhancement_enabled': True,
                'normalization_enabled': True,
                'cache_embeddings': True,
                'memory_limit_mb': 2048
            },
            'bge_small': {
                'max_urls': 5,
                'batch_size': 32,
                'quality_threshold': 0.7,
                'enhancement_enabled': True,
                'normalization_enabled': True,
                'cache_embeddings': True,
                'memory_limit_mb': 1024,
                'fast_mode': True
            },
            'router': {
                'enable_caching': True,
                'cache_size': 1000,
                'default_model': 'bge_small',
                'complexity_thresholds': {
                    'low': 0.3,
                    'medium': 0.6,
                    'high': 0.8
                }
            }
        }
