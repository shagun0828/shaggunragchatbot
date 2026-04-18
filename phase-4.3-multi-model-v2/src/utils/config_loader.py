"""
Configuration loader for Phase 4.3
Loads and validates YAML configuration files with environment variable integration
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from .env_loader import env_loader


class ConfigLoader:
    """Configuration loader and validator"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file with environment variable integration"""
        logger = logging.getLogger(__name__)
        
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                logger.warning(f"Config file {config_path} not found, using defaults")
                return ConfigLoader._get_default_config()
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Integrate environment variables
            config = ConfigLoader._integrate_env_vars(config)
            
            # Validate configuration
            validated_config = ConfigLoader._validate_config(config)
            
            logger.info(f"Loaded configuration from {config_path} with environment integration")
            return validated_config
            
        except Exception as e:
            logger.error(f"Failed to load config {config_path}: {e}")
            logger.info("Using default configuration")
            return ConfigLoader._get_default_config()
    
    @staticmethod
    def _integrate_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate environment variables into configuration"""
        # Get environment configuration
        env_config = env_loader.get_all_config()
        
        # Merge Chroma Cloud configuration
        if 'chroma_cloud' not in config:
            config['chroma_cloud'] = {}
        config['chroma_cloud'].update(env_config['chroma_cloud'])
        
        # Merge performance configuration
        if 'performance' not in config:
            config['performance'] = {}
        config['performance'].update(env_config['performance'])
        
        # Merge notification configuration
        if 'notifications' not in config:
            config['notifications'] = {}
        config['notifications'].update(env_config['notifications'])
        
        # Merge development configuration
        if 'development' not in config:
            config['development'] = {}
        config['development'].update(env_config['development'])
        
        # Override specific values from environment
        config['multi_model']['enable_parallel_processing'] = env_loader.get_bool('ENABLE_PARALLEL_PROCESSING', config['multi_model'].get('enable_parallel_processing', True))
        config['multi_model']['max_concurrent_models'] = env_loader.get_int('MAX_CONCURRENT_MODELS', config['multi_model'].get('max_concurrent_models', 2))
        
        config['bge_base']['memory_limit_mb'] = env_loader.get_int('MEMORY_LIMIT_MB', config['bge_base'].get('memory_limit_mb', 2048))
        config['bge_small']['memory_limit_mb'] = env_loader.get_int('MEMORY_LIMIT_MB', config['bge_small'].get('memory_limit_mb', 1024))
        
        config['bge_base']['batch_size'] = env_loader.get_int('DEFAULT_BATCH_SIZE', config['bge_base'].get('batch_size', 16))
        config['bge_small']['batch_size'] = env_loader.get_int('DEFAULT_BATCH_SIZE', config['bge_small'].get('batch_size', 32))
        
        # Override debug mode
        config['development']['debug_mode'] = env_loader.get_bool('DEBUG_MODE', config['development'].get('debug_mode', False))
        config['development']['mock_external_services'] = env_loader.get_bool('MOCK_EXTERNAL_SERVICES', config['development'].get('mock_external_services', True))
        
        return config
    
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
        if 'processing_strategy' not in multi_model:
            multi_model['processing_strategy'] = 'adaptive'
        if 'coordination_mode' not in multi_model:
            multi_model['coordination_mode'] = 'balanced'
        if 'enable_parallel_processing' not in multi_model:
            multi_model['enable_parallel_processing'] = True
        
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
                'processing_strategy': 'adaptive',
                'coordination_mode': 'balanced',
                'enable_parallel_processing': True,
                'max_concurrent_models': 2,
                'quality_threshold': 0.7,
                'enable_quality_assessment': True,
                'enable_performance_monitoring': True,
                'enable_adaptive_routing': True,
                'enable_load_balancing': True,
                'retry_attempts': 3,
                'retry_delay': 1.0,
                'timeout_per_url': 30.0
            },
            'bge_base': {
                'max_urls': 20,
                'batch_size': 16,
                'quality_threshold': 0.8,
                'enhancement_enabled': True,
                'normalization_enabled': True,
                'cache_embeddings': True,
                'memory_limit_mb': 2048,
                'quality_assessment': True,
                'financial_enhancement': True
            },
            'bge_small': {
                'max_urls': 5,
                'batch_size': 32,
                'quality_threshold': 0.7,
                'enhancement_enabled': True,
                'normalization_enabled': True,
                'cache_embeddings': True,
                'memory_limit_mb': 1024,
                'processing_mode': 'fast',
                'lightweight_enhancement': True,
                'quality_assessment': True
            },
            'router': {
                'enable_caching': True,
                'cache_size': 1000,
                'default_model': 'bge_small',
                'complexity_thresholds': {
                    'low': 0.3,
                    'medium': 0.6,
                    'high': 0.8,
                    'very_high': 0.9
                }
            }
        }
