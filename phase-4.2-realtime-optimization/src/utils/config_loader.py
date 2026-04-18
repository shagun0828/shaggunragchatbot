"""
Configuration loader for Phase 4.2
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
        required_sections = ['chunking', 'embedding', 'storage', 'monitoring']
        
        for section in required_sections:
            if section not in config:
                config[section] = {}
        
        return config
    
    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'chunking': {
                'queue_size': 1000,
                'batch_size': 10,
                'max_concurrent': 4,
                'quality_threshold': 0.7
            },
            'embedding': {
                'optimization_interval': 60.0,
                'quality_threshold': 0.7,
                'speed_threshold': 2.0
            },
            'storage': {
                'buffer_size': 1000,
                'batch_size': 50,
                'flush_interval': 5.0,
                'quality_threshold': 0.7
            },
            'monitoring': {
                'metrics_collection_interval': 1.0,
                'alert_thresholds': {
                    'error_rate': 0.05,
                    'processing_time': 5.0,
                    'memory_usage': 0.9
                }
            }
        }
