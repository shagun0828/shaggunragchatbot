"""
Environment variable loader for Phase 4.3
Loads and validates environment variables from .env file
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv


class EnvLoader:
    """Environment variable loader and validator"""
    
    def __init__(self, env_file: str = ".env"):
        self.logger = logging.getLogger(__name__)
        self.env_file = env_file
        
        # Load environment variables
        self._load_env_file()
        
        # Validate required variables
        self._validate_required_vars()
    
    def _load_env_file(self) -> None:
        """Load environment variables from .env file"""
        try:
            # Try to load from project root
            project_root = Path(__file__).parent.parent.parent
            env_path = project_root / self.env_file
            
            if env_path.exists():
                load_dotenv(env_path)
                self.logger.info(f"Loaded environment variables from {env_path}")
            else:
                self.logger.warning(f"Environment file {env_path} not found, using system environment")
                
        except Exception as e:
            self.logger.error(f"Failed to load environment file: {e}")
    
    def _validate_required_vars(self) -> None:
        """Validate required environment variables"""
        required_vars = []
        
        # Check Chroma Cloud configuration if enabled
        if self.get_bool('ENABLE_CHROMA_CLOUD', False):  # Default to False for testing
            if not os.getenv('CHROMA_API_KEY'):
                required_vars.append('CHROMA_API_KEY')
        
        # Report missing required variables
        if required_vars:
            self.logger.error(f"Missing required environment variables: {required_vars}")
            raise EnvironmentError(f"Missing required environment variables: {required_vars}")
    
    def get_str(self, key: str, default: Optional[str] = None) -> str:
        """Get string environment variable"""
        value = os.getenv(key, default)
        if value is None:
            self.logger.warning(f"Environment variable {key} not found, using default: {default}")
        return value or ""
    
    def get_int(self, key: str, default: int = 0) -> int:
        """Get integer environment variable"""
        try:
            value = os.getenv(key)
            if value is None:
                return default
            return int(value)
        except ValueError:
            self.logger.error(f"Invalid integer value for {key}: {value}, using default: {default}")
            return default
    
    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get float environment variable"""
        try:
            value = os.getenv(key)
            if value is None:
                return default
            return float(value)
        except ValueError:
            self.logger.error(f"Invalid float value for {key}: {value}, using default: {default}")
            return default
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean environment variable"""
        value = os.getenv(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on', 'enabled')
    
    def get_list(self, key: str, default: list = None, separator: str = ',') -> list:
        """Get list environment variable"""
        value = os.getenv(key)
        if value is None:
            return default or []
        return [item.strip() for item in value.split(separator) if item.strip()]
    
    def get_chroma_config(self) -> Dict[str, Any]:
        """Get Chroma Cloud configuration"""
        return {
            'api_key': self.get_str('CHROMA_API_KEY'),
            'tenant': self.get_str('CHROMA_TENANT', 'default'),
            'database': self.get_str('CHROMA_DATABASE', 'mutual-funds-db'),
            'host': self.get_str('CHROMA_HOST', 'https://api.trychroma.com'),
            'enabled': self.get_bool('ENABLE_CHROMA_CLOUD', True)
        }
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return {
            'level': self.get_str('LOG_LEVEL', 'INFO'),
            'file': self.get_str('LOG_FILE'),
            'debug_mode': self.get_bool('DEBUG_MODE', False)
        }
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration"""
        return {
            'max_concurrent_threads': self.get_int('MAX_CONCURRENT_THREADS', 4),
            'default_batch_size': self.get_int('DEFAULT_BATCH_SIZE', 32),
            'memory_limit_mb': self.get_int('MEMORY_LIMIT_MB', 4096)
        }
    
    def get_notification_config(self) -> Dict[str, Any]:
        """Get notification configuration"""
        return {
            'webhook_url': self.get_str('NOTIFICATION_WEBHOOK'),
            'email_enabled': self.get_bool('EMAIL_NOTIFICATIONS', False),
            'smtp_host': self.get_str('SMTP_HOST'),
            'smtp_port': self.get_int('SMTP_PORT', 587),
            'smtp_username': self.get_str('SMTP_USERNAME'),
            'smtp_password': self.get_str('SMTP_PASSWORD'),
            'notification_email': self.get_str('NOTIFICATION_EMAIL')
        }
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration"""
        return {
            'jwt_secret': self.get_str('JWT_SECRET'),
            'rate_limit_per_minute': self.get_int('RATE_LIMIT_PER_MINUTE', 100)
        }
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration"""
        return {
            'metrics_port': self.get_int('METRICS_PORT', 9090),
            'enable_metrics': self.get_bool('ENABLE_METRICS', False),
            'health_check_port': self.get_int('HEALTH_CHECK_PORT', 8080),
            'enable_health_check': self.get_bool('ENABLE_HEALTH_CHECK', True)
        }
    
    def get_development_config(self) -> Dict[str, Any]:
        """Get development configuration"""
        return {
            'debug_mode': self.get_bool('DEBUG_MODE', False),
            'enable_profiling': self.get_bool('ENABLE_PROFILING', False),
            'mock_external_services': self.get_bool('MOCK_EXTERNAL_SERVICES', True),
            'testing': self.get_bool('TESTING', False)
        }
    
    def get_external_api_config(self) -> Dict[str, Any]:
        """Get external API configuration"""
        return {
            'huggingface_api_key': self.get_str('HUGGINGFACE_API_KEY'),
            'alpha_vantage_api_key': self.get_str('ALPHA_VANTAGE_API_KEY'),
            'finnhub_api_key': self.get_str('FINNHUB_API_KEY'),
            'polygon_api_key': self.get_str('POLYGON_API_KEY')
        }
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration from environment variables"""
        return {
            'chroma_cloud': self.get_chroma_config(),
            'logging': self.get_logging_config(),
            'performance': self.get_performance_config(),
            'notifications': self.get_notification_config(),
            'security': self.get_security_config(),
            'monitoring': self.get_monitoring_config(),
            'development': self.get_development_config(),
            'external_apis': self.get_external_api_config()
        }
    
    def validate_chroma_config(self) -> bool:
        """Validate Chroma Cloud configuration"""
        chroma_config = self.get_chroma_config()
        
        if not chroma_config['enabled']:
            return True
        
        if not chroma_config['api_key']:
            self.logger.error("CHROMA_API_KEY is required when Chroma Cloud is enabled")
            return False
        
        if not chroma_config['host']:
            self.logger.error("CHROMA_HOST is required when Chroma Cloud is enabled")
            return False
        
        return True
    
    def mask_sensitive_values(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Mask sensitive values for logging"""
        sensitive_keys = [
            'api_key', 'password', 'secret', 'token', 'jwt_secret',
            'smtp_password', 'webhook_url', 'huggingface_api_key',
            'alpha_vantage_api_key', 'finnhub_api_key', 'polygon_api_key'
        ]
        
        def mask_recursive(obj):
            if isinstance(obj, dict):
                return {
                    k: mask_recursive(v) if not any(sensitive in k.lower() for sensitive in sensitive_keys) else '***MASKED***'
                    for k, v in obj.items()
                }
            elif isinstance(obj, list):
                return [mask_recursive(item) for item in obj]
            else:
                return obj
        
        return mask_recursive(config)
    
    def print_config_summary(self) -> None:
        """Print configuration summary (with masked sensitive values)"""
        config = self.get_all_config()
        masked_config = self.mask_sensitive_values(config)
        
        self.logger.info("Configuration Summary:")
        self.logger.info(f"  Chroma Cloud: {'Enabled' if masked_config['chroma_cloud']['enabled'] else 'Disabled'}")
        self.logger.info(f"  Log Level: {masked_config['logging']['level']}")
        self.logger.info(f"  Debug Mode: {masked_config['development']['debug_mode']}")
        self.logger.info(f"  Mock Services: {masked_config['development']['mock_external_services']}")
        self.logger.info(f"  Max Threads: {masked_config['performance']['max_concurrent_threads']}")
        self.logger.info(f"  Memory Limit: {masked_config['performance']['memory_limit_mb']}MB")


# Global instance for easy access
env_loader = EnvLoader()
