#!/usr/bin/env python3
"""
Test script to verify environment configuration
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from utils.env_loader import env_loader
    from utils.config_loader import ConfigLoader
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install required dependencies: pip install python-dotenv pyyaml")
    sys.exit(1)


def test_environment_configuration():
    """Test environment configuration loading"""
    print("Testing Environment Configuration")
    print("=" * 50)
    
    # Test basic environment loading
    print("\n1. Basic Environment Loading:")
    try:
        chroma_config = env_loader.get_chroma_config()
        print(f"   Chroma Cloud Enabled: {chroma_config['enabled']}")
        print(f"   Chroma Host: {chroma_config['host']}")
        print(f"   Chroma Tenant: {chroma_config['tenant']}")
        print(f"   API Key Set: {'Yes' if chroma_config['api_key'] else 'No'}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test performance configuration
    print("\n2. Performance Configuration:")
    try:
        perf_config = env_loader.get_performance_config()
        print(f"   Max Threads: {perf_config['max_concurrent_threads']}")
        print(f"   Batch Size: {perf_config['default_batch_size']}")
        print(f"   Memory Limit: {perf_config['memory_limit_mb']}MB")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test development configuration
    print("\n3. Development Configuration:")
    try:
        dev_config = env_loader.get_development_config()
        print(f"   Debug Mode: {dev_config['debug_mode']}")
        print(f"   Mock Services: {dev_config['mock_external_services']}")
        print(f"   Testing Mode: {dev_config['testing']}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test configuration validation
    print("\n4. Configuration Validation:")
    try:
        is_valid = env_loader.validate_chroma_config()
        print(f"   Chroma Config Valid: {is_valid}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test YAML config integration
    print("\n5. YAML Configuration Integration:")
    try:
        config = ConfigLoader.load_config("config/multi_model_config.yaml")
        print(f"   Config Loaded: Yes")
        print(f"   Multi-Model Strategy: {config.get('multi_model', {}).get('processing_strategy', 'unknown')}")
        print(f"   BGE-base Max URLs: {config.get('bge_base', {}).get('max_urls', 'unknown')}")
        print(f"   BGE-small Max URLs: {config.get('bge_small', {}).get('max_urls', 'unknown')}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test environment variable types
    print("\n6. Environment Variable Types:")
    try:
        # String
        test_str = env_loader.get_str('LOG_LEVEL', 'INFO')
        print(f"   String (LOG_LEVEL): {test_str}")
        
        # Integer
        test_int = env_loader.get_int('MAX_CONCURRENT_THREADS', 4)
        print(f"   Integer (MAX_CONCURRENT_THREADS): {test_int}")
        
        # Boolean
        test_bool = env_loader.get_bool('DEBUG_MODE', False)
        print(f"   Boolean (DEBUG_MODE): {test_bool}")
        
        # Float
        test_float = env_loader.get_float('MEMORY_LIMIT_MB', 4096.0)
        print(f"   Float (MEMORY_LIMIT_MB): {test_float}")
        
        # List
        test_list = env_loader.get_list('CUSTOM_VAR', ['default1', 'default2'])
        print(f"   List (CUSTOM_VAR): {test_list}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Print configuration summary
    print("\n7. Configuration Summary:")
    try:
        env_loader.print_config_summary()
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n" + "=" * 50)
    print("Environment Configuration Test Complete!")


def check_required_files():
    """Check if required files exist"""
    print("Checking Required Files")
    print("=" * 50)
    
    required_files = [
        '.env',
        '.env.example',
        'config/multi_model_config.yaml',
        'requirements.txt'
    ]
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"   {file_path}: EXISTS")
        else:
            print(f"   {file_path}: MISSING")
    
    print("\n" + "=" * 50)


def show_setup_instructions():
    """Show setup instructions if needed"""
    print("Setup Instructions")
    print("=" * 50)
    
    if not Path('.env').exists():
        print("\n1. Create .env file:")
        print("   cp .env.example .env")
    
    if not Path('config/multi_model_config.yaml').exists():
        print("\n2. Ensure config file exists:")
        print("   config/multi_model_config.yaml should exist")
    
    print("\n3. Install dependencies:")
    print("   pip install -r requirements.txt")
    
    print("\n4. Configure your API keys in .env:")
    print("   CHROMA_API_KEY=your_api_key_here")
    
    print("\n5. Run this test again to verify configuration")
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    print("Phase 4.3 Multi-Model Processing - Environment Configuration Test")
    print("=" * 70)
    
    # Check required files
    check_required_files()
    
    # Test configuration if files exist
    if Path('.env').exists():
        test_environment_configuration()
    else:
        show_setup_instructions()
    
    print("\nFor detailed setup instructions, see: docs/ENVIRONMENT_SETUP.md")
