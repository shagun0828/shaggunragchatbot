# Environment Setup Guide for Phase 4.3 Multi-Model Processing

This guide explains how to configure environment variables and API keys for the Phase 4.3 multi-model processing system.

## Quick Setup

1. **Copy the example environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit the `.env` file** with your actual API keys and configuration.

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Environment Variables

### Required Variables

#### Chroma Cloud Configuration
```env
# Get your API key from https://trychroma.com
CHROMA_API_KEY=your_chroma_cloud_api_key_here
CHROMA_TENANT=your_tenant_id_here
CHROMA_DATABASE=mutual-funds-db
CHROMA_HOST=https://api.trychroma.com
```

**How to get Chroma Cloud API key:**
1. Visit [trychroma.com](https://trychroma.com)
2. Sign up for a free account
3. Go to your dashboard
4. Copy your API key
5. Note your tenant ID (usually your username or organization name)

### Optional Variables

#### Sentence Transformers Configuration
```env
# Hugging Face API key (optional, for downloading models)
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# Model cache directory (optional)
TRANSFORMERS_CACHE=/path/to/model/cache
```

#### Notification Configuration
```env
# Webhook URL for notifications
NOTIFICATION_WEBHOOK=your_webhook_url_here

# Email notification settings
EMAIL_NOTIFICATIONS=false
SMTP_HOST=your_smtp_host_here
SMTP_PORT=587
SMTP_USERNAME=your_smtp_username_here
SMTP_PASSWORD=your_smtp_password_here
NOTIFICATION_EMAIL=your_notification_email_here
```

#### Performance Configuration
```env
# Processing settings
MAX_CONCURRENT_THREADS=4
DEFAULT_BATCH_SIZE=32
MEMORY_LIMIT_MB=4096
```

#### Development Configuration
```env
# Debug and testing
DEBUG_MODE=false
ENABLE_PROFILING=false
MOCK_EXTERNAL_SERVICES=true
TESTING=false
```

## API Key Sources

### 1. Chroma Cloud (Required)
- **Website**: https://trychroma.com
- **Purpose**: Vector database storage
- **Cost**: Free tier available
- **Setup**: Sign up and get API key from dashboard

### 2. Hugging Face (Optional)
- **Website**: https://huggingface.co
- **Purpose**: Download BGE models
- **Cost**: Free
- **Setup**: Create account and get API key from settings

### 3. Financial Data APIs (Optional)
```env
# Alpha Vantage
ALPHA_VANTAGE_API_KEY=your_key_here
# Website: https://www.alphavantage.co

# Finnhub
FINNHUB_API_KEY=your_key_here
# Website: https://finnhub.io

# Polygon.io
POLYGON_API_KEY=your_key_here
# Website: https://polygon.io
```

## Configuration Examples

### Development Environment
```env
# Development setup
DEBUG_MODE=true
MOCK_EXTERNAL_SERVICES=true
LOG_LEVEL=DEBUG
TESTING=true

# Use local Chroma (no API key needed)
ENABLE_CHROMA_CLOUD=false
```

### Production Environment
```env
# Production setup
DEBUG_MODE=false
MOCK_EXTERNAL_SERVICES=false
LOG_LEVEL=INFO
PRODUCTION=true

# Required for production
CHROMA_API_KEY=your_production_api_key
CHROMA_TENANT=your_production_tenant
```

### High-Performance Environment
```env
# Performance optimization
MAX_CONCURRENT_THREADS=8
DEFAULT_BATCH_SIZE=64
MEMORY_LIMIT_MB=8192

# Disable debug features
DEBUG_MODE=false
ENABLE_PROFILING=false
```

## Security Best Practices

### 1. Never Commit API Keys
```bash
# Add .env to .gitignore
echo ".env" >> .gitignore
echo ".env.example" >> .gitignore
```

### 2. Use Environment-Specific Files
```bash
# Development
cp .env.example .env.dev

# Staging
cp .env.example .env.staging

# Production
cp .env.example .env.prod
```

### 3. Rotate API Keys Regularly
- Change API keys every 90 days
- Update `.env` file immediately
- Restart the application

### 4. Use Strong Values
```env
# Use strong secrets
JWT_SECRET=your_very_long_and_random_secret_key_here_at_least_32_characters
```

## Validation

The system automatically validates required environment variables on startup:

### Required Variables Check
- `CHROMA_API_KEY` (if Chroma Cloud is enabled)
- `CHROMA_HOST` (if Chroma Cloud is enabled)

### Validation Messages
```
INFO: Loaded environment variables from .env
ERROR: Missing required environment variables: ['CHROMA_API_KEY']
```

## Troubleshooting

### Common Issues

#### 1. Chroma Cloud Connection Failed
```env
# Check these settings
CHROMA_API_KEY=verify_api_key_is_correct
CHROMA_TENANT=verify_tenant_is_correct
CHROMA_HOST=https://api.trychroma.com
```

#### 2. Model Download Issues
```env
# Set Hugging Face token (if private models)
HUGGINGFACE_API_KEY=your_huggingface_token

# Set cache directory with write permissions
TRANSFORMERS_CACHE=/path/to/writable/cache
```

#### 3. Memory Issues
```env
# Reduce memory usage
MEMORY_LIMIT_MB=2048
DEFAULT_BATCH_SIZE=16
MAX_CONCURRENT_THREADS=2
```

#### 4. Performance Issues
```env
# Optimize for performance
DEFAULT_BATCH_SIZE=64
MAX_CONCURRENT_THREADS=8
MEMORY_LIMIT_MB=8192
```

### Debug Mode
Enable debug mode to see detailed configuration:
```env
DEBUG_MODE=true
LOG_LEVEL=DEBUG
```

## Environment Variable Reference

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CHROMA_API_KEY` | string | - | Chroma Cloud API key |
| `CHROMA_TENANT` | string | default | Chroma Cloud tenant ID |
| `CHROMA_DATABASE` | string | mutual-funds-db | Chroma Cloud database name |
| `CHROMA_HOST` | string | https://api.trychroma.com | Chroma Cloud host URL |
| `HUGGINGFACE_API_KEY` | string | - | Hugging Face API key |
| `LOG_LEVEL` | string | INFO | Logging level |
| `DEBUG_MODE` | boolean | false | Enable debug mode |
| `MAX_CONCURRENT_THREADS` | integer | 4 | Maximum concurrent threads |
| `DEFAULT_BATCH_SIZE` | integer | 32 | Default batch size |
| `MEMORY_LIMIT_MB` | integer | 4096 | Memory limit in MB |
| `MOCK_EXTERNAL_SERVICES` | boolean | true | Mock external services |
| `TESTING` | boolean | false | Testing environment flag |

## Integration with Code

The environment variables are automatically loaded and integrated:

```python
from utils.env_loader import env_loader

# Get configuration
chroma_config = env_loader.get_chroma_config()
performance_config = env_loader.get_performance_config()

# Use in your code
api_key = env_loader.get_str('CHROMA_API_KEY')
batch_size = env_loader.get_int('DEFAULT_BATCH_SIZE')
debug_mode = env_loader.get_bool('DEBUG_MODE')
```

## Multiple Environments

For different deployment environments:

### Development
```bash
# Load development environment
export ENVIRONMENT=dev
python -c "from utils.env_loader import env_loader; env_loader.print_config_summary()"
```

### Production
```bash
# Load production environment
export ENVIRONMENT=prod
python src/main.py
```

### Docker
```dockerfile
# In Dockerfile
COPY .env.prod .env
ENV ENVIRONMENT=prod
```

## Next Steps

1. **Copy and configure** your `.env` file
2. **Install dependencies** with `pip install -r requirements.txt`
3. **Test the configuration** by running the system
4. **Monitor logs** for any configuration issues
5. **Adjust settings** based on your requirements

For more information, see the main documentation in the `docs/` directory.
