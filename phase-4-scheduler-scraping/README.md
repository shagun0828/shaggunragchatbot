# Phase 4.0: Scheduler and Scraping Service

This phase implements the automated scheduler and scraping service for mutual fund data from Groww.in URLs.

## Overview

The service automatically scrapes mutual fund data daily at 9:15 AM IST using GitHub Actions, processes the data into chunks, generates embeddings, and stores them in a vector database.

## Architecture

### Components

1. **GitHub Actions Scheduler**: Daily automated execution at 9:15 AM IST
2. **Web Scraper**: Extracts data from Groww.in mutual fund pages
3. **Data Processor**: Cleans, validates, and chunks the data
4. **Embedding Generator**: Creates vector embeddings for semantic search
5. **Vector Storage**: Stores embeddings in vector database or locally

### Folder Structure

```
phase-4-scheduler-scraping/
âââ .github/workflows/
â   âââ daily-scraping.yml          # GitHub Actions workflow
âââ src/
â   âââ main.py                    # Main entry point
â   âââ scrapers/
â   â   âââ mutual_fund_scraper.py # Web scraper for Groww URLs
â   âââ processors/
â   â   âââ data_processor.py      # Data processing pipeline
â   âââ chunkers/
â   â   âââ semantic_chunker.py    # Semantic text chunking
â   â   âââ mutual_fund_chunker.py # Domain-specific chunking
â   â   âââ fixed_size_chunker.py  # Fixed-size chunking
â   âââ embedders/
â   â   âââ sentence_transformer_embedder.py  # Local embeddings
â   â   âââ openai_embedder.py     # OpenAI embeddings
â   â   âââ financial_embedder.py  # Financial domain enhancement
â   âââ storage/
â   â   âââ vector_storage.py      # Vector database management
â   âââ models/
â   â   âââ chunk.py              # Chunk data model
â   âââ utils/
â       âââ logger.py             # Logging utilities
â       âââ notifications.py      # Notification management
â       âââ rate_limiter.py       # Rate limiting for requests
â       âââ user_agents.py        # User agent rotation
âââ config/
âââ data/
â   âââ raw/
â   âââ processed/
â   âââ embeddings/
âââ tests/
âââ requirements.txt
âââ Dockerfile
âââ README.md
```

## Target URLs

The service scrapes the following Groww.in mutual fund URLs:

1. https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth
2. https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth
3. https://groww.in/mutual-funds/hdfc-focused-fund-direct-growth
4. https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth
5. https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth

## Setup

### Environment Variables

Configure the following environment variables:

```bash
# Vector Database (optional)
VECTOR_DB_URL=your_vector_db_url
VECTOR_DB_API_KEY=your_vector_db_api_key

# OpenAI API (optional, for enhanced embeddings)
OPENAI_API_KEY=your_openai_api_key

# Notifications
NOTIFICATION_WEBHOOK=your_webhook_url

# Service Configuration
ENVIRONMENT=production
```

### GitHub Secrets

Add the following secrets to your GitHub repository:

- `VECTOR_DB_URL`: Vector database connection URL
- `VECTOR_DB_API_KEY`: Vector database API key
- `OPENAI_API_KEY`: OpenAI API key (optional)
- `NOTIFICATION_WEBHOOK`: Webhook URL for notifications

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python -m src.main
```

### Docker Deployment

```bash
# Build Docker image
docker build -t mutual-fund-scraper .

# Run container
docker run --env-file .env mutual-fund-scraper
```

## Data Extraction

The scraper extracts the following mutual fund data:

### Basic Information
- Fund name
- Category
- Risk level
- Assets Under Management (AUM)

### Performance Metrics
- Current NAV
- Historical returns (1Y, 3Y, 5Y)
- Expense ratio
- Exit load

### Portfolio Details
- Top holdings
- Sector allocation
- Asset allocation (equity/debt/cash)

### Fund Management
- Fund manager
- Inception date
- Minimum investment
- Description

## Processing Pipeline

1. **Data Cleaning**: Standardizes numeric fields and removes noise
2. **Validation**: Ensures data quality and completeness
3. **Chunking**: Creates semantic and structured chunks
4. **Embedding**: Generates vector representations
5. **Storage**: Stores in vector database or locally

## Chunking Strategy

### Structured Chunks
- Basic information chunk
- Performance metrics chunk
- Portfolio holdings chunk
- Sector allocation chunk
- Fund management chunk

### Narrative Chunks
- Semantic chunks from fund description
- Context-aware text segmentation

## Embedding Options

### Local Embeddings (Default)
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Dimension: 384
- No API costs

### OpenAI Embeddings (Optional)
- Model: `text-embedding-3-small`
- Dimension: 1536
- Requires API key

### Financial Domain Enhancement
- Boosts financial terminology relevance
- Improves semantic search for financial queries

## Monitoring

### Logs
- Structured JSON logging
- Performance metrics
- Error tracking

### Notifications
- Success/failure alerts via webhook
- Daily processing summaries
- Error notifications

### Data Quality
- Validation scores for each fund
- Chunk quality metrics
- Embedding quality checks

## GitHub Actions Schedule

The workflow runs daily at 9:15 AM IST (3:45 AM UTC):

```yaml
schedule:
  - cron: '45 3 * * *'
```

Manual triggering is also supported via `workflow_dispatch`.

## Output Files

### Processed Data
- `data/processed/{fund_name}_{timestamp}.json`: Individual fund data
- `data/processed/{fund_name}_chunks_{timestamp}.json`: Chunk data
- `data/processed/summary_{timestamp}.json`: Processing summary

### Embeddings
- `data/embeddings/embeddings_{timestamp}.npy`: Embedding arrays
- `data/embeddings/metadata_{timestamp}.json`: Chunk metadata

### Artifacts
- GitHub Actions artifacts store all data for 7 days
- Automatic cleanup after retention period

## Error Handling

### Retry Logic
- HTTP request retries with exponential backoff
- Failed chunk processing isolation
- Partial success handling

### Graceful Degradation
- Continues processing if some funds fail
- Local storage fallback for vector database
- Notification on partial failures

## Configuration

### Rate Limiting
- 2 requests per second default
- Configurable delays
- Respect robots.txt

### User Agent Rotation
- Multiple browser user agents
- Automatic rotation
- Anti-bot detection

### Data Validation
- Required field checks
- Data type validation
- Quality scoring

## Next Phases

This phase provides the data foundation for:
- Phase 5: Generation layer with LLM integration
- Phase 6: Application layer with user interfaces
- Retrieval system integration
- Real-time query processing
