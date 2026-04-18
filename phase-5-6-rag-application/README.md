# Phase 5-6 RAG Application

Advanced Retrieval-Augmented Generation system with multi-modal capabilities, real-time processing, and personalization features.

## Overview

Phase 5-6 implements the Application Layer and Advanced Features of the RAG architecture, providing a comprehensive solution for intelligent financial assistance and mutual fund information retrieval.

## Features

### Phase 5: Application Layer
- **REST API Gateway**: FastAPI-based endpoints for RAG operations
- **Chat Interface**: Real-time conversational AI with context management
- **Search Interface**: AI-enhanced search with semantic capabilities
- **Analytics Dashboard**: Comprehensive monitoring and metrics
- **GraphQL Support**: Flexible query interface
- **WebSocket Integration**: Real-time streaming responses

### Phase 6: Advanced Features
- **Multi-modal RAG**: Support for text, images, and structured data
- **Real-time Learning**: Continuous model improvement
- **Personalization**: User profiling and adaptive responses
- **Advanced Query Processing**: Query expansion and optimization
- **Intelligent Reranking**: Multiple ranking strategies
- **Webhook Integration**: Event-driven notifications

## Architecture

```
Phase 5-6 RAG Application
|
|-- src/
|   |-- api/                    # REST API endpoints
|   |   |-- rag_endpoints.py   # Core RAG functionality
|   |   |-- chat_endpoints.py  # Chat interface
|   |   |-- search_endpoints.py # Search interface
|   |   |-- monitoring_endpoints.py # System monitoring
|   |
|   |-- integration/           # External integrations
|   |   |-- chroma_client.py   # Vector database client
|   |   |-- llm_client.py      # Language model client
|   |
|   |-- advanced/              # Advanced processing
|   |   |-- query_processor.py # Query optimization
|   |   |-- reranker.py        # Result ranking
|   |   |-- context_manager.py # Conversation context
|   |
|   |-- personalization/       # User personalization
|   |   |-- user_profiler.py    # User profiling
|   |
|   |-- websocket/             # Real-time communication
|   |   |-- websocket_manager.py # WebSocket management
|   |
|   |-- graphql/               # GraphQL interface
|   |   |-- graphql_app.py      # GraphQL schema and resolvers
|   |
|   |-- monitoring/            # System monitoring
|   |   |-- metrics.py          # Metrics collection
|   |   |-- analytics.py        # Usage analytics
|   |
|   |-- ui/                    # Web interface
|   |   |-- static/
|   |   |   |-- chat.html       # Chat interface
|   |   |   |-- search.html     # Search interface
|   |   |   |-- dashboard.html   # Analytics dashboard
|   |
|   |-- multimodal/            # Multi-modal processing (Phase 6)
|   |-- webhooks/              # Webhook system (Phase 6)
|
|-- config/                    # Configuration files
|-- docs/                      # Documentation
|-- requirements.txt           # Dependencies
```

## Quick Start

### Prerequisites
- Python 3.11+
- Chroma Cloud account (for vector storage)
- OpenAI API key (for LLM integration)

### Installation

1. Clone the repository:
```bash
cd "c:\Users\LENOVO\Desktop\2nd LIP\phase-5-6-rag-application"
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

4. Run the application:
```bash
python src/main.py
```

The application will be available at `http://localhost:8000`

## API Documentation

### REST API Endpoints

#### RAG Operations
- `POST /api/v1/rag/query` - Process RAG query
- `POST /api/v1/rag/batch-query` - Batch query processing
- `POST /api/v1/rag/feedback` - Submit feedback
- `GET /api/v1/rag/similar/{query_id}` - Get similar queries

#### Chat Interface
- `POST /api/v1/chat/chat` - Send chat message
- `POST /api/v1/chat/stream` - Stream chat response
- `GET /api/v1/chat/history/{session_id}` - Get chat history
- `POST /api/v1/chat/feedback` - Submit chat feedback

#### Search Interface
- `POST /api/v1/search/search` - Perform search
- `POST /api/v1/search/advanced` - Advanced search
- `GET /api/v1/search/similar/{document_id}` - Find similar documents
- `GET /api/v1/search/autocomplete` - Get suggestions

#### Monitoring
- `GET /api/v1/monitoring/health` - System health check
- `GET /api/v1/monitoring/metrics/system` - System metrics
- `GET /api/v1/monitoring/analytics/usage` - Usage analytics
- `GET /api/v1/monitoring/alerts` - Active alerts

### GraphQL Interface

Access GraphQL Playground at `http://localhost:8000/graphql`

#### Sample Queries
```graphql
query SearchDocuments($input: SearchInput!) {
  searchDocuments(input: $input) {
    query
    searchType
    results {
      id
      text
      score
      metadata
    }
    totalResults
  }
}

mutation SubmitFeedback($input: FeedbackInput!) {
  submitFeedback(input: $input)
}
```

### WebSocket Interface

Connect to WebSocket at `ws://localhost:8000/ws`

#### Message Types
- `chat` - Send chat message
- `search` - Perform search
- `heartbeat` - Keep-alive ping
- `subscribe` - Subscribe to updates

## Configuration

### Environment Variables

```bash
# Chroma Cloud Configuration
CHROMA_API_KEY=your_chroma_api_key
CHROMA_TENANT=your_tenant
CHROMA_DATABASE=your_database
ENABLE_CHROMA_CLOUD=true

# LLM Configuration
OPENAI_API_KEY=your_openai_api_key
LLM_MODEL=gpt-4
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=1000

# Application Configuration
DEBUG=false
LOG_LEVEL=INFO
HOST=0.0.0.0
PORT=8000

# Monitoring Configuration
ENABLE_METRICS=true
METRICS_PORT=9090
HEALTH_CHECK_INTERVAL=30

# WebSocket Configuration
WEBSOCKET_HEARTBEAT_INTERVAL=30
WEBSOCKET_CONNECTION_TIMEOUT=3600

# Personalization Configuration
ENABLE_PERSONALIZATION=true
USER_PROFILE_TTL=86400
QUERY_HISTORY_LIMIT=100
```

## Advanced Features

### Multi-modal RAG

Support for processing various data types:
- Text documents
- Images and charts
- Structured financial data
- Audio transcripts

### Real-time Learning

Continuous improvement through:
- User feedback integration
- Query pattern analysis
- Performance optimization
- Model fine-tuning

### Personalization Engine

User-specific adaptation:
- Query history analysis
- Preference learning
- Context-aware responses
- Recommendation system

### Advanced Query Processing

Intelligent query handling:
- Semantic expansion
- Entity extraction
- Intent classification
- Query optimization

### Intelligent Reranking

Multiple ranking strategies:
- Cross-encoder models
- Learning-to-rank
- Maximal marginal relevance
- Quality-weighted ranking

## Monitoring and Analytics

### System Metrics
- CPU and memory usage
- Request throughput
- Response times
- Error rates
- Active connections

### Usage Analytics
- Query patterns
- User activity
- Popular content
- Performance trends

### Health Monitoring
- Component health checks
- Alert system
- Performance thresholds
- Automated notifications

## Development

### Running Tests
```bash
pytest src/tests/
```

### Code Quality
```bash
# Format code
black src/

# Lint code
flake8 src/

# Type checking
mypy src/
```

### Development Server
```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

## Deployment

### Docker Deployment
```bash
# Build image
docker build -t phase-5-6-rag .

# Run container
docker run -p 8000:8000 phase-5-6-rag
```

### Production Considerations
- Use HTTPS in production
- Configure proper CORS settings
- Set up monitoring and logging
- Implement rate limiting
- Use environment-specific configurations

## Performance Optimization

### Caching Strategy
- Query result caching
- User profile caching
- System metrics caching
- Response caching

### Scalability
- Horizontal scaling support
- Load balancing ready
- Database connection pooling
- Async processing

### Security
- API key management
- Input validation
- Rate limiting
- CORS protection

## Troubleshooting

### Common Issues

1. **Chroma Cloud Connection Issues**
   - Verify API key and credentials
   - Check network connectivity
   - Confirm tenant and database names

2. **LLM API Errors**
   - Check API key validity
   - Verify rate limits
   - Monitor token usage

3. **WebSocket Connection Issues**
   - Check firewall settings
   - Verify WebSocket support
   - Monitor connection timeouts

4. **Performance Issues**
   - Monitor system resources
   - Check query complexity
   - Review caching configuration

### Debug Mode
Enable debug logging:
```bash
export DEBUG=true
export LOG_LEVEL=DEBUG
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For support and questions:
- Check the documentation
- Review troubleshooting guide
- Contact the development team

## Roadmap

### Future Enhancements
- Advanced multi-modal processing
- Real-time model fine-tuning
- Enhanced personalization
- Expanded integrations
- Performance optimizations

### Phase 7 Planning
- Graph RAG integration
- Agentic capabilities
- Federated learning
- Quantum-enhanced retrieval

---

**Phase 5-6 RAG Application** - Advanced AI-powered financial assistance system
