# RAG (Retrieval-Augmented Generation) Architecture

## Overview

This document outlines a comprehensive RAG architecture designed to build intelligent, context-aware applications that combine the power of large language models with domain-specific knowledge retrieval.

## Architecture Components

### 1. Data Ingestion Layer

#### 1.1 Data Sources
- **Web Content**: Mutual fund data from Groww.in URLs:
  - https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth
  - https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth
  - https://groww.in/mutual-funds/hdfc-focused-fund-direct-growth
  - https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth
  - https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth
- **Structured Data**: APIs, CSV files, JSON documents
- **Unstructured Data**: Word documents, HTML pages, markdown files
- **Semi-structured Data**: XML, YAML, configuration files
- **Streaming Data**: Real-time feeds, logs, social media streams

#### 1.2 Data Preprocessing Pipeline
```
Raw Data -> Data Validation -> Text Extraction -> Content Cleaning -> 
Chunking -> Metadata Extraction -> Quality Control -> Vector Storage
```

**Key Components:**
- **Data Validators**: Schema validation, format checking
- **Web Scrapers**: HTML parsing, JavaScript rendering, anti-bot handling
- **Content Extractors**: Text extraction from web pages, data cleaning
- **Content Cleaners**: Remove HTML tags, normalize formatting, deduplication
- **Chunkers**: Semantic chunking, fixed-size chunking, recursive chunking
- **Metadata Extractors**: Fund details extraction, performance metrics, entity recognition
- **Quality Control**: Data validation, completeness checks

#### 1.3 Automated Data Scheduler
```
Daily 9:15 AM -> GitHub Actions Trigger -> Scraping Service -> 
Data Processing -> Vector Update -> System Notification
```

**GitHub Actions Scheduler Configuration:**
- **Cron Expression**: `15 9 * * *` (Every day at 9:15 AM IST)
- **Time Zone**: UTC-based execution with IST time conversion
- **Workflow File**: `.github/workflows/daily-scraping.yml`
- **Retry Mechanism**: Built-in GitHub Actions retry strategy
- **Failure Alerts**: GitHub Actions notifications + custom webhook alerts
- **Monitoring**: GitHub Actions logs + external monitoring service

**GitHub Actions Components:**
- **Scheduled Workflows**: Time-based GitHub Actions triggers
- **Self-hosted Runners**: Dedicated infrastructure for scraping tasks
- **Secrets Management**: GitHub Secrets for API keys and credentials
- **Artifact Storage**: Temporary storage for scraped data
- **Webhook Integration**: Custom notifications for success/failure

#### 1.4 Scraping Service Configuration
```
Groww URLs -> HTTP Requests -> HTML Parsing -> 
Data Extraction -> Structured Formatting -> Validation -> Storage
```

**Scraping Service Components:**
- **HTTP Client**: Async requests, retry mechanisms, rate limiting
- **HTML Parser**: BeautifulSoup, Scrapy, Playwright for dynamic content
- **Anti-Bot Handling**: User-agent rotation, proxy management, CAPTCHA handling
- **Data Extractors**: Fund performance metrics, NAV history, scheme details
- **Rate Limiting**: Respect robots.txt, implement delays between requests
- **Error Handling**: Network timeouts, parsing errors, fallback strategies
- **Data Validation**: Schema validation, completeness checks, data type verification

**Extracted Data Fields:**
- Fund basic information (name, category, risk level)
- Performance metrics (returns, NAV, expense ratio)
- Historical data (past returns, benchmarks)
- Scheme details (investment objective, asset allocation)
- Regulatory information (risk factors, tax implications)
- Market data (current NAV, daily changes, volume)

### 2. Vector Storage Layer

#### 2.1 Vector Database Options
- **Chroma Cloud**: Managed vector database service (trychroma.com), cloud-native, scalable
- **Pinecone**: Managed vector database, scalable, real-time
- **Weaviate**: GraphQL-based, semantic search capabilities
- **Chroma**: Open-source, lightweight, easy deployment
- **FAISS**: Facebook AI library, high-performance, local deployment
- **Milvus**: Open-source, distributed, cloud-native

#### 2.2 Chroma Cloud Integration
```
Local Processing -> BGE Embeddings -> Chroma Cloud API -> 
Vector Storage -> Cloud Indexing -> Global Accessibility
```

**Chroma Cloud Architecture:**
- **Cloud Service**: Managed vector database at trychroma.com
- **API Integration**: RESTful API for vector operations
- **Authentication**: API key-based authentication
- **Collection Management**: Organized collections for different data types
- **Global Access**: Accessible from anywhere with internet connection

**Chroma Cloud Components:**
```python
# Chroma Cloud Client Setup
import chromadb
from chromadb.config import Settings

class ChromaCloudManager:
    def __init__(self, api_key: str, tenant: str, database: str):
        self.client = chromadb.HttpClient(
            host="https://api.trychroma.com",
            settings=Settings(
                chroma_auth="chromadb.auth.TokenAuth",
                chroma_token=api_key,
                chroma_tenant=tenant,
                chroma_database=database
            )
        )
    
    async def upload_embeddings(self, embeddings: np.ndarray, 
                              documents: List[str], 
                              metadata: List[Dict]) -> str:
        """Upload embeddings to Chroma Cloud"""
        collection = self.client.get_or_create_collection("mutual_funds")
        
        # Add documents to collection
        collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadata,
            ids=[f"doc_{i}" for i in range(len(documents))]
        )
        
        return collection.id
```

**Chroma Cloud Features:**
- **Managed Service**: No infrastructure management required
- **Scalability**: Automatic scaling based on usage
- **High Availability**: Built-in redundancy and failover
- **Global CDN**: Fast access from anywhere
- **Security**: Enterprise-grade security and compliance
- **Monitoring**: Built-in analytics and monitoring

**Data Upload Process:**
1. **Local Processing**: Generate embeddings using BGE models locally
2. **Batch Preparation**: Prepare documents, embeddings, and metadata
3. **API Authentication**: Authenticate with Chroma Cloud API key
4. **Collection Creation**: Create or access collections in cloud
5. **Batch Upload**: Upload data in batches for efficiency
6. **Indexing**: Automatic indexing for fast retrieval
7. **Verification**: Confirm successful upload

**Chroma Cloud Configuration:**
```yaml
chroma_cloud:
  api_key: "${CHROMA_API_KEY}"
  tenant: "your-tenant-id"
  database: "mutual-funds-db"
  host: "https://api.trychroma.com"
  
  collections:
    mutual_funds: "mutual_funds_v1"
    financial_news: "financial_news_v1"
    market_data: "market_data_v1"
  
  upload_settings:
    batch_size: 100
    max_retries: 3
    timeout: 30.0
    retry_delay: 1.0
  
  indexing:
    index_type: "hnsw"
    ef_construction: 200
    ef_search: 50
    m: 16
```

#### 2.3 Embedding Strategy
- **Model Selection**: BGE-small-en-v1.5 (primary), Sentence-BERT, Cohere embeddings
- **Embedding Dimensions**: 384 dimensions (BGE-small-en-v1.5), 768, 1536 dimensions based on model
- **Batch Processing**: Efficient batch embedding for large datasets
- **Embedding Caching**: Local cache for frequently accessed embeddings
- **Local Processing**: No API costs, privacy-focused, fast inference

#### 2.3 Index Management
- **Index Types**: HNSW, IVF, LSH, Flat
- **Index Optimization**: Parameter tuning, memory management
- **Index Updates**: Incremental updates, full rebuilds
- **Multi-index Strategy**: Hybrid approaches for different data types

### 3. Retrieval Layer

#### 3.1 Query Processing
```
User Query -> Query Analysis -> Query Expansion -> 
Embedding Generation -> Vector Search -> Result Filtering
```

**Components:**
- **Query Analyzer**: Intent detection, entity extraction, query classification
- **Query Expander**: Synonym expansion, related concepts, contextual enhancement
- **Embedding Generator**: Real-time query embedding
- **Vector Searcher**: Similarity search, hybrid search capabilities

#### 3.2 Retrieval Strategies
- **Semantic Search**: Vector similarity based retrieval
- **Keyword Search**: Traditional BM25, TF-IDF
- **Hybrid Search**: Combination of semantic and keyword search
- **Multi-modal Retrieval**: Text, image, and audio retrieval
- **Temporal Retrieval**: Time-aware search, recency biasing

#### 3.3 Ranking and Filtering
- **Re-ranking Models**: Cross-encoders, learning-to-rank
- **Diversity Filters**: Maximal marginal relevance, topic diversity
- **Quality Filters**: Content quality scores, source reliability
- **Business Rules**: Access control, compliance filters

### 4. Generation Layer

#### 4.1 LLM Integration
- **Model Selection**: GPT-4, Claude, Llama, Mistral
- **Prompt Engineering**: Context formatting, instruction templates
- **Model Routing**: Specialized models for different tasks
- **Model Fine-tuning**: Domain-specific adaptation

#### 4.2 Context Management
```
Retrieved Documents -> Context Assembly -> 
Context Compression -> Prompt Construction -> LLM Generation
```

**Components:**
- **Context Assembler**: Document ordering, relevance grouping
- **Context Compressor**: Information density optimization, token management
- **Prompt Constructor**: Template filling, instruction embedding
- **Response Generator**: LLM inference, streaming responses

#### 4.3 Response Enhancement
- **Citation Generation**: Source attribution, reference linking
- **Confidence Scoring**: Answer confidence, source reliability
- **Fact Checking**: Verification against retrieved documents
- **Response Formatting**: Structured outputs, markdown, JSON

### 5. Application Layer

#### 5.1 API Gateway
- **REST APIs**: Standard RESTful endpoints
- **GraphQL**: Flexible query interfaces
- **WebSocket**: Real-time streaming responses
- **Webhooks**: Event-driven responses

#### 5.2 User Interface
- **Chat Interface**: Conversational AI interface
- **Search Interface**: Traditional search with AI enhancement
- **Dashboard**: Analytics and monitoring interface
- **Admin Panel**: System management and configuration

#### 5.3 Integration Points
- **CRM Systems**: Customer data integration
- **Document Management**: Enterprise document repositories
- **Communication Platforms**: Slack, Teams, email integration
- **Analytics Platforms**: Usage tracking, performance metrics

## Chunking Architecture

### Overview

The chunking architecture transforms raw scraped mutual fund data into optimized, semantically meaningful segments suitable for vector storage and retrieval.

### Chunking Pipeline

```
Raw Data -> Data Preprocessing -> Chunking Strategy -> 
Chunk Validation -> Metadata Enrichment -> Vector Storage
```

### Chunking Strategies

#### 1. Semantic Chunking
```python
class SemanticChunker:
    def __init__(self, embedding_model="bge-small-en-v1.5"):
        self.embedding_model = embedding_model
        self.similarity_threshold = 0.7
        self.max_chunk_size = 1000
        self.min_chunk_size = 100
    
    def chunk_semantic(self, text: str) -> List[Chunk]:
        """Split text based on semantic similarity"""
        sentences = self.split_sentences(text)
        embeddings = self.generate_embeddings(sentences)
        
        chunks = []
        current_chunk = []
        current_embedding = None
        
        for i, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
            if not current_chunk:
                current_chunk.append(sentence)
                current_embedding = embedding
            else:
                similarity = cosine_similarity(current_embedding, embedding)
                if similarity >= self.similarity_threshold and \
                   len(" ".join(current_chunk + [sentence])) <= self.max_chunk_size:
                    current_chunk.append(sentence)
                    current_embedding = self.average_embeddings(current_chunk, embeddings[i-len(current_chunk)+1:i+1])
                else:
                    chunks.append(self.create_chunk(" ".join(current_chunk)))
                    current_chunk = [sentence]
                    current_embedding = embedding
        
        if current_chunk:
            chunks.append(self.create_chunk(" ".join(current_chunk)))
        
        return chunks
```

#### 2. Fixed-Size Chunking
```python
class FixedSizeChunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_fixed(self, text: str) -> List[Chunk]:
        """Split text into fixed-size chunks with overlap"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            if chunk_words:
                chunk_text = " ".join(chunk_words)
                chunks.append(self.create_chunk(chunk_text, start_pos=i))
        
        return chunks
```

#### 3. Recursive Character Splitting
```python
class RecursiveCharacterSplitter:
    def __init__(self, separators: List[str] = None):
        self.separators = separators or ["\n\n", "\n", ".", "!", "?", " ", ""]
        self.max_chunk_size = 1000
    
    def chunk_recursive(self, text: str) -> List[Chunk]:
        """Recursively split text based on separators"""
        return self._recursive_split(text, self.separators)
    
    def _recursive_split(self, text: str, separators: List[str]) -> List[Chunk]:
        if not separators or len(text) <= self.max_chunk_size:
            return [self.create_chunk(text)]
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        splits = text.split(separator)
        chunks = []
        
        for split in splits:
            if len(split) <= self.max_chunk_size:
                chunks.append(self.create_chunk(split.strip()))
            else:
                chunks.extend(self._recursive_split(split, remaining_separators))
        
        return chunks
```

### Mutual Fund Specific Chunking

#### Fund Data Structure Awareness
```python
class MutualFundChunker:
    def __init__(self):
        self.semantic_chunker = SemanticChunker()
        self.fixed_chunker = FixedSizeChunker()
        self.fund_sections = {
            'basic_info': ['fund_name', 'category', 'risk_level', 'aum'],
            'performance': ['returns_1y', 'returns_3y', 'returns_5y', 'nav'],
            'allocation': ['equity', 'debt', 'cash', 'sector_allocation'],
            'holdings': ['top_holdings', 'portfolio_composition'],
            'metadata': ['expense_ratio', 'fund_manager', 'inception_date']
        }
    
    def chunk_fund_data(self, fund_data: Dict) -> List[Chunk]:
        """Chunk mutual fund data with domain awareness"""
        chunks = []
        
        # Create structured chunks for each section
        for section, fields in self.fund_sections.items():
            section_text = self.extract_section_text(fund_data, section, fields)
            if section_text:
                chunk = self.create_structured_chunk(
                    text=section_text,
                    section=section,
                    fund_name=fund_data.get('fund_name'),
                    metadata={'section_type': section, 'fields': fields}
                )
                chunks.append(chunk)
        
        # Create narrative chunks for descriptive content
        narrative_text = self.extract_narrative_content(fund_data)
        if narrative_text:
            semantic_chunks = self.semantic_chunker.chunk_semantic(narrative_text)
            chunks.extend(semantic_chunks)
        
        return chunks
```

### Chunk Quality Control

#### Validation Rules
```python
class ChunkValidator:
    def __init__(self):
        self.min_length = 50
        self.max_length = 2000
        self.required_metadata = ['fund_name', 'section_type']
    
    def validate_chunk(self, chunk: Chunk) -> bool:
        """Validate chunk quality"""
        # Length validation
        if len(chunk.text) < self.min_length or len(chunk.text) > self.max_length:
            return False
        
        # Content validation
        if not self.has_meaningful_content(chunk.text):
            return False
        
        # Metadata validation
        if not all(field in chunk.metadata for field in self.required_metadata):
            return False
        
        return True
    
    def has_meaningful_content(self, text: str) -> bool:
        """Check if chunk contains meaningful content"""
        # Remove common non-meaningful patterns
        text_lower = text.lower()
        meaningless_patterns = ['click here', 'read more', 'view details', 'n/a']
        
        return not any(pattern in text_lower for pattern in meaningless_patterns)
```

## Embedding Architecture

### Overview

The embedding architecture converts processed chunks into vector representations that capture semantic meaning for efficient similarity search and retrieval.

### Embedding Pipeline

```
Validated Chunks -> Text Preprocessing -> Embedding Generation -> 
Vector Normalization -> Quality Check -> Vector Storage
```

### Embedding Models

#### 1. BGE-small-en-v1.5 (Primary Model)
```python
class BGEEmbedder:
    def __init__(self, model_name: str = "bge-small-en-v1.5"):
        self.model = SentenceTransformer(model_name)
        self.dimension = 384
        self.batch_size = 32
    
    def generate_embeddings(self, chunks: List[Chunk]) -> np.ndarray:
        """Generate embeddings using BGE-small-en-v1.5"""
        texts = [chunk.text for chunk in chunks]
        
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self.model.encode(
                batch_texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        embedding = self.model.encode(
            text,
            normalize_embeddings=True
        )
        return embedding
```

#### 2. Sentence Transformers (Alternative)
```python
class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.batch_size = 32
    
    def generate_embeddings(self, chunks: List[Chunk]) -> np.ndarray:
        """Generate embeddings for chunks using sentence transformers"""
        texts = [chunk.text for chunk in chunks]
        
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self.model.encode(
                batch_texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
```

#### 3. OpenAI Embeddings (Optional)
```python
class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small"):
        self.client = openai.OpenAI()
        self.model = model
        self.dimension = 1536 if "small" in model else 3072
        self.max_tokens = 8191
    
    def generate_embeddings(self, chunks: List[Chunk]) -> np.ndarray:
        """Generate embeddings using OpenAI API"""
        texts = [self.truncate_text(chunk.text) for chunk in chunks]
        
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch_texts
            )
            batch_embeddings = [data.embedding for data in response.data]
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    def truncate_text(self, text: str) -> str:
        """Truncate text to fit within token limits"""
        tokens = text.split()
        if len(tokens) <= self.max_tokens:
            return text
        return " ".join(tokens[:self.max_tokens])
```

### Domain-Specific Embeddings

#### Financial Domain Adaptation
```python
class FinancialEmbedder:
    def __init__(self, base_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.base_model = SentenceTransformer(base_model)
        self.financial_terms = self.load_financial_vocabulary()
        self.domain_weights = self.calculate_domain_weights()
    
    def enhance_financial_embeddings(self, chunks: List[Chunk]) -> np.ndarray:
        """Enhance embeddings for financial domain"""
        base_embeddings = self.base_model.encode([chunk.text for chunk in chunks])
        
        enhanced_embeddings = []
        for i, chunk in enumerate(chunks):
            financial_score = self.calculate_financial_relevance(chunk.text)
            enhanced_embedding = base_embeddings[i] * (1 + financial_score * self.domain_weights)
            enhanced_embeddings.append(enhanced_embedding)
        
        return np.array(enhanced_embeddings)
    
    def calculate_financial_relevance(self, text: str) -> float:
        """Calculate financial domain relevance score"""
        text_lower = text.lower()
        financial_keywords = ['nav', 'returns', 'aum', 'expense ratio', 'fund', 'investment']
        
        relevance = 0.0
        for keyword in financial_keywords:
            if keyword in text_lower:
                relevance += 0.1
        
        return min(relevance, 1.0)
```

### Embedding Quality Assurance

#### Similarity Validation
```python
class EmbeddingQualityChecker:
    def __init__(self, similarity_threshold: float = 0.95):
        self.similarity_threshold = similarity_threshold
    
    def check_embedding_quality(self, embeddings: np.ndarray, chunks: List[Chunk]) -> Dict:
        """Check embedding quality and identify issues"""
        quality_report = {
            'duplicate_embeddings': [],
            'low_variance_embeddings': [],
            'outlier_embeddings': [],
            'similarity_matrix': None
        }
        
        # Check for duplicate embeddings
        similarity_matrix = cosine_similarity(embeddings)
        quality_report['similarity_matrix'] = similarity_matrix
        
        # Find highly similar pairs (potential duplicates)
        for i in range(len(similarity_matrix)):
            for j in range(i + 1, len(similarity_matrix)):
                if similarity_matrix[i][j] > self.similarity_threshold:
                    quality_report['duplicate_embeddings'].append((i, j))
        
        # Check for low variance embeddings
        embedding_variance = np.var(embeddings, axis=1)
        low_variance_threshold = np.percentile(embedding_variance, 10)
        quality_report['low_variance_embeddings'] = np.where(embedding_variance < low_variance_threshold)[0].tolist()
        
        return quality_report
```

### Vector Storage Integration

#### Batch Processing
```python
class VectorStorageManager:
    def __init__(self, vector_db_client):
        self.vector_db = vector_db_client
        self.batch_size = 100
    
    def store_embeddings(self, chunks: List[Chunk], embeddings: np.ndarray) -> bool:
        """Store chunks and embeddings in vector database"""
        try:
            for i in range(0, len(chunks), self.batch_size):
                batch_chunks = chunks[i:i + self.batch_size]
                batch_embeddings = embeddings[i:i + self.batch_size]
                
                self.vector_db.upsert(
                    vectors=self.prepare_vectors(batch_chunks, batch_embeddings),
                    namespace="mutual_funds"
                )
            
            return True
        except Exception as e:
            logger.error(f"Failed to store embeddings: {e}")
            return False
    
    def prepare_vectors(self, chunks: List[Chunk], embeddings: np.ndarray) -> List[Dict]:
        """Prepare vectors for database insertion"""
        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            vector = {
                'id': chunk.id,
                'values': embedding.tolist(),
                'metadata': {
                    'text': chunk.text,
                    'fund_name': chunk.metadata.get('fund_name'),
                    'section_type': chunk.metadata.get('section_type'),
                    'chunk_type': chunk.metadata.get('chunk_type'),
                    'created_at': datetime.utcnow().isoformat()
                }
            }
            vectors.append(vector)
        
        return vectors
```

## Implementation Details

### GitHub Actions Scheduler Implementation

#### Workflow Configuration
```yaml
# .github/workflows/daily-scraping.yml
name: Daily Mutual Fund Scraping

on:
  schedule:
    # Runs daily at 9:15 AM IST (3:45 AM UTC)
    - cron: '45 3 * * *'
  workflow_dispatch: # Allow manual triggering

jobs:
  scrape-mutual-funds:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    
    steps:
    - name: Checkout repository
      uses: actions/checkout@v3
      
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        
    - name: Run scraping service
      env:
        GROWW_API_KEY: ${{ secrets.GROWW_API_KEY }}
        VECTOR_DB_URL: ${{ secrets.VECTOR_DB_URL }}
        NOTIFICATION_WEBHOOK: ${{ secrets.NOTIFICATION_WEBHOOK }}
      run: python -m scraping_service.main
      
    - name: Upload artifacts
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: scraped-data
        path: data/
        retention-days: 7
```

#### Docker Container for Scraping
```dockerfile
# Dockerfile.scraper
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/

CMD ["python", "-m", "src.main"]
```

### Scraping Service Implementation

#### Service Architecture
```python
class MutualFundScrapingService:
    def __init__(self):
        self.urls = [
            'https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth',
            'https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth',
            'https://groww.in/mutual-funds/hdfc-focused-fund-direct-growth',
            'https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth',
            'https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth'
        ]
        self.session = aiohttp.ClientSession()
        self.extractor = FundDataExtractor()
    
    async def run_daily_scraping(self):
        """Main scraping task executed by scheduler"""
        try:
            results = await self.scrape_all_funds()
            await self.process_and_store(results)
            await self.update_vectors(results)
            await self.send_success_notification()
        except Exception as e:
            await self.handle_scraping_error(e)
    
    async def scrape_all_funds(self):
        """Scrape data from all configured URLs"""
        tasks = [self.scrape_single_fund(url) for url in self.urls]
        return await asyncio.gather(*tasks, return_exceptions=True)
```

#### Data Processing Pipeline
```python
class DataProcessingPipeline:
    def __init__(self):
        self.validator = DataValidator()
        self.cleaner = DataCleaner()
        self.chunker = SemanticChunker()
        self.embedder = EmbeddingGenerator()
    
    async def process_scraped_data(self, raw_data):
        """Process scraped data through the pipeline"""
        # Validate data structure
        validated_data = await self.validator.validate(raw_data)
        
        # Clean and normalize
        cleaned_data = await self.cleaner.clean(validated_data)
        
        # Create semantic chunks
        chunks = await self.chunker.chunk(cleaned_data)
        
        # Generate embeddings
        embeddings = await self.embedder.generate(chunks)
        
        return {
            'chunks': chunks,
            'embeddings': embeddings,
            'metadata': cleaned_data.get('metadata', {})
        }
```

## Technical Implementation

### Technology Stack

#### Backend Technologies
- **Python**: Primary development language
- **FastAPI**: API framework
- **LangChain**: RAG framework
- **Haystack**: Alternative RAG framework
- **BeautifulSoup4**: HTML parsing for web scraping
- **Scrapy**: Web scraping framework
- **Playwright**: Dynamic content rendering
- **Sentence Transformers**: Local embedding generation
- **NumPy**: Numerical computations for embeddings
- **AsyncIO**: Asynchronous programming

#### CI/CD & Scheduling
- **GitHub Actions**: Automated workflows and scheduling
- **GitHub Secrets**: Secure credential management
- **GitHub Artifacts**: Temporary data storage
- **Docker**: Containerization for reproducible environments
- **GitHub CLI**: Automation and management

#### Database Technologies
- **PostgreSQL**: Metadata storage
- **Redis**: Caching and session management
- **Elasticsearch**: Full-text search
- **Vector Database**: Chosen based on requirements

#### Infrastructure
- **Docker**: Containerization
- **Kubernetes**: Orchestration
- **AWS/Azure/GCP**: Cloud infrastructure
- **Terraform**: Infrastructure as code

### Security Considerations

#### Data Security
- **Encryption**: Data at rest and in transit
- **Access Control**: Role-based permissions
- **Data Masking**: Sensitive information protection
- **Audit Logging**: Comprehensive activity tracking

#### Privacy Protection
- **PII Detection**: Personal information identification
- **Data Anonymization**: Privacy-preserving techniques
- **Compliance**: GDPR, CCPA, HIPAA compliance
- **Data Governance**: Data lifecycle management

### Performance Optimization

#### Caching Strategy
- **Query Caching**: Frequent query results
- **Embedding Caching**: Computed embeddings
- **Response Caching**: Generated responses
- **CDN Integration**: Global content delivery

#### Scalability Design
- **Horizontal Scaling**: Load distribution
- **Microservices**: Modular architecture
- **Queue Management**: Request throttling
- **Resource Management**: Dynamic allocation

### Monitoring and Analytics

#### System Monitoring
- **Performance Metrics**: Response times, throughput
- **Error Tracking**: Exception monitoring
- **Resource Monitoring**: CPU, memory, storage
- **Health Checks**: System availability

#### Usage Analytics
- **Query Analysis**: Popular queries, failure patterns
- **User Behavior**: Interaction patterns, satisfaction metrics
- **Content Analytics**: Document usage, relevance scores
- **Business Metrics**: ROI, cost per query

## Deployment Architecture

### Development Environment
```
Local Development -> Docker Compose -> 
Staging Environment -> Production Deployment
```

### Production Deployment
- **Multi-region Deployment**: Geographic distribution
- **Blue-Green Deployment**: Zero-downtime updates
- **Canary Releases**: Gradual rollout
- **Disaster Recovery**: Backup and restoration

### CI/CD Pipeline
- **Automated Testing**: Unit, integration, end-to-end tests
- **Code Quality**: Linting, security scanning
- **Automated Deployment**: Pipeline-driven releases
- **Rollback Mechanisms**: Quick recovery options

## Best Practices

### Data Management
- **Data Quality**: Regular validation and cleaning
- **Version Control**: Document versioning
- **Backup Strategy**: Regular backups and testing
- **Data Lineage**: Track data transformations

### Model Management
- **Model Versioning**: Track model iterations
- **A/B Testing**: Compare model performance
- **Model Monitoring**: Performance degradation detection
- **Model Retraining**: Regular updates with new data

### Operational Excellence
- **Documentation**: Comprehensive system documentation
- **Error Handling**: Graceful failure management
- **Performance Tuning**: Regular optimization
- **Security Updates**: Regular patching and updates

## Future Enhancements

### Advanced Features
- **Multi-modal RAG**: Image, video, and audio processing
- **Real-time Learning**: Continuous model improvement
- **Personalization**: User-specific adaptation
- **Collaborative Filtering**: Community-based recommendations

### Emerging Technologies
- **Graph RAG**: Knowledge graph integration
- **Agentic RAG**: Autonomous agent capabilities
- **Federated Learning**: Privacy-preserving training
- **Quantum Computing**: Future quantum-enhanced retrieval

## Conclusion

This RAG architecture provides a comprehensive foundation for building intelligent, scalable, and secure applications that leverage the power of retrieval-augmented generation. The modular design allows for flexibility and adaptation to specific use cases while maintaining best practices for performance, security, and maintainability.

The architecture supports various deployment scenarios from small-scale prototypes to enterprise-grade production systems, ensuring scalability and reliability as requirements evolve.
