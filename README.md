# 🚀 Advanced RAG-Powered FAQ Chatbot with Pinecone & Azure OpenAI

A sophisticated chatbot API that combines **Retrieval-Augmented Generation (RAG)**, **Pinecone vector database**, and **Azure OpenAI GPT-4o** to provide intelligent, context-aware responses to FAQ queries.

## ✨ Features

### Core Capabilities
- 🧠 **RAG Implementation**: Advanced retrieval-augmented generation for accurate responses
- 🔍 **Semantic Search**: Pinecone-powered vector similarity search
- 📚 **Knowledge Base**: CSV-based FAQ data with intelligent chunking
- 🤖 **GPT-4o Integration**: Azure OpenAI for high-quality text generation
- 📊 **Course Generation**: AI-powered course lesson creation
- ❓ **MCQ Generator**: Multiple choice question generation
- 🎯 **Fine-tuning Ready**: Automated fine-tuning data preparation

### Technical Features
- ⚡ **FastAPI**: High-performance async API framework
- 🔐 **CORS Support**: Cross-origin resource sharing enabled
- 📈 **Analytics**: Built-in search and performance analytics
- 🛠️ **Monitoring**: Comprehensive health checks and metrics
- 🔄 **Batch Processing**: Efficient vector database operations
- 📋 **Data Validation**: Robust input validation and error handling

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI App   │────│  Azure OpenAI    │────│   User Query    │
│                 │    │   (GPT-4o +      │    │                 │
│  - RAG Chain    │    │   Embeddings)    │    │                 │
│  - Vector Store │    └──────────────────┘    └─────────────────┘
│  - QA System    │              │
└─────────────────┘              │
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌──────────────────┐
│  Pinecone DB    │    │   CSV Data       │
│  - Embeddings   │    │  - FAQ Entries   │
│  - Metadata     │    │  - Categories    │
│  - Similarity   │    │  - Descriptions  │
└─────────────────┘    └──────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Azure OpenAI account with GPT-4o access
- Pinecone account (API key provided)
- CSV data files: `new_features.csv`, `new_feature_categories.csv`

### 1. Installation

```bash
# Clone or download the project files
git clone <your-repo> # or download files
cd rag-faq-chatbot

# Run automated setup
python setup.py
```

### 2. Environment Configuration

Create `.env` file (automatically created from `.env.example`):

```env
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_KEY=your-azure-openai-api-key
AZURE_OPENAI_API_VERSION=version
AZURE_DEPLOYMENT_NAME=gpt
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-ada-002

# Pinecone (Pre-configured)
PINECONE_API_KEY=pcsk_key
```

### 3. Start the Application

```bash
# Using the main script
python main.py

# Or using the startup script
./start.sh

# The API will be available at: http://localhost:8001
```

## 📊 Data Requirements

### CSV File Structure

**new_features.csv**:
```csv
id,title,short_desc,desc,cat_id,sub_cat_name
1,"Feature Title","Brief description","Detailed description...",1,"Subcategory"
```

**new_feature_categories.csv**:
```csv
id,category_name
1,"Main Category"
```

## 🔌 API Endpoints

### Core Chat Endpoints

#### POST `/chat`
Advanced chat with RAG implementation
```json
{
  "message": "How does the new search feature work?",
  "use_rag": true
}
```

#### POST `/rag-search`
Direct vector database search
```json
{
  "query": "search functionality",
  "max_results": 5,
  "score_threshold": 0.7
}
```

### Content Generation

#### POST `/generate-course-lesson`
Generate comprehensive course lessons
```json
{
  "topic": "Machine Learning Basics",
  "difficulty_level": "intermediate",
  "lesson_duration": "60 minutes",
  "include_examples": true,
  "include_exercises": true
}
```

#### POST `/generate-mcq`
Generate multiple choice questions
```json
{
  "topic": "Python Programming",
  "number_of_questions": 10,
  "difficulty_level": "intermediate"
}
```

### System Management

#### GET `/health`
Comprehensive health check
```json
{
  "status": "healthy",
  "model": "gpt-4o",
  "vector_db": "Pinecone",
  "faq_ready": true,
  "vector_store_ready": true,
  "rag_ready": true
}
```

#### GET `/fine-tuning-data`
Get prepared fine-tuning data
```json
{
  "total_examples": 150,
  "sample_data": [...],
  "download_ready": true
}
```

## 🛠️ Configuration Options

### RAG Configuration
```python
CHUNK_SIZE = 1000          # Text chunk size for embeddings
CHUNK_OVERLAP = 200        # Overlap between chunks
MAX_RETRIEVAL_RESULTS = 5  # Maximum retrieved documents
SIMILARITY_THRESHOLD = 0.7 # Minimum similarity score
```

### Vector Database Settings
```python
PINECONE_DIMENSION = 1536  # Azure OpenAI embedding dimension
INDEX_METRIC = "cosine"    # Similarity metric
BATCH_SIZE = 100          # Batch size for vector operations
```

## 🔧 Advanced Usage

### Custom Vector Search
```python
from vector_db_utils import VectorDBManager

# Initialize manager
db_manager = VectorDBManager(pinecone_client, "faq-knowledge-base")

# Get index statistics
stats = db_manager.get_index_stats()

# Perform semantic search with filters
results = db_manager.semantic_search(
    query_vector=embedding,
    top_k=10,
    filter_metadata={"category": "features"}
)
```

### RAG Optimization
```python
from vector_db_utils import RAGOptimizer

optimizer = RAGOptimizer()

# Get search analytics
analytics = optimizer.get_search_analytics()

# Get query improvement suggestions
suggestions = optimizer.suggest_query_improvements(query, results)
```

### Embedding Quality Check
```python
from vector_db_utils import EmbeddingQualityChecker

checker = EmbeddingQualityChecker()

# Analyze embedding distribution
quality_report = checker.check_embedding_distribution(embeddings)

# Get preprocessing suggestions
suggestions = checker.suggest_preprocessing_improvements(texts)
```

## 📈 Monitoring & Analytics

### Built-in Metrics
- Query response times
- Retrieval accuracy scores
- Vector database statistics
- API endpoint usage
- Error rates and patterns

### Performance Optimization
- Automatic chunk size optimization
- Embedding quality analysis
- Query improvement suggestions
- Connection health monitoring

## 🔄 Fine-tuning Integration

### Data Preparation
The system automatically prepares fine-tuning data in OpenAI format:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful FAQ assistant..."},
    {"role": "user", "content": "What is feature X?"},
    {"role": "assistant", "content": "Feature X is..."}
  ]
}
```

### Download Fine-tuning Data
```bash
curl -o fine_tuning_data.jsonl http://localhost:8001/download-fine-tuning-data
```

## 🐛 Troubleshooting

### Common Issues

**1. Vector Database Connection Failed**
```bash
# Check Pinecone API key
python setup.py test

# Verify index exists
python -c "from pinecone import Pinecone; pc = Pinecone(api_key='your-key'); print(pc.list_indexes())"
```

**2. Azure OpenAI Authentication Error**
```bash
# Verify environment variables
python setup.py check

# Test connection manually
curl -H "api-key: YOUR_KEY" "https://YOUR_ENDPOINT/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-15-preview"
```

**3. CSV Data Loading Issues**
```bash
# Validate data structure
python -c "import pandas as pd; print(pd.read_csv('new_features.csv').columns.tolist())"
```

### Debug Mode
Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📝 API Documentation

Once running, visit:
- **Interactive Docs**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc
- **OpenAPI JSON**: http://localhost:8001/openapi.json

## 🔒 Security Considerations

- API keys stored in environment variables
- CORS configured for specific origins
- Input validation on all endpoints
- Rate limiting ready for production
- No sensitive data in logs

## 🚀 Deployment

### Production Deployment
```bash
# Using Gunicorn
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8001

# Using Docker
docker build -t rag-chatbot .
docker run -p 8001:8001 --env-file .env rag-chatbot
```

### Environment Variables for Production
```env
DEBUG_MODE=False
LOG_LEVEL=INFO
MAX_CONCURRENT_REQUESTS=50
REQUEST_TIMEOUT=60
ENABLE_ANALYTICS=True
```

## 📊 Performance Benchmarks

### Expected Performance
- **Response Time**: < 2 seconds for RAG queries
- **Throughput**: 100+ requests/minute
- **Vector Search**: < 100ms for similarity search
- **Memory Usage**: ~500MB for 10K documents

### Optimization Tips
1. Adjust chunk size based on document length
2. Use appropriate similarity thresholds
3. Implement caching for frequent queries
4. Monitor embedding quality regularly

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Azure OpenAI** for powerful language models
- **Pinecone** for vector database infrastructure
- **LangChain** for RAG framework
- **FastAPI** for high-performance API framework

## 📞 Support

For support and questions:

1. Check the troubleshooting section
2. Review API documentation at `/docs`
3. Run system check: `python setup.py check`
4. Enable debug logging for detailed error info

---
