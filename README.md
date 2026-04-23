# 🚀 Project Aegis: Advanced RAG System

An enterprise-grade Retrieval-Augmented Generation (RAG) system built for Google Colab that intelligently answers questions from corporate policy documents.

## ✨ Features

- **Smart Chunking**: Semantic chunking with 500 tokens per chunk and 15% overlap
- **LLM Metadata Extraction**: Automatic categorization and metadata tagging
- **Advanced Retrieval**:
  - Multi-query expansion (3-4 variations)
  - HyDE (Hypothetical Document Embeddings)
  - Category-based pre-filtering
  - Date-based post-filtering
  - Cross-encoder reranking (Cohere)
- **High-Quality Embeddings**: OpenAI text-embedding-3-large (3072 dimensions)
- **Source Citations**: Every answer includes references with relevance scores
- **Interactive Chat**: Built-in chat interface for continuous conversations

## 🏗️ Architecture

```
Documents (.txt) → Chunking → Metadata Extraction → Embeddings → Pinecone
                                                                      ↓
User Query → Query Expansion → Vector Search → Reranking → Generation
```

## 📋 Prerequisites

- Google Colab account
- API Keys:
  - [OpenAI API Key](https://platform.openai.com/api-keys)
  - [Pinecone API Key](https://app.pinecone.io/)
  - [Cohere API Key](https://dashboard.cohere.com/api-keys)
- Policy documents in `.txt` or `.md` format

## 🚀 Quick Start

### 1. Setup Files

Upload to your Google Drive at `/MyDrive/Project-Rag/`:
```
Project-Rag/
├── aegis_rag_helpers.py
└── Data/
    ├── security/
    ├── training/
    ├── travel/
    └── work_policies/
```

### 2. Configure Colab Secrets

In Google Colab:
1. Click the 🔑 **Secrets** icon in the left sidebar
2. Add three secrets:
   - `OPENAI_API_KEY`
   - `PINECONE_API_KEY`
   - `COHERE_API_KEY`
3. Enable **notebook access** for each

### 3. Run the Notebook

1. Upload `Project_Aegis_Clean.ipynb` to Google Colab
2. Run cells in order:
   - Install dependencies
   - Mount Google Drive
   - Load API keys
   - Import helper classes
   - Run ingestion (one time)
   - Initialize RAG system
   - Start querying!

## 💡 Usage Examples

### Basic Query
```python
result = rag_system.query("What is the travel reimbursement policy?")
print(result['answer'])
```

### With HyDE
```python
result = rag_system.query(
    "Can I expense a taxi?",
    top_k=5,
    use_hyde=True
)
```

### Interactive Chat
```python
rag_system.chat()
```

## 📊 System Components

### Ingestion Pipeline
- **MarkdownSemanticChunker**: Intelligent text chunking
- **MetadataExtractor**: LLM-powered metadata extraction
- **EmbeddingPipeline**: Batch embedding and Pinecone upload

### Retrieval Pipeline
- **QueryTransformer**: Multi-query expansion and HyDE
- **MetadataFilter**: Pre/post filtering by category and date
- **Reranker**: Cohere cross-encoder reranking
- **AdvancedRetriever**: Complete retrieval orchestration

### Generation
- **RAGGenerator**: Context-aware answer generation with citations
- **ProjectAegisRAG**: End-to-end system orchestrator

## 🔧 Configuration

Edit these variables in the notebook:

```python
DATA_DIR = "/content/drive/MyDrive/Project-Rag/Data"
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSION = 3072
INDEX_NAME = "project-aegis-rag"
LLM_MODEL = "gpt-4o-mini"
```

## 📁 File Structure

```
Project 1/
├── README.md
├── aegis_rag_helpers.py          # Core RAG components
├── Project_Aegis_Clean.ipynb     # Main notebook
└── Projectfile.txt               # Project guidelines
```

## 🧪 Testing

The notebook includes diagnostic cells for:
- Query expansion testing
- HyDE generation testing
- Category detection testing
- Performance metrics
- Batch query testing

## 🎯 Key Capabilities

- **Handles Vague Queries**: "What's the allowance?" → Expands to multiple specific variations
- **Preserves Context**: Tables and structured data remain intact
- **Smart Filtering**: Automatically detects policy category from queries
- **High Accuracy**: Cross-encoder reranking ensures top results are truly relevant
- **Transparent**: All answers include source citations and confidence scores

## 📚 Resources

- [Pinecone Documentation](https://docs.pinecone.io/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [Cohere Rerank Documentation](https://docs.cohere.com/docs/reranking)

## 🤝 Contributing

This is a project implementation based on the Project Aegis guidelines for building advanced RAG systems.

## 📄 License

MIT License - Feel free to use and modify for your projects.

---

**Built with ❤️ for enterprise policy management**
