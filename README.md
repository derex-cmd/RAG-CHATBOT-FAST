# FAST NUCES AI Assistant

A Retrieval-Augmented Generation (RAG) system designed for academic institutions, featuring advanced query processing, conversational memory, and real-time analytics.

## Overview

This AI Assistant leverages state-of-the-art natural language processing and machine learning technologies to provide intelligent, context-aware responses to university-related queries. Built with production-ready architecture and enterprise-level features, it serves as a comprehensive solution for educational institutions seeking to implement AI-powered student services.

## Key Features

### Advanced RAG Architecture
- **Multi-query retrieval** with query expansion and reranking
- **Conversational memory** for context-aware interactions
- **Document chunking** with optimized overlap strategies
- **Similarity threshold filtering** for relevant content retrieval

### Enterprise-Grade Components
- **Vector database integration** using ChromaDB for scalable document storage
- **GPU acceleration** support for improved performance
- **Real-time analytics** with comprehensive metrics tracking
- **Production-ready deployment** with automated setup scripts

### Intelligent Query Processing
- **Query enhancement** with typo correction and domain-specific expansion
- **Context-aware responses** leveraging conversation history
- **Confidence scoring** for response quality assessment
- **Fallback mechanisms** ensuring system reliability

### Document Management
- **Multi-format support** for PDF, DOCX, TXT, and Markdown files
- **Batch processing** for directory-based document ingestion
- **Metadata preservation** for source attribution
- **Incremental updates** without full database rebuilds

## Technical Architecture

### Core Technologies
- **Framework**: Streamlit for web interface
- **Language Model**: FLAN-T5 Base with HuggingFace Transformers
- **Embeddings**: MPNet Base v2 for semantic understanding
- **Vector Database**: ChromaDB for document storage and retrieval
- **Text Processing**: NLTK for natural language operations

### System Requirements
- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- GPU support optional but recommended for enhanced performance
- 5GB available disk space

## Installation

### Quick Start
```bash
# Clone the repository
git clone https://github.com/your-organization/fast-nuces-ai-assistant.git
cd fast-nuces-ai-assistant

# Run automated setup
python deploy.py

# Alternative quick start
python start.py
```

### Manual Installation
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download language data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Launch application
streamlit run streamlit_app.py
```

### Platform-Specific Launchers
- **Windows**: Double-click `start.bat`
- **Linux/macOS**: Execute `./start.sh`

## Configuration

### Environment Variables
```bash
# Model Configuration
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
LLM_MODEL=google/flan-t5-base
CHUNK_SIZE=512
CHUNK_OVERLAP=50

# Database Configuration
CHROMA_PATH=./chroma_db
COLLECTION_NAME=pdf_text_data

# Performance Settings
USE_GPU=true
CACHE_SIZE=100
```

### System Settings
The application uses a centralized configuration system located in `config.py`. Key parameters include:

- **Model Settings**: Token limits, temperature, and penalty configurations
- **Retrieval Parameters**: Similarity thresholds and context length limits
- **UI Customization**: Branding, themes, and interface preferences
- **Performance Tuning**: GPU utilization and caching strategies

## Document Management

### Adding Documents
```bash
# Interactive document ingestion
python add_documents.py

# Command-line usage
python -c "
from add_documents import DocumentIngestion
ingestion = DocumentIngestion()
ingestion.load_directory('path/to/documents')
"
```

### Supported Formats
- **PDF Files**: Full text extraction with page metadata
- **Microsoft Word**: DOCX format with paragraph structure preservation
- **Text Files**: Plain text and Markdown with encoding detection
- **Batch Processing**: Recursive directory scanning with file type filtering

### Database Operations
- **Incremental Updates**: Add new documents without affecting existing data
- **Full Replacement**: Complete database refresh with new document sets
- **Metadata Tracking**: Source attribution and page number preservation
- **Storage Optimization**: Efficient vector storage with compression

## System Testing

### Comprehensive Validation
```bash
# Run full system test
python test_system.py

# Environment validation only
python -c "from utils import validate_environment; print(validate_environment())"
```

### Test Coverage
- **Dependency Verification**: All required packages and versions
- **Hardware Assessment**: GPU availability and system specifications
- **File Integrity**: Core application files and configurations
- **Network Connectivity**: External service dependencies
- **Runtime Validation**: Component initialization and functionality

## Production Deployment

### Scalability Considerations
- **Horizontal Scaling**: Multi-instance deployment with load balancing
- **Database Optimization**: Vector index tuning for large document collections
- **Memory Management**: Efficient model loading and cache utilization
- **Performance Monitoring**: Real-time metrics and alerting systems

### Security Features
- **Input Validation**: Query sanitization and injection prevention
- **Access Control**: Session management and authentication hooks
- **Data Privacy**: Local processing with no external data transmission
- **Audit Logging**: Comprehensive activity tracking and reporting

### Monitoring and Analytics
- **Response Time Tracking**: Performance metrics with historical trends
- **Usage Analytics**: Query patterns and popular topics identification
- **System Health**: Component status and error rate monitoring
- **Export Capabilities**: Data export for external analysis and reporting

## API Reference

### Core Classes

#### UltimateRAGSystem
Primary system interface for query processing and response generation.

```python
from rag_system import UltimateRAGSystem

# Initialize system
rag = UltimateRAGSystem()

# Process query with metadata
response, metadata = rag.process_query_ultimate("Your question here")
```

#### DocumentIngestion
Document processing and database management functionality.

```python
from add_documents import DocumentIngestion

# Initialize ingestion system
ingestion = DocumentIngestion()

# Process single file
documents = ingestion.load_single_file("document.pdf")

# Process directory
documents = ingestion.load_directory("documents/")

# Add to database
ingestion.add_documents_to_vectorstore(documents)
```

### Configuration Management
Centralized configuration system with environment variable support.

```python
from config import config

# Access configuration
model_name = config.model.llm_model
chunk_size = config.model.chunk_size
database_path = config.database.chroma_path
```

## Troubleshooting

### Common Issues

**Installation Failures**
- Verify Python version compatibility (3.8+)
- Ensure sufficient disk space (5GB minimum)
- Check internet connectivity for package downloads

**Performance Issues**
- Monitor system memory usage during operation
- Verify GPU drivers if using CUDA acceleration
- Adjust chunk size and retrieval parameters for optimization

**Database Problems**
- Confirm ChromaDB directory permissions
- Validate document formats and encoding
- Check available disk space for vector storage

### Diagnostic Tools
```bash
# System validation
python test_system.py

# Environment check
python -c "from utils import validate_environment; print(validate_environment())"

# Dependency verification
python -c "from utils import check_dependencies; check_dependencies()"
```

## Contributing

### Development Setup
1. Fork the repository and create a feature branch
2. Install development dependencies: `pip install -r requirements-dev.txt`
3. Run tests: `python test_system.py`
4. Follow code style guidelines and add appropriate documentation
5. Submit pull request with comprehensive description

### Code Standards
- **Documentation**: Comprehensive docstrings for all functions and classes
- **Type Hints**: Full type annotation for better code clarity
- **Error Handling**: Robust exception management with informative messages
- **Testing**: Unit tests for new functionality and regression prevention

## License

This project is licensed under the MIT License. See the LICENSE file for detailed terms and conditions.

## Support

### Documentation
- **Installation Guide**: Comprehensive setup instructions for all platforms
- **Configuration Reference**: Detailed parameter descriptions and examples
- **API Documentation**: Complete function and class references
- **Troubleshooting Guide**: Common issues and resolution procedures

### Community Resources
- **Issue Tracking**: GitHub Issues for bug reports and feature requests
- **Discussion Forums**: Community support and knowledge sharing
- **Contributing Guidelines**: Development workflow and code standards
- **Release Notes**: Version history and feature announcements

## Acknowledgments

Built with enterprise-grade technologies and best practices for production deployment in educational environments. Special recognition to the open-source communities behind Streamlit, HuggingFace Transformers, LangChain, and ChromaDB for their foundational contributions to the AI and machine learning ecosystem.
