# Document-Based Question Answering System

A state-of-the-art Machine Learning model for document-based question answering systems with Retrieval Augmented Generation (RAG) architecture and consistent SCODi 4P design.

The system uses Natural Language Processing (NLP) and advanced embedding techniques to answer questions based on provided documents.

## Features

The system offers the following main features:

1. **Document-Based Question Answering System**:
   - Process documents in various formats (PDF, DOCX, TXT, MD, HTML)
   - Split documents into sections and chunks with semantic boundaries
   - Semantic search for relevant passages using vector embeddings
   - Generate answers to questions based on document content with source attribution

2. **Advanced ML Capabilities**:
   - Retrieval Augmented Generation (RAG) architecture
   - Dense vector representations with SentenceTransformer models
   - Cross-encoder reranking for improved retrieval precision
   - Multi-query variations for better recall
   - Semantic document chunking with context preservation

3. **Web Interface**:
   - Intuitive user interface with SCODi 4P design
   - File upload and management
   - Visualization of results with source attribution
   - Customizable settings

## Design System

The project uses the SCODi 4P design system with the following components:

- Color scheme based on SCODi Corporate Identity
- Responsive layout and components
- Consistent typography and iconography
- Modern navigation and footer

### Colors

- **Primary color**: #007f78 (Dark green/teal)
- **Secondary color**: #4b5864 (Dark gray)
- **Accent color**: #f7f7f7 (Light gray for backgrounds)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python Package Manager)

### Installing Dependencies

1. Clone the repository:
   ```bash
   git clone https://github.com/thenzler/document-based-qa-model.git
   cd document-based-qa-model
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt  # For the basic functions
   ```

### Getting Started

1. Start the application:
   ```bash
   python modern_app.py
   ```

2. Open a browser and navigate to:
   ```
   http://localhost:5000
   ```

3. Follow the instructions on the home page to initialize the system and upload documents.

## Usage

### Document-Based Question Answering

1. Navigate to the "Question & Answer" section
2. Enter a question based on your documents
3. The system searches for relevant passages and generates an answer with source references

### Document Management

1. Navigate to the "Documents" section
2. Upload new documents (PDF, DOCX, TXT, MD, HTML)
3. Adjust settings for document processing

## Architecture

The system is based on a modular architecture with the following components:

1. **Document Processing** (`DocumentProcessor`):
   - Text extraction from various document formats
   - Semantic chunking with context preservation
   - Storage and management of processed documents

2. **Question Answering System** (`DocumentQA` with RAG architecture):
   - Semantic search for relevant passages using vector embeddings
   - Cross-encoder reranking for more precise retrieval
   - Answer generation based on documents with source attribution

3. **Web Interface**:
   - Flask-based web interface with SCODi 4P design
   - Responsive design
   - Interactive components with JavaScript

## Machine Learning Components

### Vector Embeddings

The system uses SentenceTransformer models to create dense vector representations of document chunks and queries, enabling semantic similarity search.

```python
# From qa_system_rag.py
self.embedding_model = SentenceTransformer(embedding_model_name)
```

### Vector Database

Facebook AI Similarity Search (FAISS) is used for efficient vector similarity search, enabling fast retrieval of relevant document passages.

```python
# From qa_system_rag.py
embedding_dim = self.chunk_embeddings.shape[1]
self.faiss_index = faiss.IndexFlatL2(embedding_dim)
self.faiss_index.add(self.chunk_embeddings)
```

### Cross-Encoder Reranking

A two-stage retrieval approach with bi-encoder + cross-encoder reranking improves retrieval precision.

```python
# From qa_system_rag.py
self.cross_encoder = CrossEncoder(cross_encoder_model_name)
```

### Semantic Document Chunking

The system includes intelligent document chunking that preserves contextual boundaries and semantic coherence.

```python
# From qa_system_rag.py
def _create_semantic_chunks(self, text, chunk_size=1000, overlap=200):
    # Semantic chunking implementation
```

## Configuration

Configuration parameters can be adjusted in the `modern_app.py` file.

## Contributing

Contributions to the project are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is released under the MIT License - see LICENSE file for details.
