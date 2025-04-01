"""
Document-Based QA System with Retrieval Augmented Generation (RAG)
=================================================================

An enhanced implementation of the document-based QA system using modern
RAG (Retrieval Augmented Generation) architecture.
"""

import os
import json
import time
import re
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional
import logging

# Initialize logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RAG-QA")

# Sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("SentenceTransformers not available. Basic retrieval will be used.")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# FAISS for efficient vector search
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    logger.warning("FAISS not available. Basic retrieval will be used.")
    FAISS_AVAILABLE = False

# Transformers for local model support
try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForQuestionAnswering, 
        AutoModelForSeq2SeqLM,
        pipeline
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("Transformers not available. Basic retrieval will be used.")
    TRANSFORMERS_AVAILABLE = False

# Import embedding models
try:
    from src.model_embeddings import EmbeddingModel, CrossEncoderModel
    EMBEDDING_MODELS_AVAILABLE = True
except ImportError:
    logger.warning("Embedding models not available. Using built-in embeddings.")
    EMBEDDING_MODELS_AVAILABLE = False

class DocumentQA:
    """
    Enhanced DocumentQA class using Retrieval Augmented Generation (RAG)
    for improved document-based question answering
    """
    
    def __init__(
        self, 
        embedding_model_name="sentence-transformers/all-mpnet-base-v2", 
        cross_encoder_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        generation_model_name="deepset/roberta-base-squad2",
        use_gpu=False
    ):
        """
        Initialize the RAG-based QA system with improved models
        
        Args:
            embedding_model_name (str): Name of the bi-encoder embedding model
            cross_encoder_model_name (str): Name of the cross-encoder for reranking
            generation_model_name (str): Name of the QA/generation model
            use_gpu (bool): Whether to use GPU acceleration if available
        """
        self.documents = []
        self.chunks = []
        self.chunk_embeddings = None
        self.faiss_index = None
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Model initialization flags
        self.using_external_embedding_model = EMBEDDING_MODELS_AVAILABLE
        self.using_sentence_transformers = SENTENCE_TRANSFORMERS_AVAILABLE and not EMBEDDING_MODELS_AVAILABLE
        self.using_transformers = TRANSFORMERS_AVAILABLE
        self.using_faiss = FAISS_AVAILABLE
        
        # Initialize models
        try:
            # Use external embedding models if available
            if self.using_external_embedding_model:
                logger.info("Using external embedding models")
                self.embedding_model = EmbeddingModel(embedding_model_name, use_gpu=use_gpu)
                self.cross_encoder = CrossEncoderModel(cross_encoder_model_name, use_gpu=use_gpu)
            # Otherwise use sentence transformers directly
            elif self.using_sentence_transformers:
                logger.info(f"Loading embedding model: {embedding_model_name}")
                self.embedding_model = SentenceTransformer(embedding_model_name)
                if self.use_gpu:
                    self.embedding_model = self.embedding_model.to(torch.device("cuda"))
                
                logger.info(f"Loading cross-encoder model: {cross_encoder_model_name}")
                self.cross_encoder = CrossEncoder(cross_encoder_model_name)
            else:
                logger.warning("No embedding models available. Using basic retrieval.")
                self.embedding_model = None
                self.cross_encoder = None
            
            # Load QA model if transformers is available
            if self.using_transformers:
                logger.info(f"Loading generation model: {generation_model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(generation_model_name)
                self.qa_model = AutoModelForQuestionAnswering.from_pretrained(generation_model_name)
                
                if self.use_gpu:
                    self.qa_model = self.qa_model.to(torch.device("cuda"))
                
                self.qa_pipeline = pipeline(
                    "question-answering",
                    model=self.qa_model,
                    tokenizer=self.tokenizer,
                    device=0 if self.use_gpu else -1
                )
            else:
                logger.warning("Transformers not available. Using extractive answers only.")
                self.tokenizer = None
                self.qa_model = None
                self.qa_pipeline = None
            
            logger.info("Models successfully loaded")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            logger.info("Falling back to simpler methods if models could not be loaded")
            self.embedding_model = None
            self.cross_encoder = None
            self.qa_model = None
            self.tokenizer = None
            self.qa_pipeline = None
    
    def process_documents(self, docs_dir):
        """
        Process all documents in the specified directory with improved chunking
        
        Args:
            docs_dir (str): Path to the document directory
        """
        docs_path = Path(docs_dir)
        
        if not docs_path.exists():
            logger.warning(f"Directory not found: {docs_dir}")
            return
        
        # Load all documents and chunks
        self.documents = []
        self.chunks = []
        
        # Load metadata if already processed
        metadata_dir = docs_path / '.metadata'
        chunks_dir = docs_path / '.chunks'
        
        if metadata_dir.exists() and chunks_dir.exists():
            # Try to load existing processed data
            try:
                self._load_processed_data(metadata_dir, chunks_dir)
            except Exception as e:
                logger.error(f"Error loading processed data: {e}")
                logger.info("Processing documents again...")
                self._process_raw_documents(docs_path)
        else:
            # Create .metadata and .chunks directories if they don't exist
            metadata_dir.mkdir(exist_ok=True)
            chunks_dir.mkdir(exist_ok=True)
            
            # Process documents from scratch
            self._process_raw_documents(docs_path)
            
        # Create embeddings and index for semantic search
        if self._can_use_embeddings():
            self._create_embeddings_index()
        
        logger.info(f"Loaded: {len(self.documents)} documents, {len(self.chunks)} chunks")
    
    def _can_use_embeddings(self):
        """Check if embeddings can be used"""
        if self.using_external_embedding_model:
            return True
        elif self.using_sentence_transformers and self.embedding_model is not None:
            return True
        return False
    
    def _load_processed_data(self, metadata_dir, chunks_dir):
        """Load already processed documents and chunks"""
        # Load metadata
        for metadata_file in metadata_dir.glob('*.meta.json'):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    self.documents.append(metadata)
            except Exception as e:
                logger.error(f"Error loading metadata {metadata_file}: {e}")
        
        # Load chunks
        for chunks_file in chunks_dir.glob('*.chunks.json'):
            try:
                with open(chunks_file, 'r', encoding='utf-8') as f:
                    doc_chunks = json.load(f)
                    # Add document information
                    doc_name = chunks_file.stem.replace('.chunks', '')
                    doc_metadata = next((doc for doc in self.documents if doc.get('filename', '').startswith(doc_name)), None)
                    
                    for chunk in doc_chunks:
                        if doc_metadata:
                            chunk['document'] = {
                                'filename': doc_metadata.get('filename', ''),
                                'category': doc_metadata.get('category', 'general')
                            }
                        self.chunks.append(chunk)
            except Exception as e:
                logger.error(f"Error loading chunks {chunks_file}: {e}")
    
    def _process_raw_documents(self, docs_path):
        """Process raw documents and create chunks"""
        # Search for and process documents
        for file_path in docs_path.glob('*.*'):
            if file_path.suffix.lower() in ['.txt', '.md', '.pdf', '.docx', '.html']:
                try:
                    logger.info(f"Processing document: {file_path.name}")
                    document_meta = self._process_document(file_path)
                    
                    if document_meta:
                        self.documents.append(document_meta)
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path.name}: {e}")
    
    def _process_document(self, file_path):
        """
        Process a single document with improved semantic chunking
        
        Args:
            file_path (Path): Path to the document file
            
        Returns:
            dict: Document metadata
        """
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return None
            
        # Read document content
        content = self._read_document_content(file_path)
        
        if not content:
            logger.warning(f"No content found in {file_path.name} or unsupported format")
            return None
            
        # Create document metadata
        metadata = {
            "id": str(hash(str(file_path))),
            "filename": file_path.name,
            "file_type": file_path.suffix.lower()[1:],
            "category": self._guess_category(file_path),
            "size": file_path.stat().st_size,
            "upload_date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(file_path.stat().st_mtime)),
            "processed_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "chunk_size": 1000,
            "overlap": 200
        }
        
        # Split content into chunks with improved semantic boundaries
        chunks = self._create_semantic_chunks(content, chunk_size=1000, overlap=200)
        
        # Save metadata
        metadata_path = file_path.parent / '.metadata' / f"{file_path.stem}.meta.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
            
        # Save chunks
        chunks_with_doc_info = []
        for chunk in chunks:
            chunk_with_doc = dict(chunk)
            chunk_with_doc['document'] = {
                'filename': file_path.name,
                'category': metadata['category']
            }
            chunks_with_doc_info.append(chunk_with_doc)
            self.chunks.append(chunk_with_doc)
            
        chunks_path = file_path.parent / '.chunks' / f"{file_path.stem}.chunks.json"
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
            
        return metadata
        
    def _read_document_content(self, file_path):
        """Read document content based on file type with improved format handling"""
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == '.txt' or suffix == '.md':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
                    
            elif suffix == '.html':
                # Simple HTML extraction - in a full implementation, use an HTML parser
                with open(file_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    # Remove HTML tags
                    return re.sub(r'<[^>]+>', ' ', html_content)
                    
            elif suffix == '.pdf':
                logger.info("PDF extraction requires pdfplumber or PyPDF2, not supported in this version")
                return f"[PDF content from {file_path.name}]"
                
            elif suffix == '.docx':
                logger.info("DOCX extraction requires python-docx, not supported in this version")
                return f"[DOCX content from {file_path.name}]"
            
            return None
            
        except Exception as e:
            logger.error(f"Error reading {file_path.name}: {e}")
            return None
    
    def _create_semantic_chunks(self, text, chunk_size=1000, overlap=200):
        """
        Split text into chunks with improved semantic boundary detection
        
        Args:
            text (str): Text to split
            chunk_size (int): Maximum chunk size in characters
            overlap (int): Overlap between chunks in characters
            
        Returns:
            list: List of chunk objects
        """
        chunks = []
        
        if not text:
            return chunks
            
        # Split text into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_chunk = ""
        current_size = 0
        
        for paragraph in paragraphs:
            paragraph_size = len(paragraph)
            
            # If adding this paragraph would exceed the chunk size
            if current_size + paragraph_size > chunk_size and current_chunk:
                # Store current chunk
                chunks.append({
                    "text": current_chunk.strip(),
                    "start_char": len(chunks) * (chunk_size - overlap) if chunks else 0,
                    "end_char": len(chunks) * (chunk_size - overlap) + len(current_chunk) if chunks else len(current_chunk)
                })
                
                # Start new chunk with appropriate overlap
                if overlap > 0:
                    # Try to find a good sentence boundary for the overlap
                    sentences = re.split(r'(?<=[.!?])\s+', current_chunk)
                    overlap_text = ""
                    
                    # Build overlap text from the end of the previous chunk
                    for sent in reversed(sentences):
                        if len(overlap_text) + len(sent) + 1 <= overlap:
                            overlap_text = sent + " " + overlap_text
                        else:
                            break
                    
                    current_chunk = overlap_text + paragraph + "\n\n"
                    current_size = len(current_chunk)
                else:
                    current_chunk = paragraph + "\n\n"
                    current_size = paragraph_size + 2
            else:
                # Add paragraph to current chunk
                current_chunk += paragraph + "\n\n"
                current_size += paragraph_size + 2
                
                # If current chunk is big enough to store
                if current_size >= chunk_size:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "start_char": len(chunks) * (chunk_size - overlap) if chunks else 0,
                        "end_char": len(chunks) * (chunk_size - overlap) + len(current_chunk) if chunks else len(current_chunk)
                    })
                    current_chunk = ""
                    current_size = 0
        
        # Store final chunk if it exists
        if current_chunk:
            chunks.append({
                "text": current_chunk.strip(),
                "start_char": len(chunks) * (chunk_size - overlap) if chunks else 0,
                "end_char": len(chunks) * (chunk_size - overlap) + len(current_chunk) if chunks else len(current_chunk)
            })
            
        return chunks
    
    def _guess_category(self, file_path):
        """Attempt to determine document category from filename"""
        filename = file_path.name.lower()
        
        if 'readme' in filename or 'installation' in filename or 'setup' in filename:
            return 'documentation'
        elif 'product' in filename or 'produkt' in filename:
            return 'product'
        elif 'company' in filename or 'unternehmen' in filename:
            return 'company'
        elif 'kunden' in filename or 'customer' in filename:
            return 'customer'
        else:
            return 'general'
    
    def _create_embeddings_index(self):
        """Create embeddings for all chunks and a FAISS index for fast search"""
        # Check if embeddings can be used
        if not self._can_use_embeddings():
            logger.warning("Embedding capabilities not available, skipping indexing")
            return
            
        logger.info("Creating embeddings for all chunks...")
        
        # Extract texts from chunks
        texts = [chunk.get('text', '') for chunk in self.chunks]
        
        if not texts:
            logger.warning("No texts found to embed")
            return
            
        # Create embeddings for all texts
        try:
            if self.using_external_embedding_model:
                self.chunk_embeddings = self.embedding_model.get_embeddings(texts)
            else:
                self.chunk_embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            
            # Create FAISS index for fast nearest-neighbor search
            if self.using_faiss:
                embedding_dim = self.chunk_embeddings.shape[1]
                self.faiss_index = faiss.IndexFlatL2(embedding_dim)
                self.faiss_index.add(self.chunk_embeddings)
                
            logger.info(f"Embeddings index created with {len(texts)} chunks")
        except Exception as e:
            logger.error(f"Error creating embeddings index: {e}")
            self.chunk_embeddings = None
            self.faiss_index = None

    def add_document(self, file_path):
        """
        Add a single document to the system
        
        Args:
            file_path (str): Path to the document file
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return
        
        try:
            # Process document
            document_meta = self._process_document(file_path)
            
            if not document_meta:
                logger.warning(f"Document could not be processed: {file_path}")
                return
                
            # Add to document list
            self.documents.append(document_meta)
            
            # Update embeddings and index
            if self._can_use_embeddings() and self.chunk_embeddings is not None:
                # Get only the newly added chunks
                new_chunks = [c for c in self.chunks if c.get('document', {}).get('filename', '') == file_path.name]
                
                if new_chunks:
                    new_texts = [chunk.get('text', '') for chunk in new_chunks]
                    
                    if self.using_external_embedding_model:
                        new_embeddings = self.embedding_model.get_embeddings(new_texts)
                    else:
                        new_embeddings = self.embedding_model.encode(new_texts, show_progress_bar=True)
                    
                    # Add new embeddings to index
                    if self.using_faiss and self.faiss_index is not None:
                        self.faiss_index.add(new_embeddings)
                    
                    # Update the embeddings array
                    if self.chunk_embeddings is not None:
                        self.chunk_embeddings = np.vstack([self.chunk_embeddings, new_embeddings])
                    else:
                        self.chunk_embeddings = new_embeddings
            
            logger.info(f"Document successfully added and indexed: {file_path}")
        except Exception as e:
            logger.error(f"Error adding document {file_path}: {e}")
    
    def answer_question(self, question, use_generation=True, top_k=5):
        """
        Answer a question using the RAG approach
        
        Args:
            question (str): The question to answer
            use_generation (bool): Whether to generate a narrative answer
            top_k (int): Number of documents to consider
            
        Returns:
            str: The answer to the question
            list: The sources used
        """
        # Check if documents are loaded
        if not self.chunks:
            return "Please upload documents first to answer questions.", []
        
        # Create multiple query variations for better retrieval
        query_variations = self._generate_query_variations(question)
        
        # Search for relevant chunks based on all query variations
        relevant_chunks = []
        for query in query_variations:
            chunks = self._search_relevant_chunks(query, top_k)
            relevant_chunks.extend(chunks)
        
        # Remove duplicates and keep the top-k chunks
        unique_chunks = []
        chunk_ids = set()
        for chunk in relevant_chunks:
            chunk_id = chunk.get('text', '')[:100]  # Use first 100 chars as ID
            if chunk_id not in chunk_ids:
                chunk_ids.add(chunk_id)
                unique_chunks.append(chunk)
        
        relevant_chunks = unique_chunks[:top_k]
        
        if not relevant_chunks:
            return (f"Sorry, couldn't find relevant information for '{question}'. "
                   "Please try a different question or upload more relevant documents."), []
        
        # If cross-encoder is available, rerank chunks
        if (self.using_external_embedding_model and self.cross_encoder) or \
           (self.using_sentence_transformers and self.cross_encoder is not None):
            relevant_chunks = self._rerank_chunks(question, relevant_chunks)
        
        # Generate answer using the most relevant chunks
        if use_generation and self.qa_pipeline is not None:
            return self._generate_rag_answer(question, relevant_chunks)
        else:
            return self._generate_extractive_answer(question, relevant_chunks)
    
    def _generate_query_variations(self, question, num_variations=3):
        """
        Generate variations of the query for better retrieval
        
        Args:
            question (str): Original question
            num_variations (int): Number of variations to generate
            
        Returns:
            list: List of query variations
        """
        # Simple rule-based variations (in a real implementation, use an LLM)
        variations = [question]
        
        # Convert question to statement
        if question.lower().startswith('what is'):
            variations.append(question.lower().replace('what is', '').strip())
        elif question.lower().startswith('how to'):
            variations.append(question.lower().replace('how to', '').strip())
        elif question.lower().startswith('why'):
            variations.append(question.lower().replace('why', '').strip())
        
        # Add keywords only variation
        keywords = self._extract_keywords(question)
        if keywords:
            variations.append(' '.join(keywords))
            
        return variations[:num_variations]
    
    def _extract_keywords(self, text):
        """Extract keywords from text"""
        # Remove special characters and split into words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Remove stopwords (in a real implementation, use a more comprehensive list)
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                     'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'like', 
                     'through', 'over', 'before', 'after', 'between', 'under'}
        keywords = [word for word in words if word not in stopwords and len(word) > 2]
        
        return keywords
    
    def _search_relevant_chunks(self, question, top_k=5):
        """
        Search for the most relevant chunks for a question
        
        Args:
            question (str): The question
            top_k (int): Number of chunks to return
            
        Returns:
            list: The most relevant chunks
        """
        relevant_chunks = []
        
        # If semantic search is available (with embeddings)
        if self._can_use_embeddings() and self.chunk_embeddings is not None:
            try:
                # Create embedding for the question
                if self.using_external_embedding_model:
                    question_embedding = self.embedding_model.get_embeddings([question])
                else:
                    question_embedding = self.embedding_model.encode([question])
                
                # Search for nearest neighbors with FAISS
                if self.using_faiss and self.faiss_index is not None:
                    distances, indices = self.faiss_index.search(question_embedding, min(top_k, len(self.chunks)))
                    
                    # Convert indices to chunks and add distances
                    for i, idx in enumerate(indices[0]):
                        if idx < len(self.chunks):
                            chunk = self.chunks[idx].copy()
                            # Lower distance = higher relevance
                            relevance_score = 1.0 / (1.0 + distances[0][i])
                            chunk['relevance_score'] = min(relevance_score, 0.95)  # Cap at 0.95
                            relevant_chunks.append(chunk)
                    
                    return relevant_chunks
                # Use manual nearest neighbor search if FAISS not available
                elif self.chunk_embeddings is not None:
                    # Compute distances to all chunks
                    distances = []
                    for i, chunk_embedding in enumerate(self.chunk_embeddings):
                        # Simple Euclidean distance
                        dist = np.sum((question_embedding[0] - chunk_embedding) ** 2)
                        distances.append((i, dist))
                    
                    # Sort by distance (ascending)
                    distances.sort(key=lambda x: x[1])
                    
                    # Take top_k
                    for i, dist in distances[:top_k]:
                        chunk = self.chunks[i].copy()
                        relevance_score = 1.0 / (1.0 + dist)
                        chunk['relevance_score'] = min(relevance_score, 0.95)
                        relevant_chunks.append(chunk)
                    
                    return relevant_chunks
            except Exception as e:
                logger.error(f"Error in semantic search: {e}")
                # Fall back to keyword search
        
        # Fallback: Simple keyword-based search
        query_keywords = re.findall(r'\b\w+\b', question.lower())
        chunk_scores = []
        
        for i, chunk in enumerate(self.chunks):
            chunk_text = chunk.get('text', '').lower()
            score = 0
            
            for keyword in query_keywords:
                if keyword in chunk_text:
                    score += 1
            
            if score > 0:
                chunk_scores.append((i, score))
        
        # Sort by score descending
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Extract the top-k chunks
        for i, (chunk_idx, score) in enumerate(chunk_scores[:top_k]):
            chunk = self.chunks[chunk_idx].copy()
            # Normalize score to 0-1 range
            chunk['relevance_score'] = min(score / len(query_keywords), 0.95)
            relevant_chunks.append(chunk)
        
        return relevant_chunks
    
    def _rerank_chunks(self, question, chunks, top_k=5):
        """
        Rerank chunks using cross-encoder for more precise retrieval
        
        Args:
            question (str): The question
            chunks (list): List of candidate chunks
            top_k (int): Number of chunks to return
            
        Returns:
            list: Reranked chunks
        """
        if not chunks:
            return []
            
        try:
            # Prepare chunk pairs for reranking
            chunk_texts = [chunk.get('text', '')[:512] for chunk in chunks]  # Truncate to avoid too long sequences
            
            # Get reranked scores
            if self.using_external_embedding_model:
                # Use external cross encoder
                chunk_pairs = [(question, text) for text in chunk_texts]
                reranked_pairs = self.cross_encoder.rerank(question, chunk_texts, top_k)
                
                # Convert back to format with chunk objects
                reranked_chunks = []
                used_texts = set()
                
                for text, score in reranked_pairs:
                    # Find original chunk
                    for chunk in chunks:
                        chunk_text = chunk.get('text', '')[:512]
                        if chunk_text == text and text not in used_texts:
                            used_texts.add(text)
                            chunk_copy = chunk.copy()
                            chunk_copy['relevance_score'] = min(float(score), 0.95)
                            reranked_chunks.append(chunk_copy)
                            break
                
                return reranked_chunks[:top_k]
            elif self.using_sentence_transformers:
                # Use sentence-transformers cross encoder
                chunk_pairs = [[question, text] for text in chunk_texts]
                
                # Score chunks
                scores = self.cross_encoder.predict(chunk_pairs)
                
                # Create list of (chunk, score) pairs
                chunk_score_pairs = list(zip(chunks, scores))
                
                # Sort by score descending
                chunk_score_pairs.sort(key=lambda x: x[1], reverse=True)
                
                # Update relevance scores and return reranked chunks
                reranked_chunks = []
                for chunk, score in chunk_score_pairs[:top_k]:
                    chunk_copy = chunk.copy()
                    chunk_copy['relevance_score'] = min(float(score), 0.95)  # Cap at 0.95
                    reranked_chunks.append(chunk_copy)
                    
                return reranked_chunks
            else:
                return chunks[:top_k]  # No reranking possible
            
        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            return chunks[:top_k]  # Fall back to original ranking
    
    def _generate_rag_answer(self, question, relevant_chunks):
        """
        Generate an answer using Retrieval Augmented Generation
        
        Args:
            question (str): The question
            relevant_chunks (list): The most relevant chunks
            
        Returns:
            str: Generated answer
            list: Sources used
        """
        # Extract source information for the answer
        sources = []
        context_texts = []
        
        for chunk in relevant_chunks:
            chunk_text = chunk.get('text', '')
            document_info = chunk.get('document', {})
            filename = document_info.get('filename', 'Unknown')
            relevance_score = chunk.get('relevance_score', 0.0)
            
            # Extract answer from chunk using QA model
            try:
                qa_result = self.qa_pipeline(
                    question=question,
                    context=chunk_text,
                    max_answer_len=150
                )
                
                answer_text = qa_result.get('answer', '')
                confidence = qa_result.get('score', 0.0)
                
                # Extract context around the answer
                start_pos = max(0, chunk_text.lower().find(answer_text.lower()) - 100)
                end_pos = min(len(chunk_text), chunk_text.lower().find(answer_text.lower()) + len(answer_text) + 100)
                context = chunk_text[start_pos:end_pos]
                
                # If no context found, use a portion of the chunk
                if not context:
                    context = chunk_text[:300] + "..."
                
                # Add source information
                sources.append({
                    "source": f"data/documents/{filename}",
                    "filename": filename,
                    "section": "Relevant Section",
                    "relevanceScore": relevance_score,
                    "matchingSentences": [context]
                })
                
                # Add to context texts for answer generation
                context_texts.append({
                    "text": chunk_text,
                    "answer": answer_text,
                    "confidence": confidence
                })
                
            except Exception as e:
                logger.error(f"Error in QA processing: {e}")
                # Add source anyway if extraction fails
                sources.append({
                    "source": f"data/documents/{filename}",
                    "filename": filename,
                    "section": "Relevant Section",
                    "relevanceScore": relevance_score,
                    "matchingSentences": [chunk_text[:150] + "..."]
                })
        
        # Sort context texts by confidence
        context_texts.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        # Generate coherent answer from high-confidence answers
        if not context_texts:
            # Fallback to extractive answer
            return self._generate_extractive_answer(question, relevant_chunks)
        
        # Create final answer from extracted information
        final_answer = f"Basierend auf den Dokumenten ist die Antwort auf Ihre Frage '{question}':\n\n"
        
        # Take highest confidence answer as main response
        main_answer = context_texts[0].get('answer', '')
        if main_answer:
            final_answer += main_answer + "\n\n"
        else:
            # If no good answers found, use chunk text
            final_answer += context_texts[0].get('text', '')[:300] + "...\n\n"
        
        # Add additional information from other answers
        if len(context_texts) > 1:
            final_answer += "Zusätzliche Informationen aus den Dokumenten:\n"
            for i, ctx in enumerate(context_texts[1:3]):  # Max 2 additional answers
                answer = ctx.get('answer', '')
                if answer and answer != main_answer:
                    final_answer += f"- {answer}\n"
        
        return final_answer, sources
    
    def _generate_extractive_answer(self, question, relevant_chunks):
        """
        Generate an extractive answer without using the QA model
        
        Args:
            question (str): The question
            relevant_chunks (list): The most relevant chunks
            
        Returns:
            str: Extracted answer from chunks
            list: Sources used
        """
        sources = []
        relevant_texts = []
        
        for chunk in relevant_chunks:
            chunk_text = chunk.get('text', '')
            document_info = chunk.get('document', {})
            filename = document_info.get('filename', 'Unknown')
            relevance_score = chunk.get('relevance_score', 0.0)
            
            # Extract relevant sentences
            sentences = re.split(r'(?<=[.!?])\s+', chunk_text)
            matching_sentences = []
            
            query_keywords = set(re.findall(r'\b\w+\b', question.lower()))
            for sentence in sentences:
                sentence_keywords = set(re.findall(r'\b\w+\b', sentence.lower()))
                # If overlap is large enough
                if len(query_keywords.intersection(sentence_keywords)) >= 1 and len(sentence) > 20:
                    matching_sentences.append(sentence)
            
            # Use whole chunk if no sentences found
            if not matching_sentences and chunk_text:
                matching_sentences = [chunk_text[:200] + "..."]
            
            # Add source information
            if matching_sentences:
                sources.append({
                    "source": f"data/documents/{filename}",
                    "filename": filename,
                    "section": "Relevant Section",
                    "relevanceScore": relevance_score,
                    "matchingSentences": matching_sentences[:2]  # Limit to 2 sentences
                })
                
                relevant_texts.extend(matching_sentences)
        
        # Generate answer based on relevant texts
        if relevant_texts:
            # For general questions
            answer = f"Basierend auf den verfügbaren Dokumenten zu '{question}':\n\n"
            for i, text in enumerate(relevant_texts[:5]):  # Top 5 relevant texts
                answer += f"{i+1}. {text}\n\n"
        else:
            # Generic answer if no relevant texts found
            answer = f"Entschuldigung, ich konnte keine spezifischen Informationen zu '{question}' in den Dokumenten finden."
        
        return answer, sources
