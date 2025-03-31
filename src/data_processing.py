"""
Data Processing Module for Document-based QA System

This module handles all the document processing, including:
- Loading documents from various formats (txt, pdf, etc.)
- Preprocessing text (tokenization, cleaning, etc.)
- Creating document embeddings
- Indexing documents for fast retrieval
"""

import os
import re
import glob
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Tuple, Optional
from pathlib import Path

# For text processing
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import spacy

# For document indexing
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentProcessor:
    """Handles document loading, preprocessing, and indexing"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """
        Initialize the document processor.
        
        Args:
            model_name: Name of the sentence transformer model to use for embeddings
        """
        self.embedding_model = SentenceTransformer(model_name)
        try:
            self.nlp = spacy.load("de_core_news_md")
        except OSError:
            # Download if not available
            os.system("python -m spacy download de_core_news_md")
            self.nlp = spacy.load("de_core_news_md")
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.document_store = []
        self.document_embeddings = None
        self.index = None
    
    def load_documents(self, docs_dir: Union[str, Path]) -> List[Dict]:
        """
        Load all documents from a directory.
        
        Args:
            docs_dir: Directory containing document files
            
        Returns:
            List of document dictionaries with text and metadata
        """
        docs_dir = Path(docs_dir)
        
        # List all text files in the directory
        text_files = list(docs_dir.glob("**/*.txt"))
        
        documents = []
        
        for file_path in text_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                
                # Create document entry with metadata
                doc = {
                    "text": text,
                    "source": str(file_path),
                    "filename": file_path.name,
                    "created_at": os.path.getctime(file_path)
                }
                documents.append(doc)
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        print(f"Loaded {len(documents)} documents")
        return documents
    
    def preprocess_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Preprocess documents by splitting them into chunks.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of document chunk dictionaries
        """
        processed_docs = []
        
        for doc in documents:
            text = doc["text"]
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Create document chunk entries
            for i, chunk in enumerate(chunks):
                chunk_doc = {
                    "text": chunk,
                    "source": doc["source"],
                    "filename": doc["filename"],
                    "chunk_id": i,
                    "doc_id": documents.index(doc)
                }
                processed_docs.append(chunk_doc)
        
        print(f"Created {len(processed_docs)} document chunks")
        self.document_store = processed_docs
        return processed_docs
    
    def create_embeddings(self, documents: Optional[List[Dict]] = None) -> np.ndarray:
        """
        Create embeddings for all documents or provided documents.
        
        Args:
            documents: Optional list of documents to embed
            
        Returns:
            Numpy array of document embeddings
        """
        docs_to_embed = documents if documents is not None else self.document_store
        
        texts = [doc["text"] for doc in docs_to_embed]
        embeddings = self.embedding_model.encode(texts)
        
        self.document_embeddings = embeddings
        print(f"Created embeddings with shape {embeddings.shape}")
        return embeddings
    
    def build_index(self, embeddings: Optional[np.ndarray] = None) -> faiss.Index:
        """
        Build a FAISS index for fast similarity search.
        
        Args:
            embeddings: Optional pre-computed embeddings
            
        Returns:
            FAISS index
        """
        embs = embeddings if embeddings is not None else self.document_embeddings
        
        if embs is None:
            raise ValueError("No embeddings available. Call create_embeddings first.")
        
        # Get dimensions of the embeddings
        d = embs.shape[1]
        
        # Create a Flat L2 index (exact search)
        index = faiss.IndexFlatL2(d)
        
        # Add embeddings to the index
        index.add(embs.astype('float32'))
        
        self.index = index
        print(f"Built index with {index.ntotal} vectors")
        return index
    
    def process_pipeline(self, docs_dir: Union[str, Path]) -> Tuple[List[Dict], np.ndarray, faiss.Index]:
        """
        Run the full document processing pipeline.
        
        Args:
            docs_dir: Directory containing document files
            
        Returns:
            Tuple of (document_store, embeddings, index)
        """
        documents = self.load_documents(docs_dir)
        processed_docs = self.preprocess_documents(documents)
        embeddings = self.create_embeddings(processed_docs)
        index = self.build_index(embeddings)
        
        return processed_docs, embeddings, index
    
    def save_processed_data(self, save_dir: Union[str, Path]):
        """
        Save processed documents and embeddings.
        
        Args:
            save_dir: Directory to save the processed data
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save document store as JSON
        docs_df = pd.DataFrame(self.document_store)
        docs_df.to_json(save_dir / "document_store.json", orient="records")
        
        # Save embeddings as numpy array
        if self.document_embeddings is not None:
            np.save(save_dir / "document_embeddings.npy", self.document_embeddings)
        
        # Save FAISS index if available
        if self.index is not None:
            faiss.write_index(self.index, str(save_dir / "faiss_index.bin"))
        
        print(f"Saved processed data to {save_dir}")
    
    def load_processed_data(self, load_dir: Union[str, Path]):
        """
        Load previously processed documents and embeddings.
        
        Args:
            load_dir: Directory from which to load the processed data
        """
        load_dir = Path(load_dir)
        
        # Load document store
        docs_path = load_dir / "document_store.json"
        if docs_path.exists():
            docs_df = pd.read_json(docs_path)
            self.document_store = docs_df.to_dict(orient="records")
            print(f"Loaded {len(self.document_store)} document chunks")
        
        # Load embeddings
        embs_path = load_dir / "document_embeddings.npy"
        if embs_path.exists():
            self.document_embeddings = np.load(embs_path)
            print(f"Loaded embeddings with shape {self.document_embeddings.shape}")
        
        # Load FAISS index
        index_path = load_dir / "faiss_index.bin"
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
            print(f"Loaded index with {self.index.ntotal} vectors")


if __name__ == "__main__":
    # Example usage
    processor = DocumentProcessor()
    processor.process_pipeline("data/documents")
    processor.save_processed_data("data/processed")
