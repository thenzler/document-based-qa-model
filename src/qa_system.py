"""
QA System Module for Document-based QA

This module combines the document processing and model training
components to create a complete question answering system that:
1. Retrieves relevant documents for a query
2. Extracts or generates answers from those documents
3. Keeps track of document sources for citations
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Optional, Any
from pathlib import Path
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM
import faiss
from sentence_transformers import SentenceTransformer, util
import time
import re

# Import local modules
from data_processing import DocumentProcessor
from model_training import QAModelTrainer, ChurnModel


class DocumentQA:
    """Document-based Question Answering System"""
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        qa_model_name: str = "deepset/gbert-base",
        generation_model_name: str = "google/flan-t5-base",
        processed_data_dir: Optional[str] = None
    ):
        """
        Initialize the QA system.
        
        Args:
            embedding_model_name: Model for document embeddings
            qa_model_name: Model for extractive question answering
            generation_model_name: Model for answer generation
            processed_data_dir: Optional directory with pre-processed data
        """
        # Initialize document processor
        self.doc_processor = DocumentProcessor(model_name=embedding_model_name)
        
        # Initialize QA components
        self.qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.qa_model.to(self.device)
        
        # Initialize generation model for answer synthesis
        try:
            self.gen_tokenizer = AutoTokenizer.from_pretrained(generation_model_name)
            self.gen_model = AutoModelForSeq2SeqLM.from_pretrained(generation_model_name)
            self.gen_model.to(self.device)
            self.has_generation = True
        except:
            print(f"Could not load generation model {generation_model_name}. Using extractive QA only.")
            self.has_generation = False
        
        # Load pre-processed data if available
        if processed_data_dir:
            self.doc_processor.load_processed_data(processed_data_dir)
    
    def process_documents(self, docs_dir: str, save_dir: Optional[str] = None):
        """
        Process documents in the specified directory.
        
        Args:
            docs_dir: Directory containing documents
            save_dir: Optional directory to save processed data
        """
        self.doc_processor.process_pipeline(docs_dir)
        
        if save_dir:
            self.doc_processor.save_processed_data(save_dir)
    
    def retrieve_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve the most relevant documents for a query.
        
        Args:
            query: The query text
            top_k: Number of documents to retrieve
            
        Returns:
            List of relevant document dictionaries with scores
        """
        if self.doc_processor.index is None:
            raise ValueError("No document index available. Process documents first.")
        
        # Encode the query
        query_embedding = self.doc_processor.embedding_model.encode([query])[0]
        query_embedding = np.array([query_embedding]).astype('float32')
        
        # Search the index
        distances, indices = self.doc_processor.index.search(query_embedding, top_k)
        
        # Get the corresponding documents
        retrieved_docs = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.doc_processor.document_store):
                doc = self.doc_processor.document_store[idx].copy()
                doc['relevance_score'] = float(1 / (1 + distance))  # Convert distance to score
                doc['rank'] = i + 1
                retrieved_docs.append(doc)
        
        return retrieved_docs
    
    def rerank_documents(self, query: str, retrieved_docs: List[Dict], top_k: int = 3) -> List[Dict]:
        """
        Rerank retrieved documents using keyword matching and semantic similarity.
        
        Args:
            query: The query text
            retrieved_docs: List of retrieved documents
            top_k: Number of documents to keep
            
        Returns:
            Reranked list of documents
        """
        if not retrieved_docs:
            return []
        
        # Extract query keywords using the document processor
        query_keywords = self.doc_processor.extract_keywords(query)
        
        # Calculate keyword overlap scores
        for i, doc in enumerate(retrieved_docs):
            doc_keywords = set(doc.get('keywords', []))
            keyword_overlap = len(query_keywords.intersection(doc_keywords)) / max(1, len(query_keywords))
            
            # Update relevance score with keyword information
            retrieved_docs[i]['keyword_score'] = keyword_overlap
            retrieved_docs[i]['combined_score'] = (
                retrieved_docs[i]['relevance_score'] * 0.7 + 
                keyword_overlap * 0.3
            )
        
        # Sort by combined score
        reranked_docs = sorted(retrieved_docs, key=lambda x: x['combined_score'], reverse=True)
        
        # Return top_k documents
        return reranked_docs[:top_k]
    
    def extract_answer(self, question: str, context: str) -> Dict:
        """
        Extract an answer from a context using the QA model.
        
        Args:
            question: The question text
            context: The context text
            
        Returns:
            Dictionary with answer and metadata
        """
        # Tokenize input
        inputs = self.qa_tokenizer(
            question,
            context,
            return_tensors="pt",
            max_length=512,
            truncation="only_second",
            stride=128,
            return_overflowing_tokens=True,
            padding="max_length"
        )
        
        # Move inputs to device
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.qa_model(input_ids=input_ids, attention_mask=attention_mask)
            start_scores = outputs.start_logits
            end_scores = outputs.end_logits
        
        # Find the indices with the highest score
        start_idx = torch.argmax(start_scores, dim=1).item()
        end_idx = torch.argmax(end_scores, dim=1).item()
        
        # Make sure end comes after start
        if end_idx < start_idx:
            end_idx = start_idx + 10  # Limit to reasonable length
            end_idx = min(end_idx, len(input_ids[0]) - 1)
        
        # Convert token indices to string
        answer_tokens = input_ids[0][start_idx:end_idx+1]
        answer_text = self.qa_tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
        
        # Check if answer makes sense (not empty or too short)
        if not answer_text or len(answer_text) < 2:
            answer_text = "No answer found in the context."
        
        return {
            "answer": answer_text,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "score": float(torch.max(start_scores).item() + torch.max(end_scores).item()) / 2
        }
    
    def generate_answer(self, question: str, contexts: List[str]) -> Dict:
        """
        Generate an answer using the T5 generation model.
        
        Args:
            question: The question text
            contexts: List of context texts
            
        Returns:
            Dictionary with generated answer and metadata
        """
        if not self.has_generation:
            # Fallback to extractive QA
            best_answer = {"answer": "Generation model not available", "score": 0}
            for context in contexts:
                answer = self.extract_answer(question, context)
                if answer["score"] > best_answer["score"]:
                    best_answer = answer
            return best_answer
        
        # Combine contexts (up to max token limit)
        combined_context = " ".join(contexts)
        
        # Create prompt for generation
        prompt = f"Answer the question based on the context.\n\nContext: {combined_context}\n\nQuestion: {question}\n\nAnswer:"
        
        # Tokenize input
        inputs = self.gen_tokenizer(
            prompt,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding="max_length"
        )
        
        # Move inputs to device
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Generate answer
        with torch.no_grad():
            outputs = self.gen_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=150,
                num_beams=4,
                early_stopping=True
            )
        
        # Decode and clean the answer
        answer_text = self.gen_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        # Sometimes T5 repeats the question or context, remove it if possible
        answer_text = answer_text.replace(question, "").strip()
        
        return {
            "answer": answer_text,
            "score": 1.0,  # No confidence score for generation
            "is_generated": True
        }
    
    def answer_question(self, question: str, top_k: int = 5, use_generation: bool = True) -> Dict:
        """
        Answer a question using the document-based QA system.
        
        Args:
            question: The question to answer
            top_k: Number of documents to retrieve
            use_generation: Whether to use the generation model
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        start_time = time.time()
        
        # Retrieve relevant documents
        retrieved_docs = self.retrieve_documents(question, top_k=top_k)
        
        if not retrieved_docs:
            return {
                "answer": "I couldn't find any relevant documents to answer this question.",
                "sources": [],
                "processing_time": time.time() - start_time
            }
        
        # Rerank documents
        reranked_docs = self.rerank_documents(question, retrieved_docs, top_k=top_k)
        
        if use_generation and self.has_generation:
            # Get context texts
            contexts = [doc["text"] for doc in reranked_docs]
            
            # Generate answer
            answer_data = self.generate_answer(question, contexts)
            
            # Find which documents contributed to the answer
            answer_text = answer_data["answer"]
            contributing_docs = []
            
            for doc in reranked_docs:
                # Check if significant parts of the doc appear in the answer
                doc_sentences = re.split(r'[.!?]', doc["text"])
                for sentence in doc_sentences:
                    if len(sentence) > 30 and sentence.strip() in answer_text:
                        doc["contribution"] = sentence.strip()
                        contributing_docs.append(doc)
                        break
            
            # If no specific contributions found, use all docs
            if not contributing_docs:
                contributing_docs = reranked_docs
            
            # Format sources
            sources = []
            for doc in contributing_docs:
                source = {
                    "source": doc["source"],
                    "section": doc.get("section", ""),
                    "filename": doc["filename"],
                    "relevance_score": doc["combined_score"],
                    "contribution": doc.get("contribution", "")
                }
                sources.append(source)
            
        else:
            # Extract answers from each document
            answers = []
            for doc in reranked_docs:
                answer_data = self.extract_answer(question, doc["text"])
                answer_data["source"] = doc["source"]
                answer_data["section"] = doc.get("section", "")
                answer_data["filename"] = doc["filename"]
                answer_data["relevance_score"] = doc["combined_score"]
                answers.append(answer_data)
            
            # Sort answers by score
            answers.sort(key=lambda x: x["score"], reverse=True)
            
            # Select the best answer
            best_answer = answers[0]
            
            # Format sources
            sources = []
            for answer in answers:
                source = {
                    "source": answer["source"],
                    "section": answer["section"],
                    "filename": answer["filename"],
                    "relevance_score": answer["relevance_score"]
                }
                sources.append(source)
            
            answer_text = best_answer["answer"]
        
        # Prepare response
        response = {
            "answer": answer_text,
            "sources": sources,
            "processing_time": time.time() - start_time
        }
        
        return response
    
    def format_answer_with_sources(self, response: Dict) -> str:
        """
        Format the answer with source citations.
        
        Args:
            response: The response from answer_question
            
        Returns:
            Formatted answer text with citations
        """
        answer = response["answer"]
        sources = response["sources"]
        
        formatted_answer = f"Answer: {answer}\n\n"
        formatted_answer += "Sources:\n"
        
        for i, source in enumerate(sources):
            filename = Path(source["source"]).name
            section = source.get("section", "")
            score = source["relevance_score"]
            
            formatted_answer += f"[{i+1}] {filename}"
            if section:
                formatted_answer += f" - Section: {section}"
            formatted_answer += f" (Relevance: {score:.2f})\n"
        
        return formatted_answer


class DocumentQueryEngine:
    """Enhanced document query engine with support for multiple queries and explanations"""
    
    def __init__(self, qa_system: DocumentQA):
        """
        Initialize the document query engine.
        
        Args:
            qa_system: DocumentQA system for retrieving and answering
        """
        self.qa = qa_system
    
    def multi_query(self, main_question: str, num_variations: int = 3) -> Dict:
        """
        Generate multiple query variations to improve retrieval.
        
        Args:
            main_question: The main question to answer
            num_variations: Number of query variations to generate
            
        Returns:
            Combined answer with sources
        """
        # Create query variations
        variations = [main_question]
        
        # Add simple variations
        if "was ist" in main_question.lower():
            variations.append(main_question.lower().replace("was ist", "definiere"))
            variations.append(main_question.lower().replace("was ist", "erkläre"))
        elif "wie funktioniert" in main_question.lower():
            variations.append(main_question.lower().replace("wie funktioniert", "beschreibe"))
            variations.append(main_question.lower().replace("wie funktioniert", "erkläre die funktionsweise von"))
        
        # Get answers for each variation
        all_answers = []
        all_sources = []
        
        for i, query in enumerate(variations[:num_variations]):
            print(f"Processing query variation {i+1}: {query}")
            response = self.qa.answer_question(query)
            
            if response["answer"] != "I couldn't find any relevant documents to answer this question.":
                all_answers.append(response["answer"])
                all_sources.extend(response["sources"])
        
        # Combine and deduplicate sources
        unique_sources = []
        source_keys = set()
        
        for source in all_sources:
            key = f"{source['source']}:{source.get('section', '')}"
            if key not in source_keys:
                source_keys.add(key)
                unique_sources.append(source)
        
        # Sort sources by relevance
        unique_sources.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Take the best answer or combine them
        if all_answers:
            # Use the first answer (from the main query) as it's usually the best
            final_answer = all_answers[0]
        else:
            final_answer = "I couldn't find relevant information to answer this question."
        
        # Create response
        response = {
            "answer": final_answer,
            "sources": unique_sources[:5],  # Limit to top 5 sources
            "processing_time": 0  # Not tracking time for the combined query
        }
        
        return response
    
    def explain_with_documents(self, question: str, answer: str, sources: List[Dict]) -> str:
        """
        Generate an explanation for the answer based on source documents.
        
        Args:
            question: The original question
            answer: The answer provided
            sources: The source documents used
            
        Returns:
            Explanation text
        """
        source_texts = []
        for source in sources:
            # Get the actual document text
            doc_idx = None
            for i, doc in enumerate(self.qa.doc_processor.document_store):
                if doc["source"] == source["source"] and doc.get("section", "") == source.get("section", ""):
                    doc_idx = i
                    break
            
            if doc_idx is not None:
                doc = self.qa.doc_processor.document_store[doc_idx]
                source_texts.append(doc["text"])
        
        if not source_texts:
            return "No source documents available to explain this answer."
        
        # Create an explanation prompt
        explanation_prompt = f"""
        Question: {question}
        Answer: {answer}
        
        Documents that supported this answer:
        {" ".join(source_texts[:3])}  # Limit to first 3 documents to avoid token limits
        
        Explain in detail how these documents support the answer:
        """
        
        # Generate explanation using T5 if available
        if self.qa.has_generation:
            inputs = self.qa.gen_tokenizer(
                explanation_prompt,
                return_tensors="pt",
                max_length=1024,
                truncation=True,
                padding="max_length"
            ).to(self.qa.device)
            
            with torch.no_grad():
                output_ids = self.qa.gen_model.generate(
                    inputs["input_ids"],
                    max_length=300,
                    num_beams=4,
                    early_stopping=True
                )
            
            explanation = self.qa.gen_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        else:
            # Manual explanation if generation model not available
            explanation = "The answer is supported by the following documents:\n\n"
            for i, text in enumerate(source_texts[:3]):
                explanation += f"Document {i+1}:\n"
                # Extract a relevant snippet (first 200 chars)
                explanation += text[:200] + "...\n\n"
        
        return explanation


def main():
    """Main function for command-line interface"""
    parser = argparse.ArgumentParser(description="Document-based QA System")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Process documents
    process_parser = subparsers.add_parser("process", help="Process documents")
    process_parser.add_argument("--docs_dir", required=True, help="Directory with documents")
    process_parser.add_argument("--save_dir", help="Directory to save processed data")
    
    # Answer question
    qa_parser = subparsers.add_parser("qa", help="Answer a question")
    qa_parser.add_argument("--query", required=True, help="Question to answer")
    qa_parser.add_argument("--docs_dir", help="Directory with documents")
    qa_parser.add_argument("--processed_dir", help="Directory with processed data")
    qa_parser.add_argument("--top_k", type=int, default=5, help="Number of documents to retrieve")
    qa_parser.add_argument("--use_generation", action="store_true", help="Use generation model")
    qa_parser.add_argument("--explain", action="store_true", help="Generate explanation")
    
    # Multi-query enhancement
    multiqa_parser = subparsers.add_parser("multiqa", help="Answer with multiple query variations")
    multiqa_parser.add_argument("--query", required=True, help="Main question to answer")
    multiqa_parser.add_argument("--docs_dir", help="Directory with documents")
    multiqa_parser.add_argument("--processed_dir", help="Directory with processed data")
    multiqa_parser.add_argument("--variations", type=int, default=3, help="Number of query variations")
    
    args = parser.parse_args()
    
    if args.command == "process":
        # Process documents
        doc_processor = DocumentProcessor()
        doc_processor.process_pipeline(args.docs_dir)
        
        if args.save_dir:
            doc_processor.save_processed_data(args.save_dir)
    
    elif args.command == "qa":
        # Initialize QA system
        qa = DocumentQA(processed_data_dir=args.processed_dir)
        
        # Process documents if needed
        if args.docs_dir and not qa.doc_processor.document_store:
            qa.process_documents(args.docs_dir)
        
        # Answer question
        response = qa.answer_question(args.query, top_k=args.top_k, use_generation=args.use_generation)
        formatted_answer = qa.format_answer_with_sources(response)
        
        print(formatted_answer)
        
        # Generate explanation if requested
        if args.explain:
            query_engine = DocumentQueryEngine(qa)
            explanation = query_engine.explain_with_documents(
                args.query, 
                response["answer"], 
                response["sources"]
            )
            print("\nExplanation:")
            print(explanation)
    
    elif args.command == "multiqa":
        # Initialize QA system
        qa = DocumentQA(processed_data_dir=args.processed_dir)
        
        # Process documents if needed
        if args.docs_dir and not qa.doc_processor.document_store:
            qa.process_documents(args.docs_dir)
        
        # Create query engine
        query_engine = DocumentQueryEngine(qa)
        
        # Answer with multiple query variations
        response = query_engine.multi_query(args.query, num_variations=args.variations)
        
        # Format and print answer
        formatted_answer = qa.format_answer_with_sources(response)
        print(formatted_answer)


if __name__ == "__main__":
    main()
