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
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import faiss
from sentence_transformers import SentenceTransformer
import time

# Import local modules
from data_processing import DocumentProcessor
from model_training import QAModelTrainer, ChurnModel


class DocumentQA:
    """Document-based Question Answering System"""
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        qa_model_name: str = "deepset/gbert-base",
        processed_data_dir: Optional[str] = None
    ):
        """
        Initialize the QA system.
        
        Args:
            embedding_model_name: Model for document embeddings
            qa_model_name: Model for question answering
            processed_data_dir: Optional directory with pre-processed data
        """
        # Initialize document processor
        self.doc_processor = DocumentProcessor(model_name=embedding_model_name)
        
        # Initialize QA components
        self.qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.qa_model.to(self.device)
        
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
    
    def answer_question(self, question: str, top_k: int = 3) -> Dict:
        """
        Answer a question using the document-based QA system.
        
        Args:
            question: The question to answer
            top_k: Number of documents to retrieve
            
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
        
        # Extract answers from each document
        answers = []
        for doc in retrieved_docs:
            answer_data = self.extract_answer(question, doc["text"])
            answer_data["source"] = doc["source"]
            answer_data["filename"] = doc["filename"]
            answer_data["relevance_score"] = doc["relevance_score"]
            answers.append(answer_data)
        
        # Sort answers by score
        answers.sort(key=lambda x: x["score"], reverse=True)
        
        # Select the best answer
        best_answer = answers[0]
        
        # Format sources
        sources = []
        for i, answer in enumerate(answers):
            source = {
                "source": answer["source"],
                "relevance_score": answer["relevance_score"],
                "filename": answer["filename"]
            }
            sources.append(source)
        
        # Prepare response
        response = {
            "answer": best_answer["answer"],
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
            score = source["relevance_score"]
            formatted_answer += f"[{i+1}] {filename} (Relevance: {score:.2f})\n"
        
        return formatted_answer


class ChurnPredictionSystem:
    """Churn Prediction System using document-based insights"""
    
    def __init__(self, qa_system: DocumentQA, churn_model_path: Optional[str] = None):
        """
        Initialize the churn prediction system.
        
        Args:
            qa_system: DocumentQA system for providing insights
            churn_model_path: Optional path to a saved churn model
        """
        self.qa = qa_system
        self.churn_model = ChurnModel()
        
        if churn_model_path and os.path.exists(churn_model_path):
            self.churn_model.load_model(churn_model_path)
    
    def predict_churn(self, customer_data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict churn for customer data.
        
        Args:
            customer_data: DataFrame with customer data
            
        Returns:
            DataFrame with predictions
        """
        if self.churn_model.pipeline is None:
            raise ValueError("Churn model not trained or loaded")
        
        return self.churn_model.predict(customer_data)
    
    def train_from_documents(self, docs_dir: str, customer_data_path: str, save_path: str):
        """
        Train churn model using insights from documents.
        
        Args:
            docs_dir: Directory with documents
            customer_data_path: Path to customer data CSV
            save_path: Path to save the trained model
        """
        # First, make sure documents are processed
        if not self.qa.doc_processor.document_store:
            self.qa.process_documents(docs_dir)
        
        # Load customer data
        customer_data = pd.read_csv(customer_data_path)
        
        # Train the churn model
        self.churn_model.train(customer_data)
        
        # Save the model
        self.churn_model.save_model(save_path)
    
    def explain_prediction(self, customer_id: str, customer_data: pd.DataFrame) -> Dict:
        """
        Explain a churn prediction using model insights and document references.
        
        Args:
            customer_id: ID of the customer to explain
            customer_data: DataFrame with customer data
            
        Returns:
            Dictionary with explanation and document references
        """
        # Get customer data
        customer = customer_data[customer_data['id'] == customer_id]
        
        if customer.empty:
            return {"error": f"Customer ID {customer_id} not found"}
        
        # Make prediction
        prediction = self.churn_model.predict(customer)
        
        # Get SHAP values
        explainer = self.churn_model.explain_model(X_test=customer)
        
        # Extract top factors
        shap_values = explainer.shap_values(self.churn_model.pipeline.named_steps['preprocessing'].transform(customer))
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        feature_importance = list(zip(self.churn_model.feature_names, shap_values[0]))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        top_factors = [
            {"feature": feature, "importance": float(importance)}
            for feature, importance in feature_importance[:5]
        ]
        
        # Ask the QA system for insights on these factors
        insights = {}
        for factor in top_factors:
            feature = factor["feature"]
            query = f"What influences customer churn regarding {feature}?"
            insight = self.qa.answer_question(query)
            insights[feature] = insight
        
        # Prepare explanation
        explanation = {
            "customer_id": customer_id,
            "churn_probability": float(prediction['churn_probability'].iloc[0]),
            "risk_category": prediction['risk_category'].iloc[0],
            "top_factors": top_factors,
            "insights": insights
        }
        
        return explanation


def main():
    """Main function for command-line interface"""
    parser = argparse.ArgumentParser(description="Document-based QA and Churn Prediction System")
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
    qa_parser.add_argument("--top_k", type=int, default=3, help="Number of documents to retrieve")
    
    # Train churn model
    train_parser = subparsers.add_parser("train", help="Train churn model")
    train_parser.add_argument("--data", required=True, help="Path to customer data CSV")
    train_parser.add_argument("--docs_dir", help="Directory with documents")
    train_parser.add_argument("--processed_dir", help="Directory with processed data")
    train_parser.add_argument("--save_path", default="models/churn_model.pkl", help="Path to save model")
    
    # Predict churn
    predict_parser = subparsers.add_parser("predict", help="Predict churn")
    predict_parser.add_argument("--data", required=True, help="Path to customer data CSV")
    predict_parser.add_argument("--model", required=True, help="Path to trained model")
    predict_parser.add_argument("--output", help="Path to save predictions CSV")
    
    # Explain prediction
    explain_parser = subparsers.add_parser("explain", help="Explain churn prediction")
    explain_parser.add_argument("--customer_id", required=True, help="Customer ID to explain")
    explain_parser.add_argument("--data", required=True, help="Path to customer data CSV")
    explain_parser.add_argument("--model", required=True, help="Path to trained model")
    explain_parser.add_argument("--docs_dir", help="Directory with documents")
    explain_parser.add_argument("--processed_dir", help="Directory with processed data")
    
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
        response = qa.answer_question(args.query, top_k=args.top_k)
        formatted_answer = qa.format_answer_with_sources(response)
        
        print(formatted_answer)
    
    elif args.command == "train":
        # Initialize QA system
        qa = DocumentQA(processed_data_dir=args.processed_dir)
        
        # Process documents if needed
        if args.docs_dir and not qa.doc_processor.document_store:
            qa.process_documents(args.docs_dir)
        
        # Initialize churn system
        churn_system = ChurnPredictionSystem(qa)
        
        # Train churn model
        churn_system.train_from_documents(args.docs_dir, args.data, args.save_path)
    
    elif args.command == "predict":
        # Load data
        customer_data = pd.read_csv(args.data)
        
        # Initialize churn model
        churn_model = ChurnModel()
        churn_model.load_model(args.model)
        
        # Make predictions
        predictions = churn_model.predict(customer_data)
        
        # Add to original data
        result = pd.concat([customer_data, predictions], axis=1)
        
        # Save or print
        if args.output:
            result.to_csv(args.output, index=False)
            print(f"Predictions saved to {args.output}")
        else:
            print(result)
    
    elif args.command == "explain":
        # Load data
        customer_data = pd.read_csv(args.data)
        
        # Initialize QA system
        qa = DocumentQA(processed_data_dir=args.processed_dir)
        
        # Process documents if needed
        if args.docs_dir and not qa.doc_processor.document_store:
            qa.process_documents(args.docs_dir)
        
        # Initialize churn system
        churn_system = ChurnPredictionSystem(qa, args.model)
        
        # Explain prediction
        explanation = churn_system.explain_prediction(args.customer_id, customer_data)
        
        # Format and print explanation
        print(json.dumps(explanation, indent=2))


if __name__ == "__main__":
    main()
