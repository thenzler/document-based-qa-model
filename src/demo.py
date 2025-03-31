#!/usr/bin/env python3
"""
Demo Script for Document-based Churn QA System

This script demonstrates how to use the document-based QA system
to answer questions about churn prediction, with proper citation
of document sources used for the answers.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import time
import random

# Import local modules
from qa_system import DocumentQA, ChurnQASystem, DocumentQueryEngine
from data_processing import DocumentProcessor
from model_training import ChurnModel, DocumentEnhancedChurnModel

# Constants
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
CHURN_DOCS_DIR = DATA_DIR / "churn_docs"
MODELS_DIR = Path("models")

# Ensure directories exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def process_documents(docs_dir: Path, save_dir: Path = None, force_reprocess: bool = False):
    """
    Process documents for the QA system.
    
    Args:
        docs_dir: Directory with documents
        save_dir: Directory to save processed data
        force_reprocess: Whether to force reprocessing even if processed data exists
        
    Returns:
        Document processor with processed documents
    """
    if save_dir is None:
        save_dir = PROCESSED_DIR
    
    # Check if processed data already exists
    if save_dir.exists() and not force_reprocess and list(save_dir.glob("*")):
        print(f"Loading pre-processed documents from {save_dir}")
        doc_processor = DocumentProcessor()
        doc_processor.load_processed_data(save_dir)
        return doc_processor
    
    print(f"Processing documents from {docs_dir}")
    doc_processor = DocumentProcessor()
    doc_processor.process_pipeline(docs_dir)
    
    if save_dir:
        print(f"Saving processed documents to {save_dir}")
        doc_processor.save_processed_data(save_dir)
    
    return doc_processor


def initialize_qa_system(processed_dir: Path = None, use_generation: bool = True):
    """
    Initialize the document-based QA system.
    
    Args:
        processed_dir: Directory with processed document data
        use_generation: Whether to use the generation model
        
    Returns:
        Initialized QA system
    """
    if processed_dir is None:
        processed_dir = PROCESSED_DIR
    
    print("Initializing QA system...")
    
    # Initialize with German models if available, or fallback to English
    qa_model_name = "deepset/gbert-base" if use_generation else "distilbert-base-uncased-distilled-squad"
    generation_model_name = "google/flan-t5-base" if use_generation else None
    
    # Create specialized churn QA system
    qa_system = ChurnQASystem(
        embedding_model_name="sentence-transformers/all-mpnet-base-v2",
        qa_model_name=qa_model_name,
        generation_model_name=generation_model_name,
        processed_data_dir=str(processed_dir) if processed_dir.exists() else None
    )
    
    return qa_system


def create_sample_churn_data():
    """
    Create sample customer data for churn prediction.
    
    Returns:
        DataFrame with sample customer data
    """
    # Create sample data with relevant features for churn prediction
    num_customers = 10
    data = {
        'customer_id': [f'CUST-{i:04d}' for i in range(1, num_customers + 1)],
        'alter': np.random.randint(18, 70, num_customers),  # Customer age
        'vertragsdauer': np.random.randint(1, 60, num_customers),  # Contract duration in months
        'nutzungsfrequenz': np.random.randint(1, 30, num_customers),  # Usage frequency
        'support_anfragen': np.random.randint(0, 10, num_customers),  # Support requests
        'zahlungsverzoegerungen': np.random.randint(0, 5, num_customers),  # Payment delays
        'upgrades': np.random.randint(0, 3, num_customers),  # Number of upgrades
        'preiserhohungen': np.random.randint(0, 2, num_customers),  # Price increases
        'fehlermeldungen': np.random.randint(0, 15, num_customers),  # Error messages
        'nps_score': np.random.randint(1, 11, num_customers),  # NPS score
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some correlations to make prediction more realistic
    # Customers with longer contracts tend to have higher NPS
    df['nps_score'] = np.round(df['nps_score'] * (0.8 + 0.004 * df['vertragsdauer'])).clip(1, 10).astype(int)
    
    # Payment delays negatively affect NPS
    df.loc[df['zahlungsverzoegerungen'] > 2, 'nps_score'] = df.loc[df['zahlungsverzoegerungen'] > 2, 'nps_score'] - 2
    
    # Add churned column for training (not for prediction)
    # Higher probability of churn with more payment delays and fewer usages
    churn_prob = 0.2 + 0.15 * df['zahlungsverzoegerungen'] - 0.01 * df['nutzungsfrequenz'] + 0.05 * df['fehlermeldungen'] - 0.01 * df['nps_score']
    df['Churn'] = np.random.binomial(1, churn_prob.clip(0, 0.9))
    
    return df


def train_churn_model(data: pd.DataFrame, save_path: Path = None):
    """
    Train a churn prediction model.
    
    Args:
        data: DataFrame with customer data including a 'Churn' column
        save_path: Path to save the trained model
        
    Returns:
        Trained churn model
    """
    if save_path is None:
        save_path = MODELS_DIR / "churn_model.pkl"
    
    # Check if model already exists
    if save_path.exists():
        print(f"Loading existing churn model from {save_path}")
        model = DocumentEnhancedChurnModel("RandomForest")
        model.load_model(save_path)
        return model
    
    print("Training churn prediction model...")
    model = DocumentEnhancedChurnModel("RandomForest")
    results = model.train(data)
    
    print("Model training completed.")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  ROC AUC: {results['roc_auc']:.4f}")
    
    if save_path:
        model.save_model(save_path)
    
    return model


def demo_qa_workflow(qa_system, questions=None):
    """
    Demonstrate the question answering workflow with document sources.
    
    Args:
        qa_system: Initialized QA system
        questions: Optional list of questions to answer
    """
    if questions is None:
        questions = [
            "Was ist Churn Prediction?",
            "Welche Algorithmen werden für Churn Prediction verwendet?",
            "Welche Features sind wichtig für ein Churn-Prediction-Modell?",
            "Wie werden Churn-Prediction-Modelle evaluiert?",
            "Wie kann man mit Kunden mit hohem Abwanderungsrisiko umgehen?"
        ]
    
    print("\n" + "=" * 80)
    print("DOCUMENT-BASED QA DEMO")
    print("=" * 80)
    
    for i, question in enumerate(questions):
        print(f"\nQuestion {i+1}: {question}")
        print("-" * 80)
        
        # Measure response time
        start_time = time.time()
        
        # Get answer with document sources
        response = qa_system.answer_churn_question(question)
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Display formatted answer with sources
        formatted_answer = qa_system.format_answer_with_sources(response)
        print(formatted_answer)
        
        print(f"Response time: {response_time:.2f} seconds")
        print("-" * 80)


def demo_churn_prediction(model, qa_system):
    """
    Demonstrate churn prediction with document-based explanations.
    
    Args:
        model: Trained churn prediction model
        qa_system: Document QA system for explanations
    """
    print("\n" + "=" * 80)
    print("CHURN PREDICTION WITH DOCUMENT REFERENCES")
    print("=" * 80)
    
    # Create sample data for prediction
    test_data = create_sample_churn_data()
    
    # Make predictions
    predictions = model.predict_with_interventions(test_data.drop(columns=['Churn']))
    
    # Display results with document references
    print("\nChurn Prediction Results:")
    print("-" * 80)
    
    for i, (_, customer) in enumerate(test_data.iterrows(), 1):
        prediction = predictions.iloc[i-1]
        customer_id = customer['customer_id']
        risk_category = prediction['risk_category']
        churn_prob = prediction['churn_probability']
        
        print(f"\nCustomer: {customer_id}")
        print(f"Risk Category: {risk_category}")
        print(f"Churn Probability: {churn_prob:.2f}")
        
        # Get specific explanation for this customer
        explanation = model.explain_prediction_with_documents(customer)
        
        # Display risk factors
        if 'top_risk_factors' in prediction:
            print(f"Top Risk Factors: {prediction['top_risk_factors']}")
        
        # Display recommended interventions
        if 'recommended_interventions' in prediction:
            print(f"Recommended Interventions: {prediction['recommended_interventions']}")
        
        # Get document references
        doc_refs = explanation['document_references']
        if doc_refs:
            print("\nRelevant Documentation:")
            for ref in doc_refs:
                print(f"  - {ref['filename']} (Section: {ref['section']})")
        
        # For high risk customers, query the QA system for more detailed intervention advice
        if risk_category == 'Hohes Risiko':
            intervention_q = "Welche Maßnahmen sind effektiv für Kunden mit hohem Churn-Risiko?"
            intervention_response = qa_system.answer_question(intervention_q, top_k=2, use_generation=True)
            
            print("\nIntervention Strategy:")
            print(f"  {intervention_response['answer']}")
            print("  (Based on document analysis)")
        
        print("-" * 80)
        
        # Only show a few customers for brevity in the demo
        if i >= 3:
            print("\n[Showing only 3 customers for brevity]")
            break


def interactive_qa_session(qa_system):
    """
    Run an interactive Q&A session where the user can ask questions.
    
    Args:
        qa_system: Initialized QA system
    """
    print("\n" + "=" * 80)
    print("INTERACTIVE QA SESSION")
    print("=" * 80)
    print("Type your questions about churn prediction (or 'exit' to quit)")
    
    while True:
        print("\nQuestion: ", end="")
        question = input().strip()
        
        if question.lower() in ('exit', 'quit', 'q'):
            break
        
        if not question:
            continue
        
        # Process the question
        response = qa_system.answer_churn_question(question)
        
        # Display the answer with sources
        formatted_answer = qa_system.format_answer_with_sources(response)
        print("\n" + formatted_answer)


def main():
    """Main function to run the demo"""
    parser = argparse.ArgumentParser(description="Demo for Document-based Churn QA System")
    parser.add_argument("--docs_dir", type=str, default="data/churn_docs",
                        help="Directory with documents")
    parser.add_argument("--processed_dir", type=str, default="data/processed",
                        help="Directory for processed data")
    parser.add_argument("--interactive", action="store_true",
                        help="Run interactive QA session")
    parser.add_argument("--force_reprocess", action="store_true",
                        help="Force document reprocessing")
    parser.add_argument("--no_generation", action="store_true",
                        help="Disable generation model (faster but less accurate)")
    
    args = parser.parse_args()
    
    # Convert to Path objects
    docs_dir = Path(args.docs_dir)
    processed_dir = Path(args.processed_dir)
    
    # Process documents
    doc_processor = process_documents(
        docs_dir, 
        processed_dir, 
        force_reprocess=args.force_reprocess
    )
    
    # Initialize QA system
    qa_system = initialize_qa_system(
        processed_dir=processed_dir,
        use_generation=not args.no_generation
    )
    
    # If no processed data was loaded, process the documents now
    if not qa_system.doc_processor.document_store:
        print("No processed data found. Processing documents...")
        qa_system.process_documents(str(docs_dir), str(processed_dir))
    
    # Create and train churn model
    sample_data = create_sample_churn_data()
    churn_model = train_churn_model(sample_data)
    
    # Run demos
    demo_qa_workflow(qa_system)
    demo_churn_prediction(churn_model, qa_system)
    
    # Run interactive session if requested
    if args.interactive:
        interactive_qa_session(qa_system)
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
