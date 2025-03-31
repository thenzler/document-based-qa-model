"""
Demo script for the Document-based QA System

This script provides an easy way to:
1. Process and index documents from a folder
2. Ask questions about the documents
3. Get answers with source attributions

Usage:
  python demo.py --docs_dir /path/to/documents
"""

import os
import argparse
import sys
from pathlib import Path
import time

from qa_system import DocumentQA, DocumentQueryEngine


def process_documents(qa, docs_dir, processed_dir):
    """Process documents if needed."""
    # Check if processed data exists
    if processed_dir and os.path.exists(processed_dir):
        try:
            qa.doc_processor.load_processed_data(processed_dir)
            print(f"Loaded processed documents from {processed_dir}")
            return True
        except Exception as e:
            print(f"Error loading processed data: {e}")
    
    # Process documents
    print(f"Processing documents from {docs_dir}...")
    qa.process_documents(docs_dir, save_dir=processed_dir)
    print("Document processing complete!")
    return True


def interactive_qa(qa, query_engine=None):
    """Interactive QA session."""
    print("\n=== Document QA System ===")
    print("Ask questions about the documents. Type 'exit' to quit.")
    
    while True:
        print("\nEnter your question:")
        question = input("> ").strip()
        
        if question.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break
        
        if not question:
            continue
        
        start_time = time.time()
        
        # Determine whether to use multi-query and generation
        use_multiquery = len(question.split()) > 3  # Only use for non-trivial questions
        use_generation = True  # Default to use generation if available
        
        try:
            if use_multiquery and query_engine:
                print("Using multi-query approach for better results...")
                response = query_engine.multi_query(question)
            else:
                print("Finding answer...")
                response = qa.answer_question(question, use_generation=use_generation)
            
            formatted_answer = qa.format_answer_with_sources(response)
            print("\n" + formatted_answer)
            
            # Ask if user wants an explanation
            print("\nDo you want a detailed explanation of how this answer was derived? (y/n)")
            want_explanation = input("> ").strip().lower()
            
            if want_explanation in ["y", "yes"]:
                if query_engine:
                    print("\nGenerating explanation...")
                    explanation = query_engine.explain_with_documents(
                        question, 
                        response["answer"], 
                        response["sources"]
                    )
                    print("\nExplanation:")
                    print(explanation)
                else:
                    print("Explanation feature requires the query engine to be initialized.")
            
            print(f"\nProcessing time: {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error processing question: {e}")


def setup_paths(args):
    """Setup and validate paths."""
    # Convert to Path objects
    docs_dir = Path(args.docs_dir) if args.docs_dir else None
    
    # Setup processed directory if not specified
    if not args.processed_dir:
        if docs_dir:
            processed_dir = docs_dir.parent / "processed"
        else:
            processed_dir = Path("data/processed")
    else:
        processed_dir = Path(args.processed_dir)
    
    # Create directories if they don't exist
    if docs_dir and not docs_dir.exists():
        print(f"Documents directory {docs_dir} does not exist.")
        return None, None
    
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    return docs_dir, processed_dir


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Document QA Demo")
    parser.add_argument("--docs_dir", help="Directory with documents")
    parser.add_argument("--processed_dir", help="Directory for processed data")
    parser.add_argument("--use_generation", action="store_true", help="Use generation model")
    
    args = parser.parse_args()
    
    # Setup paths
    docs_dir, processed_dir = setup_paths(args)
    
    if not docs_dir and not processed_dir.exists():
        print("Either documents directory or processed data directory must be provided.")
        return
    
    # Initialize the QA system
    try:
        qa = DocumentQA(
            qa_model_name="deepset/gbert-base", 
            generation_model_name="google/flan-t5-base"
        )
        
        # Process documents
        if docs_dir:
            success = process_documents(qa, docs_dir, processed_dir)
            if not success:
                return
        elif processed_dir.exists():
            qa.doc_processor.load_processed_data(processed_dir)
        else:
            print("No documents or processed data available.")
            return
        
        # Initialize query engine for enhanced features
        query_engine = DocumentQueryEngine(qa)
        
        # Start interactive QA
        interactive_qa(qa, query_engine)
        
    except Exception as e:
        print(f"Error initializing the system: {e}")
        return


if __name__ == "__main__":
    main()
