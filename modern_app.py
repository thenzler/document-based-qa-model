"""
SCODi 4P Modern Document QA System
==================================

Modern implementation of document-based QA model with RAG capabilities
and consistent SCODi 4P design.
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from werkzeug.utils import secure_filename

# Add src directory to path
sys.path.append('./src')

from flask import Flask, render_template, request, jsonify, send_from_directory, url_for, redirect

# Import modules after path adjustment
try:
    from src.data_processing import DocumentProcessor
    # Try to import the RAG QA system first
    try:
        from src.qa_system_rag import DocumentQA
        print("RAG-based QA System successfully loaded")
        using_rag = True
    except ImportError as e:
        print(f"RAG-based QA System could not be loaded: {e}")
        print("Using standard QA System without RAG")
        try:
            from src.qa_system_llm import DocumentQA
            print("LLM-based QA System loaded")
            using_rag = False
            using_llm = True
        except ImportError as e:
            print(f"LLM-based QA System could not be loaded: {e}")
            print("Using basic QA System")
            from src.qa_system import DocumentQA
            using_rag = False
            using_llm = False
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure the src directory exists and contains the required modules.")
    sys.exit(1)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = Path('data/uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max file size
app.config['DOCS_FOLDER'] = Path('data/documents')
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'docx', 'md', 'html', 'xlsx', 'xml'}

# SCODi 4P Design Configuration
SCODI_DESIGN = {
    # Main color palette
    "colors": {
        "primary": "#007f78",       # Primary color (dark green/teal from SCODi logo)
        "secondary": "#4b5864",     # Secondary color (dark gray from navbar)
        "accent": "#f7f7f7",        # Accent color (light gray for backgrounds)
        "success": "#32a852",       # Success color (green)
        "warning": "#ffc107",       # Warning color (yellow)
        "error": "#dc3545",         # Error color (red)
        "info": "#17a2b8",          # Info color (blue)
    },
    # Page configuration
    "show_process_menu": True,
    "company_name": "SCODi Software AG",
    "app_version": "2.0",
    "current_year": datetime.now().year,
    # Design type (modern)
    "design_type": "modern",
    # Model info
    "using_rag": using_rag,
    "model_name": "RAG QA System" if using_rag else 
                 "LLM QA System" if 'using_llm' in locals() and using_llm else
                 "Basic QA System"
}

# Global variables
qa_system = None
doc_processor = None
recent_questions = []

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def init_system():
    global qa_system, doc_processor
    
    if qa_system is None:
        # Initialize QA System with GPU support if available
        use_gpu = True
        
        try:
            if 'using_rag' in globals() and using_rag:
                qa_system = DocumentQA(use_gpu=use_gpu)
                print("RAG-based QA System initialized")
            elif 'using_llm' in globals() and using_llm:
                qa_system = DocumentQA(use_gpu=use_gpu)
                print("LLM-based QA System initialized")
            else:
                qa_system = DocumentQA()
                print("Basic QA System initialized")
                
            # Process documents
            docs_dir = app.config['DOCS_FOLDER']
            if docs_dir.exists():
                qa_system.process_documents(str(docs_dir))
        except Exception as e:
            print(f"Error initializing QA system: {e}")
            # Create a basic DocumentQA instance as fallback
            from src.qa_system import DocumentQA as BasicDocumentQA
            qa_system = BasicDocumentQA()
            print("Fallback to basic QA System")
    
    if doc_processor is None:
        doc_processor = DocumentProcessor()

# Main routes with consistent SCODi 4P design
@app.route('/')
def index():
    return render_template('modern_index.html', 
                          design=SCODI_DESIGN,
                          page_title="SCODi 4P - Document QA System")

@app.route('/qa')
def qa_page():
    return render_template('modern_qa.html', 
                          recent_questions=recent_questions[-5:], 
                          design=SCODI_DESIGN,
                          page_title="SCODi 4P - Question & Answer")

@app.route('/documents')
def documents_page():
    init_system()
    docs = []
    
    if doc_processor and app.config['DOCS_FOLDER'].exists():
        docs = doc_processor.list_documents(str(app.config['DOCS_FOLDER']))
    
    return render_template('modern_documents.html', 
                          documents=docs, 
                          design=SCODI_DESIGN,
                          page_title="SCODi 4P - Document Management")

# API endpoints
@app.route('/api/ask', methods=['POST'])
def ask_question():
    global qa_system, recent_questions
    
    # Initialize QA system if needed
    init_system()
    
    # Get question from request
    question = request.json.get('question', '')
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    # Add to recent questions list
    if question not in recent_questions:
        recent_questions.insert(0, question)
        recent_questions = recent_questions[:10]  # Store max 10 questions
    
    # Answer question
    try:
        answer, sources = qa_system.answer_question(question)
        return jsonify({
            "answer": answer,
            "sources": sources
        })
    except Exception as e:
        print(f"Error answering question: {str(e)}")
        return jsonify({"error": f"Error answering question: {str(e)}"}), 500

@app.route('/api/documents', methods=['GET'])
def get_documents():
    init_system()
    try:
        docs = []
        if doc_processor and app.config['DOCS_FOLDER'].exists():
            docs = doc_processor.list_documents(str(app.config['DOCS_FOLDER']))
        return jsonify(docs)
    except Exception as e:
        return jsonify({"error": f"Error retrieving documents: {str(e)}"}), 500

@app.route('/api/documents/upload', methods=['POST'])
def upload_document():
    init_system()
    
    if 'file' not in request.files:
        return jsonify({"error": "No file in request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No filename provided"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        category = request.form.get('category', 'general')
        
        # Ensure directories exist
        os.makedirs(app.config['DOCS_FOLDER'], exist_ok=True)
        
        # Save file
        file_path = app.config['DOCS_FOLDER'] / filename
        file.save(file_path)
        
        # Process document
        try:
            if doc_processor:
                doc_processor.process_document(str(file_path), category)
            if qa_system:
                qa_system.add_document(str(file_path))
            
            return jsonify({
                "success": True,
                "message": f"Document '{filename}' successfully uploaded and processed.",
                "document": {
                    "filename": filename,
                    "path": str(file_path),
                    "category": category,
                    "upload_date": datetime.now().isoformat()
                }
            })
        except Exception as e:
            return jsonify({"error": f"Error processing document: {str(e)}"}), 500
    
    return jsonify({"error": "File format not allowed"}), 400

@app.route('/api/documents/process', methods=['POST'])
def process_documents():
    init_system()
    force_reprocess = request.json.get('forceReprocess', False)
    
    try:
        if doc_processor and app.config['DOCS_FOLDER'].exists():
            results = doc_processor.process_all_documents(
                str(app.config['DOCS_FOLDER']), 
                force_reprocess=force_reprocess
            )
        if qa_system:
            qa_system.process_documents(str(app.config['DOCS_FOLDER']))
        
        return jsonify({
            "success": True,
            "message": "Documents successfully processed.",
            "results": results if 'results' in locals() else {}
        })
    except Exception as e:
        return jsonify({"error": f"Error processing documents: {str(e)}"}), 500

@app.route('/api/qa/recent-questions', methods=['GET'])
def get_recent_questions():
    limit = int(request.args.get('limit', 5))
    return jsonify(recent_questions[:limit])

@app.route('/api/qa/answer', methods=['POST'])
def answer_question():
    global qa_system, recent_questions
    
    # Initialize QA system if needed
    init_system()
    
    # Get parameters from request
    question = request.json.get('question', '')
    use_generation = request.json.get('useGeneration', True)
    top_k = request.json.get('topK', 5)
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    # Add to recent questions list
    if question not in recent_questions:
        recent_questions.insert(0, question)
        recent_questions = recent_questions[:10]  # Store max 10 questions
    
    # Answer question
    try:
        start_time = datetime.now()
        
        # Generate answer
        answer, sources = qa_system.answer_question(
            question, 
            use_generation=use_generation,
            top_k=top_k
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Ensure answer is a string
        if not isinstance(answer, str):
            answer = str(answer)
            
        # Ensure sources is a list
        if not isinstance(sources, list):
            sources = []
            
        return jsonify({
            "answer": answer,
            "sources": sources,
            "processingTime": processing_time,
            "using_rag": SCODI_DESIGN["using_rag"]
        })
    except Exception as e:
        print(f"Error in answer_question: {str(e)}")
        return jsonify({"error": f"Error answering question: {str(e)}"}), 500

@app.route('/api/init', methods=['POST'])
def initialize_system():
    try:
        init_system()
        return jsonify({
            "success": True,
            "message": "System successfully initialized",
            "using_rag": SCODI_DESIGN["using_rag"],
            "model_name": SCODI_DESIGN["model_name"]
        })
    except Exception as e:
        return jsonify({"error": f"Error during initialization: {str(e)}"}), 500

@app.route('/api/system/status', methods=['GET'])
def system_status():
    """Returns the current system status"""
    return jsonify({
        "using_rag": SCODI_DESIGN["using_rag"],
        "model_name": SCODI_DESIGN["model_name"],
        "documents_loaded": len(qa_system.documents) if qa_system else 0,
        "initialized": qa_system is not None,
        "scodi_version": "4P",
        "design_type": SCODI_DESIGN["design_type"]
    })

if __name__ == '__main__':
    # Ensure upload directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['DOCS_FOLDER'], exist_ok=True)
    
    print(f"SCODi 4P Document QA System started with {SCODI_DESIGN['design_type']} design")
    print(f"Model: {SCODI_DESIGN['model_name']}")
    app.run(debug=True)
