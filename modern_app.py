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
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ModernApp")

# Add src directory to path
sys.path.append('./src')

from flask import Flask, render_template, request, jsonify, send_from_directory, url_for, redirect

# Import modules after path adjustment
try:
    from src.data_processing import DocumentProcessor
    # Import model trainer
    try:
        from src.train_model import get_latest_trained_models, ModelTrainer
        logger.info("Model training module successfully loaded")
        training_available = True
    except ImportError as e:
        logger.warning(f"Model training module could not be loaded: {e}")
        training_available = False
        
    # Try to import the RAG QA system first
    try:
        from src.qa_system_rag import DocumentQA
        logger.info("RAG-based QA System successfully loaded")
        using_rag = True
        using_llm = True
    except ImportError as e:
        logger.warning(f"RAG-based QA System could not be loaded: {e}")
        logger.info("Using standard QA System without RAG")
        try:
            from src.qa_system_llm import DocumentQA
            logger.info("LLM-based QA System loaded")
            using_rag = False
            using_llm = True
        except ImportError as e:
            logger.warning(f"LLM-based QA System could not be loaded: {e}")
            logger.info("Using basic QA System")
            from src.qa_system import DocumentQA
            using_rag = False
            using_llm = False
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure the src directory exists and contains the required modules.")
    sys.exit(1)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = Path('data/uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max file size
app.config['DOCS_FOLDER'] = Path('data/documents')
app.config['MODELS_FOLDER'] = Path('models')
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
    "using_llm": using_llm,
    "model_name": "RAG QA System" if using_rag else 
                 "LLM QA System" if using_llm else
                 "Basic QA System",
    # Training availability
    "training_available": training_available
}

# Global variables
qa_system = None
doc_processor = None
model_trainer = None
recent_questions = []
active_models = {}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def init_system():
    global qa_system, doc_processor, model_trainer, active_models
    
    if qa_system is None:
        # Initialize QA System with GPU support if available
        use_gpu = True
        
        try:
            # Check for trained models
            if 'training_available' in globals() and training_available:
                # Get latest trained models
                active_models = get_latest_trained_models(app.config['MODELS_FOLDER'])
                logger.info(f"Found trained models: {active_models}")
                
                # Initialize model trainer
                model_trainer = ModelTrainer(models_dir=app.config['MODELS_FOLDER'], use_gpu=use_gpu)
                
                # Initialize QA system with trained models if available
                if 'using_rag' in globals() and using_rag:
                    if active_models:
                        # Extract model paths from active_models
                        embedding_model = active_models.get('embedding_model', None)
                        cross_encoder_model = active_models.get('cross_encoder_model', None)
                        qa_model = active_models.get('qa_model', None)
                        
                        # Use trained models if available, otherwise use defaults
                        qa_system = DocumentQA(
                            embedding_model_name=embedding_model if embedding_model else "sentence-transformers/all-mpnet-base-v2",
                            cross_encoder_model_name=cross_encoder_model if cross_encoder_model else "cross-encoder/ms-marco-MiniLM-L-6-v2",
                            generation_model_name=qa_model if qa_model else "deepset/roberta-base-squad2",
                            use_gpu=use_gpu
                        )
                        logger.info("RAG-based QA System initialized with trained models")
                    else:
                        # No trained models, use defaults
                        qa_system = DocumentQA(use_gpu=use_gpu)
                        logger.info("RAG-based QA System initialized with default models")
                elif 'using_llm' in globals() and using_llm:
                    qa_system = DocumentQA(use_gpu=use_gpu)
                    logger.info("LLM-based QA System initialized")
                else:
                    qa_system = DocumentQA()
                    logger.info("Basic QA System initialized")
            else:
                # Training not available, use default initialization
                if 'using_rag' in globals() and using_rag:
                    qa_system = DocumentQA(use_gpu=use_gpu)
                    logger.info("RAG-based QA System initialized with default models")
                elif 'using_llm' in globals() and using_llm:
                    qa_system = DocumentQA(use_gpu=use_gpu)
                    logger.info("LLM-based QA System initialized")
                else:
                    qa_system = DocumentQA()
                    logger.info("Basic QA System initialized")
                
            # Process documents
            docs_dir = app.config['DOCS_FOLDER']
            if docs_dir.exists():
                qa_system.process_documents(str(docs_dir))
        except Exception as e:
            logger.error(f"Error initializing QA system: {e}")
            # Create a basic DocumentQA instance as fallback
            from src.qa_system import DocumentQA as BasicDocumentQA
            qa_system = BasicDocumentQA()
            logger.info("Fallback to basic QA System")
    
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

@app.route('/models')
def models_page():
    init_system()
    
    # Get model info
    model_info = {
        "active_models": active_models,
        "training_available": training_available
    }
    
    return render_template('modern_models.html', 
                          model_info=model_info, 
                          design=SCODI_DESIGN,
                          page_title="SCODi 4P - Model Management")

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
        logger.error(f"Error answering question: {str(e)}")
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
        logger.error(f"Error in answer_question: {str(e)}")
        return jsonify({"error": f"Error answering question: {str(e)}"}), 500

@app.route('/api/init', methods=['POST'])
def initialize_system():
    try:
        init_system()
        return jsonify({
            "success": True,
            "message": "System successfully initialized",
            "using_rag": SCODI_DESIGN["using_rag"],
            "model_name": SCODI_DESIGN["model_name"],
            "active_models": active_models
        })
    except Exception as e:
        return jsonify({"error": f"Error during initialization: {str(e)}\n{str(sys.exc_info())}"}), 500

@app.route('/api/models/train', methods=['POST'])
def train_model():
    global qa_system, model_trainer, active_models
    
    if not training_available or model_trainer is None:
        return jsonify({"error": "Model training is not available"}), 400
    
    # Get training parameters
    model_type = request.json.get('modelType', 'all')
    epochs = int(request.json.get('epochs', 3))
    
    try:
        # Initialize system if needed
        init_system()
        
        # Get document contents for training
        doc_contents = []
        if app.config['DOCS_FOLDER'].exists():
            for file_path in app.config['DOCS_FOLDER'].glob('*.txt'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        doc_contents.append(f.read())
                except Exception as e:
                    logger.error(f"Error reading {file_path}: {e}")
        
        if not doc_contents:
            return jsonify({"error": "No text documents found for training"}), 400
        
        # Generate training data
        embedding_data, cross_encoder_data, qa_data = model_trainer.generate_training_data_from_documents(
            doc_contents, num_examples=1000
        )
        
        # Train models based on type
        results = {}
        
        if model_type in ['all', 'embedding'] and embedding_data:
            embedding_path = model_trainer.train_embedding_model(
                embedding_data, epochs=epochs
            )
            if embedding_path:
                results['embedding_model'] = embedding_path
        
        if model_type in ['all', 'cross-encoder'] and cross_encoder_data:
            cross_encoder_path = model_trainer.train_cross_encoder(
                cross_encoder_data, epochs=epochs
            )
            if cross_encoder_path:
                results['cross_encoder_model'] = cross_encoder_path
        
        if model_type in ['all', 'qa'] and qa_data:
            qa_path = model_trainer.train_qa_model(
                qa_data, epochs=epochs
            )
            if qa_path:
                results['qa_model'] = qa_path
        
        if not results:
            return jsonify({"error": "No models were successfully trained"}), 500
        
        # Update active models
        active_models = model_trainer.get_active_models()
        
        # Reset QA system to use new models
        qa_system = None
        init_system()
        
        return jsonify({
            "success": True,
            "message": "Models successfully trained and activated",
            "trained_models": results,
            "active_models": active_models
        })
    except Exception as e:
        logger.error(f"Error training models: {str(e)}")
        return jsonify({"error": f"Error training models: {str(e)}"}), 500

@app.route('/api/system/status', methods=['GET'])
def system_status():
    """Returns the current system status"""
    # Get active models if not initialized
    global active_models
    if not active_models and training_available:
        try:
            active_models = get_latest_trained_models(app.config['MODELS_FOLDER'])
        except Exception as e:
            logger.error(f"Error getting active models: {e}")
    
    return jsonify({
        "using_rag": SCODI_DESIGN["using_rag"],
        "using_llm": SCODI_DESIGN["using_llm"],
        "model_name": SCODI_DESIGN["model_name"],
        "documents_loaded": len(qa_system.documents) if qa_system else 0,
        "initialized": qa_system is not None,
        "scodi_version": "4P",
        "design_type": SCODI_DESIGN["design_type"],
        "training_available": SCODI_DESIGN["training_available"],
        "active_models": active_models
    })

if __name__ == '__main__':
    # Ensure upload directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['DOCS_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODELS_FOLDER'], exist_ok=True)
    
    logger.info(f"SCODi 4P Document QA System started with {SCODI_DESIGN['design_type']} design")
    logger.info(f"Model: {SCODI_DESIGN['model_name']}")
    app.run(debug=True)
