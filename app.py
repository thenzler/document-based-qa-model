"""
Flask Web Application for Document-based QA and Churn Prediction

This module provides a web interface for:
1. Document-based question answering with source citations
2. Churn prediction with explanations
3. Document management and visualization
"""

import os
import sys

# Pfad zum src-Verzeichnis hinzufügen
sys.path.append('./src')

# Erst nach dem Hinzufügen des Pfads importieren
from src.data_processing import DocumentProcessor
from src.qa_system import DocumentQA, ChurnQASystem
from src.model_training import ChurnModel, DocumentEnhancedChurnModel

# Initialize constants
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
CHURN_DOCS_DIR = DATA_DIR / "churn_docs"
MODELS_DIR = Path("models")
UPLOAD_DIR = DATA_DIR / "uploads"

# Ensure directories exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['UPLOAD_FOLDER'] = str(UPLOAD_DIR)

# Initialize global variables
qa_system = None
churn_model = None
doc_processor = None
processing_status = {"status": "idle", "message": ""}
is_processing = False
customer_data = None


def initialize_qa_system(processed_dir=None, use_generation=True):
    """Initialize the QA system with appropriate models"""
    global qa_system, processing_status
    
    if processed_dir is None:
        processed_dir = PROCESSED_DIR
    
    processing_status = {"status": "initializing", "message": "Initializing QA system..."}
    
    # Initialize with appropriate models
    qa_model_name = "deepset/gbert-base" if use_generation else "distilbert-base-uncased-distilled-squad"
    generation_model_name = "google/flan-t5-base" if use_generation else None
    
    # Create specialized churn QA system
    qa_system = ChurnQASystem(
        embedding_model_name="sentence-transformers/all-mpnet-base-v2",
        qa_model_name=qa_model_name,
        generation_model_name=generation_model_name,
        processed_data_dir=str(processed_dir) if processed_dir.exists() else None
    )
    
    processing_status = {"status": "ready", "message": "QA system initialized"}
    return qa_system


def process_documents(docs_dir=None, save_dir=None, force_reprocess=False):
    """Process documents for the QA system"""
    global doc_processor, qa_system, processing_status, is_processing
    
    is_processing = True
    
    if docs_dir is None:
        docs_dir = CHURN_DOCS_DIR
    
    if save_dir is None:
        save_dir = PROCESSED_DIR
    
    # Check if processed data already exists
    if save_dir.exists() and not force_reprocess and list(save_dir.glob("*")):
        processing_status = {"status": "loading", "message": "Loading pre-processed documents..."}
        doc_processor = DocumentProcessor()
        doc_processor.load_processed_data(save_dir)
        
        # Update QA system with loaded data
        if qa_system:
            qa_system.doc_processor = doc_processor
        
        processing_status = {"status": "ready", "message": "Documents loaded successfully"}
        is_processing = False
        return doc_processor
    
    # Process documents
    processing_status = {"status": "processing", "message": "Processing documents..."}
    doc_processor = DocumentProcessor()
    
    # Process in a separate thread to not block the UI
    def process_thread():
        global is_processing, processing_status
        try:
            doc_processor.process_pipeline(docs_dir)
            
            if save_dir:
                processing_status = {"status": "saving", "message": "Saving processed documents..."}
                doc_processor.save_processed_data(save_dir)
            
            # Update QA system with processed data
            if qa_system:
                qa_system.doc_processor = doc_processor
            
            processing_status = {"status": "ready", "message": "Documents processed successfully"}
        except Exception as e:
            processing_status = {"status": "error", "message": f"Error processing documents: {str(e)}"}
        
        is_processing = False
    
    threading.Thread(target=process_thread).start()
    return doc_processor


def load_churn_model(model_path=None):
    """Load the churn prediction model"""
    global churn_model, processing_status
    
    if model_path is None:
        model_path = MODELS_DIR / "churn_model.pkl"
    
    processing_status = {"status": "loading", "message": "Loading churn model..."}
    
    # Check if model exists
    if not model_path.exists():
        # Create sample data and train a model
        processing_status = {"status": "training", "message": "Training churn model..."}
        
        # Create sample data
        data = pd.read_csv(DATA_DIR / "customer_data.csv")
        
        # Train a model
        churn_model = DocumentEnhancedChurnModel("RandomForest")
        churn_model.train(data)
        churn_model.save_model(model_path)
    else:
        # Load existing model
        churn_model = DocumentEnhancedChurnModel("RandomForest")
        churn_model.load_model(model_path)
    
    processing_status = {"status": "ready", "message": "Churn model loaded"}
    return churn_model


def get_document_stats():
    """Get statistics about the processed documents"""
    if doc_processor is None or not doc_processor.document_store:
        return {
            "total_documents": 0,
            "chunks": 0,
            "avg_chunk_size": 0,
            "sources": []
        }
    
    # Calculate stats
    doc_sources = {}
    total_text_length = 0
    
    for doc in doc_processor.document_store:
        source = doc["source"]
        if source in doc_sources:
            doc_sources[source] += 1
        else:
            doc_sources[source] = 1
        
        total_text_length += len(doc["text"])
    
    # Format sources for display
    sources = [
        {"name": Path(source).name, "chunks": count} 
        for source, count in doc_sources.items()
    ]
    
    return {
        "total_documents": len(doc_sources),
        "chunks": len(doc_processor.document_store),
        "avg_chunk_size": total_text_length // max(1, len(doc_processor.document_store)),
        "sources": sources
    }


@app.route('/')
def index():
    """Render the home page"""
    return render_template('index.html')


@app.route('/qa')
def qa_page():
    """Render the QA page"""
    return render_template('qa.html')


@app.route('/churn')
def churn_page():
    """Render the churn prediction page"""
    return render_template('churn.html')


@app.route('/documents')
def documents_page():
    """Render the documents management page"""
    return render_template('documents.html')


@app.route('/api/status')
def get_status():
    """Get the current processing status"""
    global processing_status
    
    # Add additional info to status
    status_data = processing_status.copy()
    
    # Add system initialization status
    status_data["qa_system_ready"] = qa_system is not None
    status_data["doc_processor_ready"] = doc_processor is not None and bool(doc_processor.document_store)
    status_data["churn_model_ready"] = churn_model is not None
    status_data["is_processing"] = is_processing
    
    # Add document stats if available
    if doc_processor is not None and doc_processor.document_store:
        status_data["document_stats"] = get_document_stats()
    
    return jsonify(status_data)


@app.route('/api/initialize', methods=['POST'])
def initialize_system():
    """Initialize the QA system and other components"""
    global qa_system, doc_processor, churn_model, processing_status
    
    # Get initialization parameters
    use_generation = request.json.get('use_generation', True)
    force_reprocess = request.json.get('force_reprocess', False)
    
    # Initialize QA system
    if qa_system is None:
        qa_system = initialize_qa_system(use_generation=use_generation)
    
    # Process documents
    if doc_processor is None or not doc_processor.document_store or force_reprocess:
        doc_processor = process_documents(force_reprocess=force_reprocess)
    
    # Load churn model
    if churn_model is None:
        churn_model = load_churn_model()
    
    return jsonify({"status": "initializing", "message": "System initialization started"})


@app.route('/api/ask', methods=['POST'])
def ask_question():
    """Answer a question using the QA system"""
    global qa_system
    
    # Check if QA system is initialized
    if qa_system is None:
        return jsonify({"error": "QA system not initialized"}), 400
    
    # Get question from request
    question = request.json.get('question', '')
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    # Answer question
    try:
        response = qa_system.answer_churn_question(question)
        
        # Format sources for better display
        formatted_sources = []
        for i, source in enumerate(response.get("sources", [])):
            filename = Path(source.get("source", "")).name
            section = source.get("section", "")
            score = source.get("relevance_score", 0)
            
            formatted_source = {
                "id": i + 1,
                "filename": filename,
                "section": section,
                "score": f"{score:.2f}"
            }
            
            # Add matching sentences if available
            matching_sentences = source.get("matching_sentences", [])
            if matching_sentences:
                formatted_source["evidence"] = [
                    s[:100] + ('...' if len(s) > 100 else '') 
                    for s in matching_sentences
                ]
            
            formatted_sources.append(formatted_source)
        
        # Create response
        result = {
            "answer": response.get("answer", ""),
            "sources": formatted_sources,
            "processing_time": response.get("processing_time", 0),
            "query_variations": response.get("query_variations", [])
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": f"Error answering question: {str(e)}"}), 500


@app.route('/api/predict-churn', methods=['POST'])
def predict_churn():
    """Make churn predictions for customer data"""
    global churn_model
    
    # Check if churn model is initialized
    if churn_model is None:
        return jsonify({"error": "Churn model not initialized"}), 400
    
    # Get customer data
    customer_data = request.json.get('data')
    if not customer_data:
        return jsonify({"error": "No customer data provided"}), 400
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(customer_data)
        
        # Make predictions
        predictions = churn_model.predict_with_interventions(df)
        
        # Format results for JSON response
        results = []
        for i, row in predictions.iterrows():
            customer = {col: df.iloc[i][col] for col in df.columns}
            
            # Add prediction results
            prediction = {
                "customer_id": customer.get("customer_id", f"C{i+1}"),
                "churn_probability": float(row["churn_probability"]),
                "risk_category": row["risk_category"],
                "top_risk_factors": row.get("top_risk_factors", "").split(", ") if "top_risk_factors" in row else [],
                "recommended_interventions": row.get("recommended_interventions", "").split(", ") if "recommended_interventions" in row else []
            }
            
            # Get document explanations
            explanation = churn_model.explain_prediction_with_documents(df.iloc[i])
            
            # Add document references
            prediction["document_references"] = [
                {"filename": ref["filename"], "section": ref["section"]}
                for ref in explanation["document_references"]
            ]
            
            results.append(prediction)
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({"error": f"Error making predictions: {str(e)}"}), 500


@app.route('/api/upload-document', methods=['POST'])
def upload_document():
    """Upload a new document to the system"""
    global processing_status
    
    # Check if file is in request
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    
    # Check if filename is empty
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Save the file
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)
    
    # Process the document in background
    processing_status = {"status": "uploading", "message": f"Uploaded {file.filename}. Processing..."}
    
    # Trigger reprocessing of documents
    process_documents(docs_dir=UPLOAD_DIR, force_reprocess=True)
    
    return jsonify({"message": f"File {file.filename} uploaded successfully. Processing started."})


@app.route('/api/upload-customer-data', methods=['POST'])
def upload_customer_data():
    """Upload customer data for churn prediction"""
    global customer_data
    
    # Check if file is in request
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    
    # Check if filename is empty
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Check file extension
    if not file.filename.endswith('.csv'):
        return jsonify({"error": "Only CSV files are supported"}), 400
    
    try:
        # Parse CSV
        customer_data = pd.read_csv(file)
        
        # Convert to JSON for response
        data_json = customer_data.to_dict(orient='records')
        
        return jsonify({"message": "Customer data uploaded successfully", "data": data_json})
    
    except Exception as e:
        return jsonify({"error": f"Error parsing CSV: {str(e)}"}), 500


@app.route('/api/document-list')
def get_document_list():
    """Get a list of all documents in the system"""
    global doc_processor
    
    if doc_processor is None or not doc_processor.document_store:
        return jsonify([])
    
    # Get unique document sources
    doc_sources = {}
    for doc in doc_processor.document_store:
        source = doc["source"]
        filename = Path(source).name
        
        if filename in doc_sources:
            doc_sources[filename]["chunks"] += 1
        else:
            doc_sources[filename] = {
                "filename": filename,
                "path": source,
                "chunks": 1,
                "size": os.path.getsize(source) if os.path.exists(source) else 0
            }
    
    return jsonify(list(doc_sources.values()))


@app.route('/api/document-content/<path:filename>')
def get_document_content(filename):
    """Get the content of a specific document"""
    global doc_processor
    
    if doc_processor is None or not doc_processor.document_store:
        return jsonify({"error": "No documents processed"}), 404
    
    # Find document chunks with the given filename
    doc_chunks = []
    for doc in doc_processor.document_store:
        if Path(doc["source"]).name == filename:
            doc_chunks.append({
                "section": doc.get("section", ""),
                "text": doc["text"],
                "keywords": doc.get("keywords", [])
            })
    
    if not doc_chunks:
        return jsonify({"error": f"Document {filename} not found"}), 404
    
    return jsonify(doc_chunks)


@app.route('/static/<path:path>')
def send_static(path):
    """Serve static files"""
    return send_from_directory('static', path)


if __name__ == '__main__':
    # Initialize system components
    initialize_qa_system()
    process_documents()
    load_churn_model()
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
