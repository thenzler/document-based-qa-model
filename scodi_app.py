"""
SCODi 4P - Dokumentenbasiertes QA-Modell
=======================================

Erweiterte Version des document-based-qa-model mit SCODi 4P Design
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from werkzeug.utils import secure_filename

# Pfad zum src-Verzeichnis hinzufügen
sys.path.append('./src')

from flask import Flask, render_template, request, jsonify, send_from_directory, url_for, redirect

# Erst nach dem Hinzufügen des Pfads importieren
try:
    from src.data_processing import DocumentProcessor
    # Versuche, das LLM-basierte QA-System zu importieren
    try:
        from src.qa_system_llm import DocumentQA
        print("LLM-basiertes QA-System erfolgreich geladen")
        using_llm = True
    except ImportError as e:
        print(f"LLM-basiertes QA-System konnte nicht geladen werden: {e}")
        print("Verwende Standard-QA-System ohne LLM")
        from src.qa_system import DocumentQA
        using_llm = False
    
    from src.model_training import ChurnModel
except ImportError as e:
    print(f"Import-Fehler: {e}")
    print("Stellen Sie sicher, dass das src-Verzeichnis existiert und die benötigten Module enthält.")
    sys.exit(1)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = Path('data/uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB maximale Dateigröße
app.config['DOCS_FOLDER'] = Path('data/documents')
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'docx', 'md', 'html', 'xlsx', 'xml'}

# SCODi 4P Design-Konfiguration
SCODI_DESIGN = {
    # Hauptfarbpalette
    "colors": {
        "primary": "#007f78",       # Primärfarbe (Dunkelgrün/Türkis aus SCODi Logo)
        "secondary": "#4b5864",     # Sekundärfarbe (Dunkelgrau aus der Navigationsleiste)
        "accent": "#f7f7f7",        # Akzentfarbe (Hellgrau für Hintergründe)
        "success": "#32a852",       # Erfolgsfarbe (Grün)
        "warning": "#ffc107",       # Warnfarbe (Gelb)
        "error": "#dc3545",         # Fehlerfarbe (Rot)
        "info": "#17a2b8",          # Infofarbe (Blau)
    },
    # Seiten-Konfiguration
    "show_process_menu": True,      # Zeigt den Prozess-Menüpunkt in der Navigation
    "company_name": "SCODi Software AG",
    "app_version": "1.0",
    "current_year": datetime.now().year
}

# Globale Variablen
qa_system = None
doc_processor = None
churn_model = None
recent_questions = []

# Status-Flag für LLM-Nutzung
is_using_llm = using_llm

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def init_system():
    global qa_system, doc_processor, churn_model
    
    if qa_system is None:
        # Initialisiere QA-System mit GPU-Unterstützung, falls verfügbar
        if is_using_llm:
            try:
                qa_system = DocumentQA(use_gpu=True)
                print("QA-System mit LLM-Unterstützung initialisiert")
            except Exception as e:
                print(f"Fehler bei der LLM-Initialisierung: {e}")
                # Fallback auf nicht-LLM Version
                from src.qa_system import DocumentQA as StandardDocumentQA
                qa_system = StandardDocumentQA()
                print("Fallback auf Standard-QA-System")
        else:
            qa_system = DocumentQA()
            print("Standard-QA-System initialisiert")
            
        # Dokumente verarbeiten
        docs_dir = app.config['DOCS_FOLDER']
        if docs_dir.exists():
            qa_system.process_documents(str(docs_dir))
    
    if doc_processor is None:
        doc_processor = DocumentProcessor()
    
    if churn_model is None:
        churn_model = ChurnModel()

# Hauptrouten mit SCODi 4P Design
@app.route('/')
def index():
    return render_template('index.html', 
                          using_llm=is_using_llm, 
                          design=SCODI_DESIGN,
                          page_title="SCODi 4P - Dokumentenbasiertes QA")

@app.route('/qa')
def qa_page():
    return render_template('qa.html', 
                          recent_questions=recent_questions[-5:], 
                          using_llm=is_using_llm,
                          design=SCODI_DESIGN,
                          page_title="SCODi 4P - Frage & Antwort")

@app.route('/documents')
def documents_page():
    init_system()
    docs = []
    
    if doc_processor and app.config['DOCS_FOLDER'].exists():
        docs = doc_processor.list_documents(str(app.config['DOCS_FOLDER']))
    
    return render_template('documents.html', 
                          documents=docs, 
                          using_llm=is_using_llm,
                          design=SCODI_DESIGN,
                          page_title="SCODi 4P - Dokumentenmanagement")

@app.route('/churn')
def churn_page():
    return render_template('churn.html', 
                          using_llm=is_using_llm,
                          design=SCODI_DESIGN,
                          page_title="SCODi 4P - Churn-Prediction")

# API-Endpunkte (Funktionalität bleibt gleich, verbesserte Fehlermeldungen)
@app.route('/api/ask', methods=['POST'])
def ask_question():
    global qa_system, recent_questions
    
    # Initialisiere QA-System falls nötig
    init_system()
    
    # Hole Frage aus der Anfrage
    question = request.json.get('question', '')
    if not question:
        return jsonify({"error": "Keine Frage angegeben"}), 400
    
    # Zur Liste der kürzlich gestellten Fragen hinzufügen
    if question not in recent_questions:
        recent_questions.insert(0, question)
        recent_questions = recent_questions[:10]  # Maximal 10 Fragen speichern
    
    # Beantworte Frage
    try:
        answer, sources = qa_system.answer_question(question)
        return jsonify({
            "answer": answer,
            "sources": sources
        })
    except Exception as e:
        print(f"Fehler beim Beantworten der Frage: {str(e)}")
        return jsonify({
            "error": f"Fehler beim Beantworten der Frage: {str(e)}",
            "errorType": "QAError"
        }), 500

@app.route('/api/documents', methods=['GET'])
def get_documents():
    init_system()
    try:
        docs = []
        if doc_processor and app.config['DOCS_FOLDER'].exists():
            docs = doc_processor.list_documents(str(app.config['DOCS_FOLDER']))
        return jsonify(docs)
    except Exception as e:
        return jsonify({
            "error": f"Fehler beim Abrufen der Dokumente: {str(e)}",
            "errorType": "DocumentError"
        }), 500

@app.route('/api/documents/upload', methods=['POST'])
def upload_document():
    init_system()
    
    if 'file' not in request.files:
        return jsonify({"error": "Keine Datei im Request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Kein Dateiname angegeben"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        category = request.form.get('category', 'allgemein')
        
        # Stelle sicher, dass die Verzeichnisse existieren
        os.makedirs(app.config['DOCS_FOLDER'], exist_ok=True)
        
        # Speichere Datei
        file_path = app.config['DOCS_FOLDER'] / filename
        file.save(file_path)
        
        # Verarbeite Dokument
        try:
            if doc_processor:
                doc_processor.process_document(str(file_path), category)
            if qa_system:
                qa_system.add_document(str(file_path))
            
            return jsonify({
                "success": True,
                "message": f"Dokument '{filename}' erfolgreich hochgeladen und verarbeitet.",
                "document": {
                    "filename": filename,
                    "path": str(file_path),
                    "category": category,
                    "upload_date": datetime.now().isoformat()
                }
            })
        except Exception as e:
            return jsonify({
                "error": f"Fehler bei der Verarbeitung des Dokuments: {str(e)}",
                "errorType": "ProcessingError"
            }), 500
    
    return jsonify({"error": "Dateiformat nicht erlaubt"}), 400

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
            "message": "Dokumente erfolgreich verarbeitet.",
            "results": results if 'results' in locals() else {}
        })
    except Exception as e:
        return jsonify({
            "error": f"Fehler bei der Verarbeitung der Dokumente: {str(e)}",
            "errorType": "BatchProcessingError"
        }), 500

@app.route('/api/qa/recent-questions', methods=['GET'])
def get_recent_questions():
    limit = int(request.args.get('limit', 5))
    return jsonify(recent_questions[:limit])

@app.route('/api/qa/answer', methods=['POST'])
def answer_question():
    global qa_system, recent_questions
    
    # Initialisiere QA-System falls nötig
    init_system()
    
    # Parameter aus der Anfrage holen
    question = request.json.get('question', '')
    use_generation = request.json.get('useGeneration', True)
    top_k = request.json.get('topK', 5)
    
    if not question:
        return jsonify({"error": "Keine Frage angegeben"}), 400
    
    # Zur Liste der kürzlich gestellten Fragen hinzufügen
    if question not in recent_questions:
        recent_questions.insert(0, question)
        recent_questions = recent_questions[:10]  # Maximal 10 Fragen speichern
    
    # Beantworte Frage
    try:
        start_time = datetime.now()
        
        # Antwort generieren
        answer, sources = qa_system.answer_question(
            question, 
            use_generation=use_generation,
            top_k=top_k
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Stelle sicher, dass die Antwort ein String ist
        if not isinstance(answer, str):
            answer = str(answer)
            
        # Stelle sicher, dass die Quellen eine Liste sind
        if not isinstance(sources, list):
            sources = []
            
        return jsonify({
            "answer": answer,
            "sources": sources,
            "processingTime": processing_time,
            "using_llm": is_using_llm
        })
    except Exception as e:
        print(f"Fehler bei answer_question: {str(e)}")
        return jsonify({
            "error": f"Fehler beim Beantworten der Frage: {str(e)}",
            "errorType": "AnswerGenerationError"
        }), 500

@app.route('/api/churn/predict', methods=['POST'])
def predict_churn():
    init_system()
    
    try:
        # Kundendaten aus dem Request holen
        data = request.json.get('data', [])
        if not data:
            # Überprüfen, ob eine Datei hochgeladen wurde
            if 'file' in request.files:
                file = request.files['file']
                if file.filename != '':
                    # CSV-Datei verarbeiten
                    # Hier müsste eigentlich Code zur CSV-Verarbeitung stehen
                    pass
        
        # Churn-Vorhersage durchführen
        if churn_model:
            predictions = churn_model.predict(data)
            return jsonify(predictions)
        else:
            return jsonify({
                "error": "Churn-Modell nicht initialisiert",
                "errorType": "ModelNotInitialized"
            }), 500
    except Exception as e:
        return jsonify({
            "error": f"Fehler bei der Churn-Vorhersage: {str(e)}",
            "errorType": "PredictionError"
        }), 500

@app.route('/api/init', methods=['POST'])
def initialize_system():
    try:
        init_system()
        return jsonify({
            "success": True,
            "message": "System erfolgreich initialisiert",
            "using_llm": is_using_llm
        })
    except Exception as e:
        return jsonify({
            "error": f"Fehler bei der Initialisierung: {str(e)}",
            "errorType": "InitializationError"
        }), 500

@app.route('/api/system/status', methods=['GET'])
def system_status():
    """Gibt den aktuellen Status des Systems zurück"""
    return jsonify({
        "using_llm": is_using_llm,
        "documents_loaded": len(qa_system.documents) if qa_system else 0,
        "initialized": qa_system is not None,
        "scodi_version": "4P",
        "design_theme": "scodi-4p"
    })

if __name__ == '__main__':
    # Stelle sicher, dass die Upload-Verzeichnisse existieren
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['DOCS_FOLDER'], exist_ok=True)
    
    print(f"SCODi 4P QA-System gestartet")
    print(f"LLM-Unterstützung: {'Aktiviert' if is_using_llm else 'Deaktiviert'}")
    app.run(debug=True)
