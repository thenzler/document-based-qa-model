import os
import sys

# Pfad zum src-Verzeichnis hinzufügen
sys.path.append('./src')

from flask import Flask, render_template, request, jsonify, send_from_directory
import json
from pathlib import Path

# Erst nach dem Hinzufügen des Pfads importieren
try:
    from src.data_processing import DocumentProcessor
    from src.qa_system import DocumentQA
    from src.model_training import ChurnModel
except ImportError as e:
    print(f"Import-Fehler: {e}")
    print("Stellen Sie sicher, dass das src-Verzeichnis existiert und die benötigten Module enthält.")
    sys.exit(1)

app = Flask(__name__)

# Globale Variablen
qa_system = None
doc_processor = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/ask', methods=['POST'])
def ask_question():
    global qa_system
    
    # Initialisiere QA-System falls nötig
    if qa_system is None:
        try:
            qa_system = DocumentQA()
            # Dokumente verarbeiten
            docs_dir = Path("data/churn_docs")
            if docs_dir.exists():
                qa_system.process_documents(str(docs_dir))
        except Exception as e:
            return jsonify({"error": f"Fehler bei der Initialisierung: {str(e)}"}), 500
    
    # Hole Frage aus der Anfrage
    question = request.json.get('question', '')
    if not question:
        return jsonify({"error": "Keine Frage angegeben"}), 400
    
    # Beantworte Frage
    try:
        response = qa_system.answer_question(question)
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"Fehler beim Beantworten der Frage: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
