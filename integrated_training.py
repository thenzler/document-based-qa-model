#!/usr/bin/env python3
"""
Integriertes Training und Download für das dokumentenbasierte QA-System
======================================================================

Dieses Skript fügt eine zusätzliche Route zur bestehenden Flask-App hinzu,
um das Modelltraining und den Download zu ermöglichen.

Führen Sie diese Datei aus, anstatt die enhanced_app_with_training.py direkt zu starten.
"""

import os
import sys
import json
import time
import logging
import zipfile
import shutil
import uuid
from pathlib import Path
from datetime import datetime
from threading import Thread
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory, redirect, url_for, Blueprint

# Pfad zum src-Verzeichnis hinzufügen
sys.path.append('./src')

# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("integrated-training")

# Import der original app
try:
    from enhanced_app_with_training import app, init_system, qa_system
except ImportError:
    logger.error("Konnte enhanced_app_with_training.py nicht importieren.")
    logger.error("Bitte stellen Sie sicher, dass die Datei existiert und ausführbar ist.")
    sys.exit(1)

# Import der Trainingskomponenten
try:
    from src.local_model_trainer import LocalModelTrainer
except ImportError:
    logger.error("Konnte src/local_model_trainer.py nicht importieren.")
    logger.error("Bitte stellen Sie sicher, dass die Datei existiert.")
    sys.exit(1)

# Basisverzeichnisse
MODELS_DIR = Path('models/local')
MODELS_DIR.mkdir(exist_ok=True, parents=True)

# Trainings-Status
training_status = {
    "active": False,
    "progress": 0,
    "message": "Kein Training aktiv",
    "error": None,
    "completed": False,
    "model_path": None
}

# Trainings-Thread
training_thread = None

# Blueprint für Modelltraining
training_bp = Blueprint('training', __name__)

@training_bp.route('/training')
def training_page():
    """Seite für Modelltraining anzeigen"""
    return render_template('training.html', 
                          title="Modelltraining",
                          status=training_status)

@training_bp.route('/api/training/start', methods=['POST'])
def start_training():
    """Startet ein neues Training"""
    global training_status, training_thread
    
    if training_status["active"]:
        return jsonify({
            "success": False,
            "message": "Es läuft bereits ein Training"
        }), 400
    
    # Trainingskonfiguration aus dem Request
    data = request.json or {}
    
    base_model = data.get('base_model', 'distilbert-base-uncased')
    epochs = int(data.get('epochs', 3))
    batch_size = int(data.get('batch_size', 4))
    learning_rate = float(data.get('learning_rate', 5e-5))
    optimize = bool(data.get('optimize', True))
    
    # Prüfe, ob Dokumente vorhanden sind
    docs_dir = Path('data/documents')
    if not docs_dir.exists() or not any(docs_dir.glob("**/*.*")):
        return jsonify({
            "success": False,
            "message": "Keine Dokumente gefunden. Bitte laden Sie zuerst Dokumente hoch."
        }), 400
    
    # Training-Status initialisieren
    training_status.update({
        "active": True,
        "progress": 0,
        "message": "Training wird vorbereitet...",
        "error": None,
        "completed": False,
        "model_path": None,
        "config": {
            "base_model": base_model,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimize": optimize
        }
    })
    
    # Training in einem separaten Thread starten
    training_thread = Thread(
        target=train_model_thread,
        args=(base_model, epochs, batch_size, learning_rate, optimize)
    )
    training_thread.daemon = True
    training_thread.start()
    
    return jsonify({
        "success": True,
        "message": "Training gestartet"
    })

@training_bp.route('/api/training/status')
def get_training_status():
    """Gibt den aktuellen Trainingsstatus zurück"""
    return jsonify(training_status)

@training_bp.route('/api/training/cancel', methods=['POST'])
def cancel_training():
    """Bricht das laufende Training ab"""
    global training_status
    
    if not training_status["active"]:
        return jsonify({
            "success": False,
            "message": "Kein aktives Training zum Abbrechen"
        }), 400
    
    # Status zurücksetzen
    training_status.update({
        "active": False,
        "message": "Training abgebrochen",
        "error": "Manuell abgebrochen",
        "completed": False
    })
    
    return jsonify({
        "success": True,
        "message": "Training abgebrochen"
    })

@training_bp.route('/api/training/download')
def download_model():
    """Lädt das trainierte Modell herunter"""
    if not training_status["completed"] or not training_status["model_path"]:
        return jsonify({
            "success": False,
            "message": "Kein fertiges Modell zum Herunterladen verfügbar"
        }), 400
    
    model_path = training_status["model_path"]
    
    if not os.path.exists(model_path):
        return jsonify({
            "success": False,
            "message": f"Modelldatei nicht gefunden: {model_path}"
        }), 404
    
    # Modell als Attachment senden
    return send_file(
        model_path,
        as_attachment=True,
        download_name=os.path.basename(model_path)
    )

def train_model_thread(base_model, epochs, batch_size, learning_rate, optimize):
    """Führt das Training in einem separaten Thread durch"""
    global training_status
    
    try:
        # Ausgabeverzeichnis
        output_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = MODELS_DIR / f"train_{output_id}"
        output_dir.mkdir(exist_ok=True)
        
        # Modell-Trainer initialisieren
        trainer = LocalModelTrainer(
            base_model_name=base_model,
            model_type="qa",
            docs_dir="data/documents",
            output_dir=output_dir
        )
        
        # Status aktualisieren: Basis-Modell laden
        update_status(
            message="Lade Basis-Modell...",
            progress=5
        )
        
        # Basis-Modell laden
        if not trainer.load_base_model():
            raise Exception("Fehler beim Laden des Basis-Modells")
        
        # Status aktualisieren: Trainingsdaten vorbereiten
        update_status(
            message="Bereite Trainingsdaten vor...",
            progress=15
        )
        
        # Trainingsdaten vorbereiten
        train_file = trainer.prepare_training_data()
        if not train_file:
            raise Exception("Fehler bei der Vorbereitung der Trainingsdaten")
        
        # Status aktualisieren: Modell trainieren
        update_status(
            message=f"Trainiere Modell (Epoch 1/{epochs})...",
            progress=20
        )
        
        # Modell trainieren
        if not trainer.train_model(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate):
            raise Exception("Fehler beim Training des Modells")
        
        # Status aktualisieren: Modell optimieren
        if optimize:
            update_status(
                message="Optimiere Modell...",
                progress=80
            )
            
            # Modell optimieren
            if not trainer.optimize_model(quantize=True, onnx_export=True):
                logger.warning("Fehler bei der Modelloptimierung, fahre trotzdem fort")
        
        # Status aktualisieren: Modell verpacken
        update_status(
            message="Verpacke Modell zur Distribution...",
            progress=90
        )
        
        # Modell verpacken
        package_path = trainer.package_model()
        if not package_path:
            raise Exception("Fehler beim Verpacken des Modells")
        
        # Training erfolgreich abgeschlossen
        update_status(
            message="Training erfolgreich abgeschlossen",
            progress=100,
            completed=True,
            model_path=package_path
        )
        
    except Exception as e:
        logger.error(f"Fehler beim Training: {e}")
        update_status(
            message=f"Fehler beim Training: {str(e)}",
            progress=0,
            error=str(e)
        )

def update_status(message=None, progress=None, error=None, completed=None, model_path=None):
    """Aktualisiert den Trainingsstatus"""
    global training_status
    
    if message is not None:
        training_status["message"] = message
    
    if progress is not None:
        training_status["progress"] = progress
    
    if error is not None:
        training_status["error"] = error
        training_status["active"] = False
    
    if completed is not None:
        training_status["completed"] = completed
        if completed:
            training_status["active"] = False
    
    if model_path is not None:
        training_status["model_path"] = model_path

# Hauptfunktion
def main():
    # Blueprint registrieren
    app.register_blueprint(training_bp)
    
    # App starten
    print("Starte erweiterte QA-App mit integriertem Modelltraining")
    app.run(debug=True, host='0.0.0.0')

if __name__ == "__main__":
    main()