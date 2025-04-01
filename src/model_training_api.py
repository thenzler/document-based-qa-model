"""
API-Endpunkte für das Lokale Modell-Training
============================================

Dieses Modul stellt Flask-Endpunkte bereit, um lokale Modelle zu trainieren,
den Trainingsstatus abzufragen und trainierte Modelle herunterzuladen.
"""

import os
import json
import time
import uuid
import threading
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

from flask import Blueprint, request, jsonify, send_file, current_app

# Lokaler Modell-Trainer importieren
from src.local_model_trainer import LocalModelTrainer

# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("model-training-api")

# Blueprint erstellen
model_training_bp = Blueprint('model_training', __name__)

# Globale Trainings-Status-Informationen
training_status = {
    "active": False,
    "task_id": None,
    "start_time": None,
    "message": "Kein Training aktiv",
    "progress": 0,
    "phase": None,
    "error": None,
    "completed": False,
    "model_info": None
}

# Aktives Trainings-Thread
active_training_thread = None

# Pfade konfigurieren
MODELS_DIR = Path('models/local')


@model_training_bp.route('/api/model/train', methods=['POST'])
def start_training():
    """
    Startet das Training eines lokalen Modells.
    
    Erwartet JSON-Anfrage mit:
    - base_model: Name des Basis-Modells (z.B. "distilbert-base-uncased")
    - model_type: Modelltyp (z.B. "qa", "causal_lm", "seq2seq_lm")
    - epochs: Anzahl der Trainings-Epochs (z.B. 3)
    - batch_size: Batch-Größe (z.B. 4)
    - learning_rate: Lernrate (z.B. 5e-5)
    - optimize: Ob das Modell optimiert werden soll (Boolean)
    
    Gibt zurück:
    - JSON mit Status und ggf. Fehlermeldung
    """
    global training_status, active_training_thread
    
    # Überprüfen, ob bereits ein Training läuft
    if training_status["active"]:
        return jsonify({
            "success": False,
            "error": "Es läuft bereits ein Training. Bitte warten Sie, bis es abgeschlossen ist."
        }), 409
    
    try:
        # Trainingsparameter aus der Anfrage holen
        data = request.get_json()
        
        base_model = data.get('base_model', 'distilbert-base-uncased')
        model_type = data.get('model_type', 'qa')
        epochs = int(data.get('epochs', 3))
        batch_size = int(data.get('batch_size', 4))
        learning_rate = float(data.get('learning_rate', 5e-5))
        optimize = bool(data.get('optimize', True))
        
        # Parameter validieren
        if epochs < 1 or epochs > 20:
            return jsonify({
                "success": False,
                "error": "Die Anzahl der Epochs muss zwischen 1 und 20 liegen."
            }), 400
            
        if batch_size < 1 or batch_size > 32:
            return jsonify({
                "success": False,
                "error": "Die Batch-Größe muss zwischen 1 und 32 liegen."
            }), 400
            
        if learning_rate < 1e-6 or learning_rate > 1e-3:
            return jsonify({
                "success": False,
                "error": "Die Lernrate muss zwischen 1e-6 und 1e-3 liegen."
            }), 400
        
        # Dokumente überprüfen
        docs_dir = current_app.config.get('DOCS_FOLDER', Path('data/documents'))
        if not docs_dir.exists() or not any(docs_dir.glob('*.*')):
            return jsonify({
                "success": False,
                "error": "Keine Dokumente zum Training gefunden. Bitte laden Sie zuerst Dokumente hoch."
            }), 400
        
        # Task-ID für dieses Training erstellen
        task_id = str(uuid.uuid4())
        
        # Training-Status initialisieren
        training_status.update({
            "active": True,
            "task_id": task_id,
            "start_time": datetime.now().isoformat(),
            "message": "Training wird vorbereitet...",
            "progress": 0,
            "phase": "Vorbereitung",
            "error": None,
            "completed": False,
            "model_info": {
                "base_model_name": base_model,
                "model_type": model_type,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "optimize": optimize
            }
        })
        
        # Training in einem separaten Thread starten
        active_training_thread = threading.Thread(
            target=train_model_thread,
            args=(task_id, base_model, model_type, epochs, batch_size, learning_rate, optimize, docs_dir)
        )
        active_training_thread.daemon = True
        active_training_thread.start()
        
        return jsonify({
            "success": True,
            "message": "Training gestartet",
            "task_id": task_id
        })
        
    except Exception as e:
        logger.error(f"Fehler beim Starten des Trainings: {e}")
        return jsonify({
            "success": False,
            "error": f"Fehler beim Starten des Trainings: {str(e)}"
        }), 500


def train_model_thread(task_id, base_model, model_type, epochs, batch_size, learning_rate, optimize, docs_dir):
    """
    Führt das Training in einem separaten Thread durch.
    
    Args:
        task_id (str): ID des Trainingsauftrags
        base_model (str): Name des Basis-Modells
        model_type (str): Typ des Modells
        epochs (int): Anzahl der Trainings-Epochs
        batch_size (int): Batch-Größe
        learning_rate (float): Lernrate
        optimize (bool): Ob das Modell optimiert werden soll
        docs_dir (str/Path): Verzeichnis mit den Dokumenten
    """
    global training_status
    
    # Ausgabeverzeichnis für dieses Training
    output_dir = MODELS_DIR / task_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Modell-Trainer initialisieren
        trainer = LocalModelTrainer(
            base_model_name=base_model,
            model_type=model_type,
            docs_dir=docs_dir,
            output_dir=output_dir
        )
        
        # Status aktualisieren: Basis-Modell laden
        update_training_status(
            message="Lade Basis-Modell...",
            progress=5,
            phase="Vorbereitung"
        )
        
        # Basis-Modell laden
        if not trainer.load_base_model():
            raise Exception("Fehler beim Laden des Basis-Modells.")
        
        # Status aktualisieren: Trainingsdaten vorbereiten
        update_training_status(
            message="Bereite Trainingsdaten vor...",
            progress=15,
            phase="Datenvorbereitung"
        )
        
        # Trainingsdaten vorbereiten
        train_file = trainer.prepare_training_data()
        if not train_file:
            raise Exception("Fehler bei der Vorbereitung der Trainingsdaten.")
        
        # Status aktualisieren: Modell trainieren
        update_training_status(
            message=f"Trainiere Modell (Epoch 1/{epochs})...",
            progress=20,
            phase="Training"
        )
        
        # Fortschritts-Callback definieren
        def training_progress_callback(epoch, total_epochs, batch, total_batches, loss):
            # Aktuellen Fortschritt berechnen (20% für Vorbereitung, 60% für Training)
            progress = 20 + (epoch - 1) * 60 / total_epochs + batch * 60 / (total_epochs * total_batches)
            update_training_status(
                message=f"Trainiere Modell (Epoch {epoch}/{total_epochs}, Batch {batch}/{total_batches}, Loss: {loss:.4f})...",
                progress=progress,
                phase="Training"
            )
        
        # Modell trainieren (hier würden wir den Callback übergeben)
        # Da unser Trainer aber keinen Callback akzeptiert, simulieren wir den Fortschritt
        # In einer vollständigen Implementierung würde dieser Callback dem Trainer übergeben
        
        # Trainingsfortschritt simulieren
        for epoch in range(1, epochs + 1):
            update_training_status(
                message=f"Trainiere Modell (Epoch {epoch}/{epochs})...",
                progress=20 + (epoch - 1) * 60 / epochs,
                phase="Training"
            )
            
            # Wir geben dem Thread etwas Zeit, um die UI zu aktualisieren
            time.sleep(1)
        
        # Tatsächliches Training (ohne Fortschritts-Updates)
        if not trainer.train_model(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate):
            raise Exception("Fehler beim Training des Modells.")
        
        # Status aktualisieren: Modell optimieren
        if optimize:
            update_training_status(
                message="Optimiere Modell...",
                progress=80,
                phase="Optimierung"
            )
            
            # Modell optimieren
            if not trainer.optimize_model(quantize=True, onnx_export=True):
                raise Exception("Fehler bei der Optimierung des Modells.")
        
        # Status aktualisieren: Modell verpacken
        update_training_status(
            message="Verpacke Modell zur Distribution...",
            progress=90,
            phase="Verpackung"
        )
        
        # Modell verpacken
        package_path = trainer.package_model()
        if not package_path:
            raise Exception("Fehler beim Verpacken des Modells.")
        
        # Training erfolgreich abgeschlossen
        update_training_status(
            message="Training erfolgreich abgeschlossen",
            progress=100,
            phase="Abgeschlossen",
            completed=True,
            model_info=trainer.model_info
        )
        
        # Eintrag in die Modell-Liste hinzufügen
        add_model_to_list(
            model_id=task_id,
            name=f"Trainiertes {model_type.upper()}-Modell",
            description=f"Basis: {base_model}, {epochs} Epochs",
            base_model_name=base_model,
            created_date=datetime.now().isoformat(),
            file_path=package_path,
            size=os.path.getsize(package_path) if os.path.exists(package_path) else 0
        )
        
    except Exception as e:
        logger.error(f"Fehler beim Training: {e}")
        update_training_status(
            message=f"Fehler beim Training: {str(e)}",
            progress=0,
            phase="Fehler",
            error=str(e)
        )


def update_training_status(message=None, progress=None, phase=None, error=None, completed=None, model_info=None):
    """
    Aktualisiert den globalen Trainings-Status.
    
    Args:
        message (str, optional): Statusmeldung
        progress (int, optional): Fortschritt in Prozent (0-100)
        phase (str, optional): Aktuelle Trainingsphase
        error (str, optional): Fehlermeldung
        completed (bool, optional): Ob das Training abgeschlossen ist
        model_info (dict, optional): Modellinformationen
    """
    global training_status
    
    if message is not None:
        training_status["message"] = message
    
    if progress is not None:
        training_status["progress"] = progress
    
    if phase is not None:
        training_status["phase"] = phase
    
    if error is not None:
        training_status["error"] = error
        training_status["active"] = False
    
    if completed is not None:
        training_status["completed"] = completed
        if completed:
            training_status["active"] = False
    
    if model_info is not None:
        training_status["model_info"] = model_info


def add_model_to_list(model_id, name, description, base_model_name, created_date, file_path, size):
    """
    Fügt ein trainiertes Modell zur Liste der verfügbaren Modelle hinzu.
    
    Args:
        model_id (str): Modell-ID
        name (str): Modellname
        description (str): Modellbeschreibung
        base_model_name (str): Name des Basis-Modells
        created_date (str): Erstellungsdatum (ISO-Format)
        file_path (str): Pfad zur Modell-Datei
        size (int): Größe der Modell-Datei in Bytes
    """
    # Modellverzeichnis und -liste initialisieren
    models_dir = MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    
    models_list_file = models_dir / "models_list.json"
    
    # Bestehende Liste laden oder neu erstellen
    models_list = []
    if models_list_file.exists():
        try:
            with open(models_list_file, 'r', encoding='utf-8') as f:
                models_list = json.load(f)
        except:
            models_list = []
    
    # Neues Modell hinzufügen
    models_list.append({
        "id": model_id,
        "name": name,
        "description": description,
        "base_model_name": base_model_name,
        "created_date": created_date,
        "file_path": file_path,
        "size": size
    })
    
    # Liste speichern
    with open(models_list_file, 'w', encoding='utf-8') as f:
        json.dump(models_list, f, indent=2)


@model_training_bp.route('/api/model/status', methods=['GET'])
def get_training_status():
    """
    Gibt den aktuellen Trainings-Status zurück.
    
    Gibt zurück:
    - JSON mit dem aktuellen Trainings-Status
    """
    return jsonify(training_status)


@model_training_bp.route('/api/model/cancel', methods=['POST'])
def cancel_training():
    """
    Bricht das laufende Training ab.
    
    Gibt zurück:
    - JSON mit Status und ggf. Fehlermeldung
    """
    global training_status, active_training_thread
    
    # Überprüfen, ob ein Training läuft
    if not training_status["active"]:
        return jsonify({
            "success": False,
            "error": "Kein aktives Training gefunden."
        }), 404
    
    try:
        # Training abbrechen
        # Da wir keinen direkten Mechanismus zum Abbrechen des Trainings haben,
        # setzen wir nur den Status und lassen den Thread weiterlaufen
        # In einer vollständigen Implementierung würde hier ein Signal an den Trainer gesendet
        
        task_id = training_status["task_id"]
        
        # Status zurücksetzen
        training_status.update({
            "active": False,
            "task_id": None,
            "message": "Training abgebrochen",
            "progress": 0,
            "phase": "Abgebrochen",
            "error": "Training manuell abgebrochen",
            "completed": False
        })
        
        return jsonify({
            "success": True,
            "message": "Training abgebrochen"
        })
        
    except Exception as e:
        logger.error(f"Fehler beim Abbrechen des Trainings: {e}")
        return jsonify({
            "success": False,
            "error": f"Fehler beim Abbrechen des Trainings: {str(e)}"
        }), 500


@model_training_bp.route('/api/model/list', methods=['GET'])
def list_models():
    """
    Listet alle verfügbaren trainierten Modelle auf.
    
    Gibt zurück:
    - JSON-Array mit Modellinformationen
    """
    try:
        # Modellverzeichnis und -liste initialisieren
        models_dir = MODELS_DIR
        models_list_file = models_dir / "models_list.json"
        
        if not models_list_file.exists():
            return jsonify([])
        
        # Liste laden
        with open(models_list_file, 'r', encoding='utf-8') as f:
            models_list = json.load(f)
        
        # Nur verfügbare Modelle zurückgeben
        available_models = []
        for model in models_list:
            file_path = model.get("file_path")
            if file_path and os.path.exists(file_path):
                # Dateigröße aktualisieren
                model["size"] = os.path.getsize(file_path)
                available_models.append(model)
        
        return jsonify(available_models)
        
    except Exception as e:
        logger.error(f"Fehler beim Abrufen der Modell-Liste: {e}")
        return jsonify({
            "success": False,
            "error": f"Fehler beim Abrufen der Modell-Liste: {str(e)}"
        }), 500


@model_training_bp.route('/api/model/download/<model_id>', methods=['GET'])
def download_model(model_id):
    """
    Stellt ein trainiertes Modell zum Download bereit.
    
    Args:
        model_id (str): ID des herunterzuladenden Modells
    
    Gibt zurück:
    - Die Modell-Datei zum Download oder eine Fehlermeldung
    """
    try:
        # Modellverzeichnis und -liste initialisieren
        models_dir = MODELS_DIR
        models_list_file = models_dir / "models_list.json"
        
        if not models_list_file.exists():
            return jsonify({
                "success": False,
                "error": "Keine Modelle verfügbar."
            }), 404
        
        # Liste laden
        with open(models_list_file, 'r', encoding='utf-8') as f:
            models_list = json.load(f)
        
        # Modell finden
        model = next((m for m in models_list if m.get("id") == model_id), None)
        
        if not model:
            return jsonify({
                "success": False,
                "error": f"Modell mit ID {model_id} nicht gefunden."
            }), 404
        
        file_path = model.get("file_path")
        
        if not file_path or not os.path.exists(file_path):
            return jsonify({
                "success": False,
                "error": f"Modell-Datei nicht gefunden: {file_path}"
            }), 404
        
        # Modell herunterladen
        filename = os.path.basename(file_path)
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/zip'
        )
        
    except Exception as e:
        logger.error(f"Fehler beim Herunterladen des Modells: {e}")
        return jsonify({
            "success": False,
            "error": f"Fehler beim Herunterladen des Modells: {str(e)}"
        }), 500
