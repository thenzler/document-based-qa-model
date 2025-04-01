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
    from enhanced_app_with_training import app, init_system, qa_system, SCODI_DESIGN, get_documents
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
                          page_title="SCODi 4P - Modelltraining",
                          title="Modelltraining",
                          status=training_status,
                          design=SCODI_DESIGN)

# Direkter Zugang zur Trainingsseite
@app.route('/modell-training')
def model_training_redirect():
    """Direkter Zugang zur Trainingsseite"""
    return redirect(url_for('training.training_page'))

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

# Füge einen Menüeintrag für das Training zu unified_app.html hinzu
def add_training_menu_to_unified_app():
    """Fügt einen Menüeintrag zum Training in die unified_app.html ein"""
    app_path = Path("templates/unified_app.html")
    
    if not app_path.exists():
        logger.warning(f"unified_app.html nicht gefunden: {app_path}")
        return
    
    try:
        with open(app_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Suche nach dem Navigation ul-Element
        nav_start = content.find('<ul class="navbar-nav ms-auto">')
        if nav_start != -1:
            # Suche nach dem Ende von </ul>
            nav_end = content.find('</ul>', nav_start)
            if nav_end != -1:
                # Füge den Menüeintrag für Modelltraining vor dem Ende von </ul> ein
                training_menu_item = '''
                    <li class="nav-item">
                        <a class="nav-link" href="/modell-training">
                            <i class="fas fa-brain me-1"></i> Modell-Training
                        </a>
                    </li>'''                
                
                # Prüfe ob der Eintrag bereits existiert
                if "Modell-Training" not in content[nav_start:nav_end]:
                    # Füge den Eintrag ein
                    new_content = content[:nav_end] + training_menu_item + content[nav_end:]
                    
                    # Erstelle ein Backup
                    backup_path = app_path.with_suffix(".html.bak")
                    with open(backup_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    
                    # Schreibe die neue Datei
                    with open(app_path, "w", encoding="utf-8") as f:
                        f.write(new_content)
                    
                    logger.info(f"Menüeintrag für Modelltraining zu {app_path} hinzugefügt")
        
    except Exception as e:
        logger.error(f"Fehler beim Hinzufügen des Menüeintrags: {e}")

# Erstelle Training-Template
def create_training_template():
    """Erstellt das Training-Template, falls es nicht existiert"""
    template_path = Path("templates/training.html")
    
    if not template_path.exists():
        logger.info(f"Erstelle Training-Template: {template_path}")
        
        template_content = '''{% extends "modern_layout.html" %}

{% block title %}Modell-Training{% endblock %}

{% block content %}
<div class="container">
    <div class="row">
        <div class="col-md-12">
            <h1>Modell-Training</h1>
            <div class="card">
                <div class="card-body">
                    <h3>Neues Training starten</h3>
                    <form id="trainingForm">
                        <div class="form-group">
                            <label for="baseModel">Basis-Modell:</label>
                            <select class="form-control" id="baseModel">
                                <option value="distilbert-base-uncased">distilbert-base-uncased (Englisch)</option>
                                <option value="distilbert-base-multilingual-cased">distilbert-base-multilingual-cased (Mehrsprachig)</option>
                                <option value="bert-base-german-cased">bert-base-german-cased (Deutsch)</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="epochs">Anzahl der Epochs:</label>
                            <input type="number" class="form-control" id="epochs" value="3" min="1" max="10">
                        </div>
                        <div class="form-group">
                            <label for="batchSize">Batch-Größe:</label>
                            <input type="number" class="form-control" id="batchSize" value="4" min="1" max="16">
                        </div>
                        <div class="form-group">
                            <label for="learningRate">Lernrate:</label>
                            <input type="number" class="form-control" id="learningRate" value="0.00005" step="0.00001" min="0.00001" max="0.001">
                        </div>
                        <div class="form-check">
                            <input type="checkbox" class="form-check-input" id="optimize" checked>
                            <label class="form-check-label" for="optimize">Modell nach dem Training optimieren</label>
                        </div>
                        <button type="button" id="startTraining" class="btn btn-primary mt-3">Training starten</button>
                    </form>
                </div>
            </div>
            
            <div class="card mt-4">
                <div class="card-body">
                    <h3>Training-Status</h3>
                    <div id="trainingStatus">
                        <p><strong>Status:</strong> <span id="statusMessage">{{ status.message }}</span></p>
                        <div class="progress">
                            <div id="progressBar" class="progress-bar" role="progressbar" style="width: {{ status.progress }}%;" aria-valuenow="{{ status.progress }}" aria-valuemin="0" aria-valuemax="100">{{ status.progress }}%</div>
                        </div>
                        
                        <div id="trainingControls" class="mt-3" {% if not status.active %}style="display: none;"{% endif %}>
                            <button id="cancelTraining" class="btn btn-danger">Training abbrechen</button>
                        </div>
                        
                        <div id="downloadSection" class="mt-3" {% if not status.completed %}style="display: none;"{% endif %}>
                            <a id="downloadModel" href="/api/training/download" class="btn btn-success">Modell herunterladen</a>
                        </div>
                        
                        <div id="errorSection" class="mt-3" {% if not status.error %}style="display: none;"{% endif %}>
                            <div class="alert alert-danger">
                                <strong>Fehler:</strong> <span id="errorMessage">{{ status.error }}</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
    $(document).ready(function() {
        // Training starten
        $("#startTraining").click(function() {
            const baseModel = $("#baseModel").val();
            const epochs = parseInt($("#epochs").val());
            const batchSize = parseInt($("#batchSize").val());
            const learningRate = parseFloat($("#learningRate").val());
            const optimize = $("#optimize").prop("checked");
            
            // Validierung
            if (isNaN(epochs) || epochs < 1 || isNaN(batchSize) || batchSize < 1 || isNaN(learningRate) || learningRate <= 0) {
                alert("Bitte geben Sie gültige Werte ein");
                return;
            }
            
            // Training starten
            $.ajax({
                url: "/api/training/start",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    base_model: baseModel,
                    epochs: epochs,
                    batch_size: batchSize,
                    learning_rate: learningRate,
                    optimize: optimize
                }),
                success: function(response) {
                    console.log("Training gestartet", response);
                    $("#trainingControls").show();
                    startStatusPolling();
                },
                error: function(xhr) {
                    alert("Fehler beim Starten des Trainings: " + (xhr.responseJSON ? xhr.responseJSON.message : xhr.statusText));
                }
            });
        });
        
        // Training abbrechen
        $("#cancelTraining").click(function() {
            if (confirm("Möchten Sie das Training wirklich abbrechen?")) {
                $.ajax({
                    url: "/api/training/cancel",
                    type: "POST",
                    success: function(response) {
                        console.log("Training abgebrochen", response);
                    },
                    error: function(xhr) {
                        alert("Fehler beim Abbrechen des Trainings: " + xhr.statusText);
                    }
                });
            }
        });
        
        // Status-Polling
        function startStatusPolling() {
            const pollInterval = setInterval(function() {
                $.ajax({
                    url: "/api/training/status",
                    type: "GET",
                    success: function(status) {
                        updateStatusUI(status);
                        
                        // Wenn Training abgeschlossen oder Fehler aufgetreten ist, Polling stoppen
                        if (!status.active) {
                            clearInterval(pollInterval);
                        }
                    },
                    error: function(xhr) {
                        console.error("Fehler beim Abrufen des Status:", xhr);
                    }
                });
            }, 2000); // Alle 2 Sekunden aktualisieren
        }
        
        // Status-UI aktualisieren
        function updateStatusUI(status) {
            $("#statusMessage").text(status.message);
            $("#progressBar").css("width", status.progress + "%").attr("aria-valuenow", status.progress).text(status.progress + "%");
            
            if (status.active) {
                $("#trainingControls").show();
            } else {
                $("#trainingControls").hide();
            }
            
            if (status.completed) {
                $("#downloadSection").show();
            } else {
                $("#downloadSection").hide();
            }
            
            if (status.error) {
                $("#errorSection").show();
                $("#errorMessage").text(status.error);
            } else {
                $("#errorSection").hide();
            }
        }
        
        // Initial Status prüfen
        {% if status.active %}
        startStatusPolling();
        {% endif %}
    });
</script>
{% endblock %}'''
        
        with open(template_path, "w", encoding="utf-8") as f:
            f.write(template_content)

# Hauptfunktion
def main():
    # Erstelle Trainings-Template und füge Menüeintrag hinzu
    create_training_template()
    add_training_menu_to_unified_app()
    
    # Blueprint registrieren
    app.register_blueprint(training_bp)
    
    # Debug-Ausgabe
    logger.info(f"Registrierte Blueprints: {list(app.blueprints.keys())}")
    
    # App starten
    print("=" * 80)
    print("Starte erweiterte QA-App mit integriertem Modelltraining")
    print("Sie können die Trainingsseite direkt unter http://localhost:5000/modell-training erreichen")
    print("=" * 80)
    app.run(debug=True, host='0.0.0.0')

if __name__ == "__main__":
    main()