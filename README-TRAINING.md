# Modelltraining und -Download-Anleitung

Diese Anleitung erklärt, wie Sie das dokumentenbasierte QA-Modell mit Ihren eigenen Dokumenten trainieren und dann herunterladen können.

## 1. Installation und Start

### Voraussetzungen

- Python 3.8 oder höher
- Alle Pakete aus `requirements-enhaced.txt`

### Installation

```bash
# Abhängigkeiten installieren
pip install -r requirements-enhaced.txt
```

### Start der Anwendung

Verwenden Sie die integrierte Training-Version anstatt der Standard-App:

```bash
python integrated_training.py
```

Die Anwendung startet einen Webserver, den Sie über http://localhost:5000 erreichen können.

## 2. Dokumente hochladen

Bevor Sie ein Modell trainieren können, müssen Sie zuerst Dokumente hochladen:

1. Gehen Sie zur Startseite http://localhost:5000
2. Navigieren Sie zum Bereich "Documents"
3. Verwenden Sie die Upload-Funktion, um Ihre Dokumente hochzuladen
4. Unterstützte Formate: PDF, DOCX, TXT, MD, HTML

Die Dokumente werden automatisch verarbeitet und für das Training vorbereitet.

## 3. Modelltraining starten

Nachdem Sie Dokumente hochgeladen haben, können Sie das Training starten:

1. Navigieren Sie im Menü zum Punkt "Modell-Training"
2. Konfigurieren Sie das Training:
   - **Basis-Modell**: Wählen Sie ein Modell, das zu Ihren Daten passt
     - `distilbert-base-uncased` - Für englische Dokumente
     - `distilbert-base-multilingual-cased` - Für mehrsprachige Dokumente
     - `bert-base-german-cased` - Für deutsche Dokumente
   - **Anzahl der Epochs**: Mehr Epochen können bessere Ergebnisse liefern, aber das Training dauert länger (3-5 empfohlen)
   - **Batch-Größe**: Abhängig von Ihrem System (kleiner für weniger RAM)
   - **Lernrate**: Die Standardeinstellung von 0.00005 sollte für die meisten Fälle funktionieren
   - **Modell optimieren**: Erzeugt ein optimiertes Modell für schnellere Inferenz

3. Klicken Sie auf "Training starten"

## 4. Überwachung des Trainingsfortschritts

Nach dem Start des Trainings können Sie den Fortschritt auf der Training-Seite verfolgen:

- Der Fortschrittsbalken zeigt den aktuellen Status an
- Statusmeldungen informieren über die aktuelle Phase
- Bei Bedarf können Sie das Training mit "Training abbrechen" vorzeitig beenden

Das Training durchläuft folgende Phasen:
1. Laden des Basis-Modells (5%)
2. Vorbereitung der Trainingsdaten (15%)
3. Training des Modells (20-80%)
4. Optimierung (80-90%) (falls ausgewählt)
5. Verpackung für den Download (90-100%)

## 5. Herunterladen des trainierten Modells

Nach erfolgreichem Abschluss des Trainings erscheint ein "Modell herunterladen" Button. Klicken Sie darauf, um das trainierte Modell als ZIP-Datei herunterzuladen.

Die ZIP-Datei enthält:
- Das trainierte Modell
- Den Tokenizer
- Eine README-Datei mit Informationen zur Verwendung
- Optional: Optimierte Versionen des Modells

## 6. Verwendung des trainierten Modells

Sie können das heruntergeladene Modell mit der Hugging Face Transformers-Bibliothek verwenden:

```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# Pfad zum entpackten Modell
model_path = "pfad/zum/entpackten/modell"

# Tokenizer und Modell laden
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForQuestionAnswering.from_pretrained(model_path)

# Beispielverwendung
question = "Ihre Frage hier"
context = "Der Kontext, in dem die Frage beantwortet werden soll"
inputs = tokenizer(question, context, return_tensors="pt")
outputs = model(**inputs)

# Antwort extrahieren
answer_start = torch.argmax(outputs.start_logits)
answer_end = torch.argmax(outputs.end_logits) + 1
answer = tokenizer.decode(inputs.input_ids[0][answer_start:answer_end])
print(f"Antwort: {answer}")
```

## Fehlerbehebung

### Probleme beim Training

- **Training startet nicht**: Stellen Sie sicher, dass Sie Dokumente hochgeladen haben
- **Out of Memory-Fehler**: Reduzieren Sie die Batch-Größe
- **Langsames Training**: Versuchen Sie ein kleineres Basis-Modell oder weniger Epochen

### Probleme beim Download

- **Download-Button erscheint nicht**: Das Training wurde möglicherweise nicht erfolgreich abgeschlossen
- **Fehler beim Herunterladen**: Prüfen Sie, ob die Datei erfolgreich erstellt wurde (in models/local/packaged/)

Bei anhaltenden Problemen prüfen Sie die Konsolenausgabe der Anwendung für detailliertere Fehlermeldungen.