# Dokumentenbasiertes Frage-Antwort-System

Dieses Repository enthält ein Machine Learning Modell, das auf Dokumenten und Text trainiert werden kann, um Fragen zu beantworten und die zur Beantwortung verwendeten Quellen nachzuverfolgen.

## Überblick

Das System kombiniert moderne NLP-Techniken mit Information Retrieval, um:
1. Dokumente zu indexieren und zu speichern
2. Relevante Dokumente für eine Abfrage zu finden
3. Antworten aus diesen Dokumenten zu extrahieren
4. Die Quellen der Informationen nachzuverfolgen

Als Anwendungsbeispiel wird ein Churn-Prediction-Modell implementiert, wie es im bereitgestellten Dokument beschrieben ist.

## Projektstruktur

- `src/`: Enthält den Quellcode
  - `data_processing.py`: Funktionen zur Datenverarbeitung
  - `model_training.py`: Code für das Modelltraining
  - `qa_system.py`: Implementierung des Q&A-Systems
- `notebooks/`: Jupyter Notebooks für Demos
- `data/`: Verzeichnis für Trainingsdaten
- `models/`: Gespeicherte Modelle

## Installation und Nutzung

```bash
# Repository klonen
git clone https://github.com/thenzler/document-based-qa-model.git
cd document-based-qa-model

# Abhängigkeiten installieren
pip install -r requirements.txt

# Beispiel für die Nutzung
python src/qa_system.py --query "Was sind die wichtigsten Faktoren für Churn-Prediction?" --docs_path "data/documents/"
```

## Funktionsweise

Das System arbeitet in folgenden Schritten:

1. **Dokumentenverarbeitung**: Texte werden vorverarbeitet und indiziert
2. **Retrieval**: Für eine Anfrage werden die relevantesten Dokumente gefunden
3. **Antwortgenerierung**: Aus den gefundenen Dokumenten wird eine Antwort erstellt
4. **Quellennachweis**: Für jede Information wird die Quelle im Dokument nachverfolgt

## Anwendungsbeispiel: Churn-Prediction

Das Repository enthält ein vollständiges Beispiel für ein Churn-Prediction-Modell, das Kundendaten analysiert und Abwanderungsrisiken vorhersagt. Es umfasst:

- Datenverarbeitung und Feature-Engineering
- Trainieren verschiedener Modelle (Random Forest, Gradient Boosting, etc.)
- Hyperparameter-Optimierung
- Modellevaluation und -interpretation
- API für Vorhersagen

## Quellen

Diese Implementierung basiert auf der Analyse von Modellen für Churn-Prediction, wie im bereitgestellten Dokument beschrieben.