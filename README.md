# Dokumentenbasiertes Frage-Antwort-System für Churn Prediction

Dieses Repository enthält ein Machine Learning Modell, das auf Dokumenten und Text trainiert werden kann, um Fragen zu beantworten und die zur Beantwortung verwendeten Quellen nachzuverfolgen. Als Anwendungsbeispiel wird ein Churn-Prediction-Modell implementiert.

## Überblick

Das System kombiniert moderne NLP-Techniken mit Information Retrieval, um:

1. Dokumente zu indexieren und zu speichern
2. Relevante Dokumente für eine Abfrage zu finden
3. Antworten aus diesen Dokumenten zu extrahieren oder zu generieren
4. Die Quelldokumente für die Antworten nachzuverfolgen und zu zitieren

Die Hauptfunktionen umfassen:

- **Dokumentenverarbeitung**: Unterstützung verschiedener Dateiformate (Text, PDF, DOCX, etc.)
- **Semantische Suche**: Auffinden relevanter Dokumente basierend auf Ähnlichkeit und Schlüsselwörtern
- **Antwortgenerierung**: Extraktion oder Generierung von Antworten aus relevanten Dokumenten
- **Quellennachweis**: Nachverfolgung der Dokumentquellen für jede Antwort
- **Churn-Prediction**: Vorhersage von Kundenabwanderung mit erklärbaren Ergebnissen

## Projektstruktur

```
.
├── data/
│   ├── churn_docs/         # Dokumente mit Informationen über Churn Prediction
│   ├── processed/          # Vorverarbeitete und indizierte Dokumente
│   └── customer_data.csv   # Beispieldaten für Churn-Prediction
│
├── models/                 # Gespeicherte Modelle
│
├── src/
│   ├── data_processing.py  # Dokumentenverarbeitung und Indexierung
│   ├── model_training.py   # Modelltraining für Churn Prediction und QA
│   ├── qa_system.py        # Implementierung des Frage-Antwort-Systems
│   └── demo.py             # Demonstrationsscript
│
└── README.md
```

## Installation

```bash
# Repository klonen
git clone https://github.com/thenzler/document-based-qa-model.git
cd document-based-qa-model

# Abhängigkeiten installieren
pip install -r requirements.txt
```

## Schnellstart

Führen Sie das Demo-Script aus, um das System in Aktion zu sehen:

```bash
python src/demo.py
```

Das Demo zeigt:
1. Wie Dokumente verarbeitet werden
2. Wie Fragen zu Churn Prediction beantwortet werden
3. Wie Quellen für die Antworten nachverfolgt werden
4. Wie Churn-Vorhersagen mit Erklärungen und Dokumentenreferenzen gemacht werden

## Verwendung

### Verarbeitung von Dokumenten

```bash
python src/demo.py --docs_dir path/to/documents --processed_dir path/to/save --force_reprocess
```

### Beantworten von Fragen mit Quellennachweis

```bash
# Interaktiver Modus
python src/demo.py --interactive

# Direkte Frage
python src/qa_system.py qa --query "Was ist Churn Prediction?" --docs_dir data/churn_docs --explain
```

### Churn Prediction mit Dokumentreferenzen

Im interaktiven Modus können Sie sowohl Fragen zu Churn Prediction stellen als auch die Abwanderungswahrscheinlichkeit von Kunden vorhersagen, wobei beide Funktionen relevante Dokumente als Quellen nutzen.

## Funktionsweise

Das System arbeitet in folgenden Schritten:

1. **Dokumentenverarbeitung**:
   - Laden verschiedener Dokumenttypen
   - Aufteilung in Abschnitte und Chunks
   - Extraktion von Schlüsselwörtern
   - Erstellung von Embeddings mit Sentence Transformers

2. **Retrieval**:
   - Umwandlung der Anfrage in Embeddings
   - Suche nach ähnlichen Dokumenten mit FAISS
   - Neuranking basierend auf Relevanz und Schlüsselwortüberschneidung

3. **Antwortgenerierung**:
   - Extraktion relevanter Information aus den Dokumenten
   - Generierung einer kohärenten Antwort
   - Zuordnung der Informationen zu den Quelldokumenten

4. **Quellennachweis**:
   - Nachverfolgung der verwendeten Dokumente
   - Bestimmung des Beitrags jedes Dokuments zur Antwort
   - Erstellung von formatierten Quellenangaben

## Churn-Prediction-Modell

Das implementierte Churn-Prediction-Modell:

- Analysiert Kundendaten mit verschiedenen Features (Nutzungsmetriken, Kundenzufriedenheit, etc.)
- Berechnet die Abwanderungswahrscheinlichkeit und Risikokategorie
- Identifiziert die wichtigsten Risikofaktoren für jeden Kunden
- Empfiehlt Maßnahmen basierend auf den Dokumenten und den Risikofaktoren
- Verfolgt die Dokumentquellen für Erklärungen und Empfehlungen nach

## Erweiterungsmöglichkeiten

- Integration weiterer Dokumentquellen und Formate
- Verbesserung der Antwortgenerierung durch Fine-Tuning auf spezifische Domains
- Implementierung von Feedback-Schleifen für kontinuierliches Lernen
- Anpassung an verschiedene Anwendungsfälle neben Churn Prediction

## Technische Details

- **Frameworks**: PyTorch, Transformers, FAISS, scikit-learn
- **Modelle**: Sentence Transformers für Embeddings, QA-Modelle für Antwortextraktion, T5 für Antwortgenerierung
- **Datenverarbeitung**: Spacy für NLP, langchain für Dokumentverarbeitung

## Lizenz

Open Source unter der MIT Lizenz.
