# Document-Based Question Answering System

Ein hochmodernes Machine Learning-Modell für dokumentenbasierte Frage-Antwort-Systeme mit Retrieval Augmented Generation (RAG) Architektur und konsistentem SCODi 4P Design.

Das System nutzt Natural Language Processing (NLP) und fortschrittliche Embedding-Techniken, um Fragen basierend auf bereitgestellten Dokumenten zu beantworten.

## Funktionen

Das System bietet folgende Hauptfunktionen:

1. **Dokumentenbasiertes Frage-Antwort-System**:
   - Dokumente in verschiedenen Formaten verarbeiten (PDF, DOCX, TXT, MD, HTML)
   - Dokumente in Abschnitte und Chunks mit semantischen Grenzen unterteilen
   - Semantische Suche nach relevanten Passagen mit Vektoreinbettungen
   - Generieren von Antworten auf Fragen basierend auf Dokumentinhalten mit Quellenangabe

2. **Fortschrittliche ML-Funktionen**:
   - Retrieval Augmented Generation (RAG) Architektur
   - Dichte Vektordarstellungen mit SentenceTransformer-Modellen
   - Cross-Encoder-Reranking für verbesserte Retrieval-Präzision
   - Multi-Query-Variationen für bessere Ergebnisse
   - Semantische Dokumentchunking mit Kontexterhaltung

3. **Web-Benutzeroberfläche**:
   - Intuitive Benutzeroberfläche mit SCODi 4P Design
   - Datei-Upload und -Verwaltung
   - Visualisierung der Ergebnisse mit Quellenangabe
   - Anpassbare Einstellungen

## Design-System

Das Projekt verwendet das SCODi 4P Design-System mit folgenden Komponenten:

- Farbschema basierend auf SCODi Corporate Identity
- Responsive Layout und Komponenten
- Konsistente Typografie und Ikonografie
- Moderne Navigation und Footer

### Farben

- **Primärfarbe**: #007f78 (Dunkelgrün/Türkis)
- **Sekundärfarbe**: #4b5864 (Dunkelgrau)
- **Akzentfarbe**: #f7f7f7 (Hellgrau für Hintergründe)

## Installation

### Voraussetzungen

- Python 3.8 oder höher
- pip (Python Package Manager)

### Installation der Abhängigkeiten

1. Repository klonen:
   ```bash
   git clone https://github.com/thenzler/document-based-qa-model.git
   cd document-based-qa-model
   ```

2. Abhängigkeiten installieren:
   ```bash
   pip install -r requirements.txt
   ```

### Erste Schritte

1. Anwendung starten:
   ```bash
   python modern_app.py
   ```

2. Browser öffnen und navigieren zu:
   ```
   http://localhost:5000
   ```

3. Den Anweisungen auf der Startseite folgen, um das System zu initialisieren und Dokumente hochzuladen.

## Verwendung

### Dokumentenbasiertes Frage-Antwort-System

1. Zum Bereich "Question & Answer" navigieren
2. Eine Frage basierend auf deinen Dokumenten eingeben
3. Das System durchsucht relevante Passagen und generiert eine Antwort mit Quellenangaben

### Dokumentenverwaltung

1. Zum Bereich "Documents" navigieren
2. Neue Dokumente hochladen (PDF, DOCX, TXT, MD, HTML)
3. Einstellungen für die Dokumentverarbeitung anpassen

## Architektur

Das System basiert auf einer modularen Architektur mit folgenden Komponenten:

1. **Dokumentenverarbeitung** (`DocumentProcessor`):
   - Textextraktion aus verschiedenen Dokumentformaten
   - Semantisches Chunking mit Kontexterhaltung
   - Speicherung und Verwaltung verarbeiteter Dokumente

2. **Frage-Antwort-System** (`DocumentQA` mit RAG-Architektur):
   - Semantische Suche nach relevanten Passagen mit Vektoreinbettungen
   - Cross-Encoder-Reranking für präzisere Ergebnisse
   - Antwortgenerierung basierend auf Dokumenten mit Quellenangabe

3. **Web-Oberfläche**:
   - Flask-basierte Weboberfläche mit SCODi 4P Design
   - Responsives Design
   - Interaktive Komponenten mit JavaScript

## Machine Learning Komponenten

### Vektoreinbettungen

Das System verwendet SentenceTransformer-Modelle, um dichte Vektordarstellungen von Dokumentchunks und Anfragen zu erstellen, was eine semantische Ähnlichkeitssuche ermöglicht.

```python
# Von qa_system_rag.py
self.embedding_model = SentenceTransformer(embedding_model_name)
```

### Vektordatenbank

Facebook AI Similarity Search (FAISS) wird für effiziente Vektorähnlichkeitssuche verwendet, was eine schnelle Auffindung relevanter Dokumentpassagen ermöglicht.

```python
# Von qa_system_rag.py
embedding_dim = self.chunk_embeddings.shape[1]
self.faiss_index = faiss.IndexFlatL2(embedding_dim)
self.faiss_index.add(self.chunk_embeddings)
```

### Cross-Encoder-Reranking

Ein zweistufiger Retrieval-Ansatz mit Bi-Encoder + Cross-Encoder-Reranking verbessert die Retrieval-Präzision.

```python
# Von qa_system_rag.py
self.cross_encoder = CrossEncoder(cross_encoder_model_name)
```

### Semantisches Dokumentchunking

Das System enthält intelligentes Dokumentchunking, das Kontextgrenzen und semantische Kohärenz bewahrt.

```python
# Von qa_system_rag.py
def _create_semantic_chunks(self, text, chunk_size=1000, overlap=200):
    # Semantisches Chunking-Implementation
```

## Konfiguration

Konfigurationsparameter können in der Datei `modern_app.py` angepasst werden.

## Mitmachen

Beiträge zum Projekt sind willkommen! Bitte folge diesen Schritten:

1. Repository forken
2. Feature-Branch erstellen (`git checkout -b feature/neue-funktion`)
3. Änderungen committen (`git commit -am 'Neue Funktion hinzufügen'`)
4. Zum Branch pushen (`git push origin feature/neue-funktion`)
5. Pull Request erstellen

## Lizenz

Dieses Projekt steht unter der MIT-Lizenz - siehe LICENSE-Datei für Details.
