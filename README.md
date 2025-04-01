# Document-Based Question Answering System

Ein modernes Machine Learning Modell für dokumentenbasierte Frage-Antwort-Systeme mit Retrieval Augmented Generation (RAG) Architektur und konsistentem SCODi 4P Design.

Das System nutzt Natural Language Processing (NLP) und fortschrittliche Embedding-Techniken, um auf Basis von Dokumenten präzise Antworten auf Fragen zu generieren.

## Funktionen

Das System bietet folgende Hauptfunktionen:

1. **Dokumentenbasiertes Frage-Antwort-System**:
   - Dokumente in verschiedenen Formaten einlesen und verarbeiten (PDF, DOCX, TXT, MD, HTML)
   - Dokumente semantisch in Abschnitte und Chunks unterteilen
   - Semantische Suche nach relevanten Passagen mittels Vektorembeddings
   - Generieren von Antworten auf Fragen basierend auf den Dokumentinhalten mit Quellenangaben
   - Cross-Encoder Reranking für präzisere Ergebnisse

2. **Fortschrittliche ML-Funktionen**:
   - Retrieval Augmented Generation (RAG) Architektur
   - Dichte Vektorrepräsentationen mit SentenceTransformer-Modellen
   - Multi-Query-Techniken für besseren Recall
   - Semantische Dokumentenaufteilung mit Kontexterhaltung
   - Verbesserte Antwortgenerierung

3. **Web-Benutzeroberfläche**:
   - Intuitive Benutzeroberfläche im SCODi 4P Design
   - Datei-Upload und -Management
   - Visualisierung der Ergebnisse mit Quellenangaben
   - Anpassbare Einstellungen

## Design-System

Das Projekt verwendet das SCODi 4P Design-System mit folgenden Komponenten:

- Farbschema basierend auf SCODi Corporate Identity
- Responsive Layout und Komponenten
- Konsistente Typografie und Ikonografie
- Modernes Navigations- und Footer-Design

### Farben

- **Primärfarbe**: #007f78 (Dunkelgrün/Türkis)
- **Sekundärfarbe**: #4b5864 (Dunkelgrau)
- **Akzentfarbe**: #f7f7f7 (Hellgrau für Hintergründe)

## Installation

### Voraussetzungen

- Python 3.8 oder höher
- pip (Python Package Manager)

### Installation der Abhängigkeiten

1. Klone das Repository:
   ```bash
   git clone https://github.com/thenzler/document-based-qa-model.git
   cd document-based-qa-model
   ```

2. Installiere die Abhängigkeiten:
   ```bash
   pip install -r requirements.txt
   ```

### Erste Schritte

1. Starte die Anwendung:
   ```bash
   python modern_app.py
   ```

2. Öffne einen Browser und navigiere zu:
   ```
   http://localhost:5000
   ```

3. Folge den Anweisungen auf der Startseite, um das System zu initialisieren und Dokumente hochzuladen.

## Nutzung

### Dokumentenbasiertes Frage-Antwort-System

1. Navigiere zum Bereich "Question & Answer"
2. Gib eine Frage ein, die basierend auf deinen Dokumenten beantwortet werden soll
3. Das System sucht nach relevanten Passagen und generiert eine Antwort mit Quellenangaben

### Dokumente verwalten

1. Navigiere zum Bereich "Documents"
2. Lade neue Dokumente hoch (PDF, DOCX, TXT, MD, HTML)
3. Passe Einstellungen für die Dokumentverarbeitung an
4. Verarbeite alle Dokumente mit einem Klick

## Architektur

Das System basiert auf einer modularen Architektur mit folgenden Komponenten:

1. **Dokumentenverarbeitung** (`DocumentProcessor`):
   - Extraktion von Text aus verschiedenen Dokumentformaten
   - Semantische Chunking mit Kontexterhaltung
   - Speicherung und Verwaltung der verarbeiteten Dokumente

2. **Embedding-Modelle** (`EmbeddingModel`, `CrossEncoderModel`):
   - Vektorrepräsentationen für semantische Ähnlichkeit
   - Reranking für präzisere Ergebnisse

3. **Frage-Antwort-System** mit RAG-Architektur (`DocumentQA`):
   - Semantische Suche nach relevanten Passagen mittels Vektorembeddings
   - Cross-Encoder Reranking für präzisere Ergebnisse
   - Generierung von Antworten basierend auf relevanten Dokumenten mit Quellenangaben

4. **Web-Oberfläche**:
   - Flask-basiertes Web-Interface im SCODi 4P Design
   - Responsive Design
   - Interaktive Komponenten mit JavaScript

## Machine Learning Komponenten

### Vektor-Embeddings

Das System verwendet SentenceTransformer-Modelle, um dichte Vektorrepräsentationen von Dokumentchunks und Abfragen zu erstellen, was semantische Ähnlichkeitssuche ermöglicht.

```python
# Aus model_embeddings.py
def get_embeddings(self, texts: List[str], batch_size=32, show_progress=True) -> np.ndarray:
    # Modell laden, falls noch nicht geladen
    self.load_model()
    
    if self.model is None:
        print("Warning: Embedding model not available, returning random embeddings")
        # Fallback auf zufällige Embeddings
        return np.random.rand(len(texts), 384)  # 384 ist eine gängige Embedding-Größe
    
    try:
        # Embeddings generieren
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size,
            show_progress_bar=show_progress
        )
        
        # Cache aktualisieren
        for i, text in enumerate(texts):
            # Hash des Textes als Schlüssel verwenden, um Speicher zu sparen
            text_hash = hash(text)
            self.embeddings_cache[text_hash] = embeddings[i]
        
        self.last_update = datetime.now()
        
        return embeddings
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        # Fallback auf zufällige Embeddings
        return np.random.rand(len(texts), 384)
```

### Vektor-Datenbank

Facebook AI Similarity Search (FAISS) wird für effiziente Vektorähnlichkeitssuche verwendet, was eine schnelle Abrufung relevanter Dokumentpassagen ermöglicht.

```python
# Aus qa_system_rag.py
def _create_embeddings_index(self):
    # Überprüfen, ob Embeddings verwendet werden können
    if not self._can_use_embeddings():
        logger.warning("Embedding capabilities not available, skipping indexing")
        return
        
    logger.info("Creating embeddings for all chunks...")
    
    # Extrahiere Texte aus Chunks
    texts = [chunk.get('text', '') for chunk in self.chunks]
    
    if not texts:
        logger.warning("No texts found to embed")
        return
        
    # Erstelle Embeddings für alle Texte
    try:
        if self.using_external_embedding_model:
            self.chunk_embeddings = self.embedding_model.get_embeddings(texts)
        else:
            self.chunk_embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Erstelle FAISS-Index für schnelle Nearest-Neighbor-Suche
        if self.using_faiss:
            embedding_dim = self.chunk_embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatL2(embedding_dim)
            self.faiss_index.add(self.chunk_embeddings)
```

### Cross-Encoder Reranking

Ein zweistufiger Retrieval-Ansatz mit Bi-Encoder + Cross-Encoder Reranking verbessert die Retrieval-Präzision.

```python
# Aus model_embeddings.py
def rerank(self, query: str, passages: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
    # Lade Modell falls nicht bereits geladen
    self.load_model()
    
    if self.model is None or not passages:
        print("Warning: Cross-encoder model not available or no passages, skipping reranking")
        return [(passage, 0.5) for passage in passages[:top_k]]
    
    try:
        # Erstelle Passage-Paare für Reranking
        passage_pairs = [[query, passage] for passage in passages]
        
        # Bewerte Passagen
        scores = self.model.predict(passage_pairs)
        
        # Erstelle Liste von (passage, score) Paaren
        passage_score_pairs = list(zip(passages, scores))
        
        # Sortiere nach Score absteigend und nehme top_k
        passage_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return passage_score_pairs[:top_k]
    except Exception as e:
        print(f"Error reranking passages: {e}")
        return [(passage, 0.5) for passage in passages[:top_k]]
```

### Semantisches Dokumenten-Chunking

Das System beinhaltet intelligentes Dokumenten-Chunking, das kontextuelle Grenzen und semantische Kohärenz bewahrt.

```python
# Aus qa_system_rag.py
def _create_semantic_chunks(self, text, chunk_size=1000, overlap=200):
    chunks = []
    
    if not text:
        return chunks
        
    # Text in Absätze aufteilen
    paragraphs = re.split(r'\n\s*\n', text)
    
    current_chunk = ""
    current_size = 0
    
    for paragraph in paragraphs:
        paragraph_size = len(paragraph)
        
        # Wenn das Hinzufügen dieses Absatzes die Chunk-Größe überschreiten würde
        if current_size + paragraph_size > chunk_size and current_chunk:
            # Speichere aktuellen Chunk
            chunks.append({
                "text": current_chunk.strip(),
                "start_char": len(chunks) * (chunk_size - overlap) if chunks else 0,
                "end_char": len(chunks) * (chunk_size - overlap) + len(current_chunk) if chunks else len(current_chunk)
            })
            
            # Starte neuen Chunk mit passender Überlappung
            if overlap > 0:
                # Versuche, eine gute Satzgrenze für die Überlappung zu finden
                sentences = re.split(r'(?<=[.!?])\s+', current_chunk)
                overlap_text = ""
                
                # Baue Überlappungstext vom Ende des vorherigen Chunks
                for sent in reversed(sentences):
                    if len(overlap_text) + len(sent) + 1 <= overlap:
                        overlap_text = sent + " " + overlap_text
                    else:
                        break
                
                current_chunk = overlap_text + paragraph + "\n\n"
                current_size = len(current_chunk)
            else:
                current_chunk = paragraph + "\n\n"
                current_size = paragraph_size + 2
        else:
            # Füge Absatz zum aktuellen Chunk hinzu
            current_chunk += paragraph + "\n\n"
            current_size += paragraph_size + 2
```

## Konfiguration

Konfigurationsparameter können in der Datei `modern_app.py` angepasst werden.

## Beitrag

Beiträge zum Projekt sind willkommen! Bitte folge diesen Schritten:

1. Forke das Repository
2. Erstelle einen Feature-Branch (`git checkout -b feature/neue-funktion`)
3. Committe deine Änderungen (`git commit -am 'Neue Funktion hinzufügen'`)
4. Pushe zum Branch (`git push origin feature/neue-funktion`)
5. Erstelle einen Pull Request

## Lizenz

Dieses Projekt ist unter der MIT-Lizenz veröffentlicht - siehe LICENSE-Datei für Details.
