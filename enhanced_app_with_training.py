"""
Erweiterte Version der SCODi Dokumenten-QA-Anwendung
===================================================

Diese Version integriert:
- RAG mit OpenAI/Claude
- Einheitliche Benutzeroberfläche
- Lokales Modell-Training und -Download
"""

import os
import sys
import json
import time
import uuid
import logging
from datetime import datetime
from pathlib import Path
from werkzeug.utils import secure_filename

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Enhanced-RAG-QA")

# Flask importieren und konfigurieren
try:
    from flask import Flask, render_template, request, jsonify, send_from_directory, url_for, redirect, session
    from flask_cors import CORS
except ImportError:
    logger.error("Flask oder Flask-CORS nicht installiert. Bitte installieren Sie diese Pakete.")
    sys.exit(1)

# Pfad zum src-Verzeichnis hinzufügen
sys.path.append('./src')

try:
    # Importiere bestehende Module
    from src.data_processing import DocumentProcessor
    
    # Versuche externe LLM-APIs zu importieren
    try:
        import openai
        from openai import OpenAI
        OPENAI_AVAILABLE = True
    except ImportError:
        logger.warning("OpenAI API nicht verfügbar. Installieren Sie sie mit 'pip install openai'")
        OPENAI_AVAILABLE = False
        
    try:
        import anthropic
        ANTHROPIC_AVAILABLE = True
    except ImportError:
        logger.warning("Anthropic API nicht verfügbar. Installieren Sie sie mit 'pip install anthropic'")
        ANTHROPIC_AVAILABLE = False
        
    # Importiere verbesserte Embedding-Module
    try:
        import numpy as np
        import faiss
        from sentence_transformers import SentenceTransformer, CrossEncoder
        EMBEDDINGS_AVAILABLE = True
    except ImportError:
        logger.warning("Embedding-Abhängigkeiten nicht installiert. Semantische Suche wird eingeschränkt sein.")
        EMBEDDINGS_AVAILABLE = False
        
    # Importiere bestehende QA-Implementierungen
    try:
        from src.qa_system_rag import DocumentQA as RagDocumentQA
        RAG_AVAILABLE = True
    except ImportError:
        logger.warning("RAG-System nicht verfügbar. Verwende alternatives System.")
        RAG_AVAILABLE = False
        try:
            from src.qa_system_llm import DocumentQA as LlmDocumentQA
            LLM_AVAILABLE = True
        except ImportError:
            logger.warning("LLM-System nicht verfügbar. Verwende Basisimplementierung.")
            from src.qa_system import DocumentQA as BaseDocumentQA
            LLM_AVAILABLE = False
    
    # Importiere das Model-Training-API
    try:
        from src.model_training_api import model_training_bp
        MODEL_TRAINING_AVAILABLE = True
    except ImportError:
        logger.warning("Modell-Training-API nicht verfügbar. Diese Funktionalität wird deaktiviert.")
        MODEL_TRAINING_AVAILABLE = False
            
except ImportError as e:
    logger.error(f"Import-Fehler: {e}")
    logger.error("Stellen Sie sicher, dass das src-Verzeichnis existiert und die benötigten Module enthält.")
    sys.exit(1)

class EnhancedDocumentQA:
    """
    Erweiterte Implementierung des dokumentenbasierten QA-Systems mit
    Integration externer LLM-APIs wie OpenAI und Claude.
    """
    
    def __init__(
        self, 
        docs_dir="data/documents",
        embedding_model_name="sentence-transformers/all-mpnet-base-v2",
        use_openai=True,
        use_claude=False,
        openai_api_key=None,
        claude_api_key=None,
        use_gpu=False,
        enable_websearch=False
    ):
        """
        Initialisiert das erweiterte QA-System mit externen LLM-APIs
        
        Args:
            docs_dir (str): Verzeichnis für Dokumente
            embedding_model_name (str): Name des Embedding-Modells
            use_openai (bool): OpenAI API verwenden
            use_claude (bool): Claude API verwenden
            openai_api_key (str): OpenAI API-Schlüssel
            claude_api_key (str): Claude API-Schlüssel
            use_gpu (bool): GPU-Beschleunigung aktivieren (falls verfügbar)
            enable_websearch (bool): Websearch für unbekannte Fragen aktivieren
        """
        self.docs_dir = Path(docs_dir)
        self.documents = []
        self.chunks = []
        self.chunk_embeddings = None
        self.faiss_index = None
        self.use_gpu = use_gpu
        self.enable_websearch = enable_websearch
        
        # Erstelle Dokumentverzeichnis falls nicht vorhanden
        os.makedirs(self.docs_dir, exist_ok=True)
        os.makedirs(self.docs_dir / '.metadata', exist_ok=True)
        os.makedirs(self.docs_dir / '.chunks', exist_ok=True)
        
        # Initialisiere externe APIs
        self.openai_client = None
        self.claude_client = None
        self.active_llm = None
        
        # API-Schlüssel setzen
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.claude_api_key = claude_api_key or os.environ.get("ANTHROPIC_API_KEY")
        
        # Initialisiere externe LLMs
        if use_openai and OPENAI_AVAILABLE and self.openai_api_key:
            try:
                self.openai_client = OpenAI(api_key=self.openai_api_key)
                self.active_llm = "openai"
                logger.info("OpenAI API erfolgreich initialisiert")
            except Exception as e:
                logger.error(f"Fehler bei OpenAI API-Initialisierung: {e}")
        
        if use_claude and ANTHROPIC_AVAILABLE and self.claude_api_key:
            try:
                self.claude_client = anthropic.Anthropic(api_key=self.claude_api_key)
                if not self.active_llm:  # Nur als Fallback wenn OpenAI nicht verfügbar
                    self.active_llm = "claude"
                logger.info("Claude API erfolgreich initialisiert")
            except Exception as e:
                logger.error(f"Fehler bei Claude API-Initialisierung: {e}")
        
        # Initialisiere lokale Fallback-Systeme
        self.rag_system = None
        self.llm_system = None
        self.base_system = None
        
        # Initalisiere Embeddings
        if EMBEDDINGS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(embedding_model_name)
                if use_gpu and torch.cuda.is_available():
                    self.embedding_model = self.embedding_model.to(torch.device("cuda"))
                
                # CrossEncoder für Reranking
                self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                logger.info("Embedding-Modelle erfolgreich geladen")
            except Exception as e:
                logger.error(f"Fehler beim Laden der Embedding-Modelle: {e}")
                self.embedding_model = None
                self.cross_encoder = None
        else:
            self.embedding_model = None
            self.cross_encoder = None
        
        # Initialisiere Dokumentprozessor
        self.doc_processor = DocumentProcessor()
        
        # Lade Dokumente
        self._load_documents()
    
    def _load_documents(self):
        """Lädt alle verfügbaren Dokumente"""
        if self.docs_dir.exists():
            # Lade Metadaten
            metadata_dir = self.docs_dir / '.metadata'
            chunks_dir = self.docs_dir / '.chunks'
            
            if metadata_dir.exists() and chunks_dir.exists():
                try:
                    self._load_processed_data(metadata_dir, chunks_dir)
                    logger.info(f"Geladene Dokumente: {len(self.documents)}, Chunks: {len(self.chunks)}")
                    
                    # Erstelle Embedding-Index
                    if self.embedding_model is not None:
                        self._create_embeddings_index()
                except Exception as e:
                    logger.error(f"Fehler beim Laden der Dokumente: {e}")
    
    def _load_processed_data(self, metadata_dir, chunks_dir):
        """Lädt verarbeitete Dokumente und Chunks"""
        # Lade Metadaten
        for metadata_file in metadata_dir.glob('*.meta.json'):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    self.documents.append(metadata)
            except Exception as e:
                logger.error(f"Fehler beim Laden der Metadaten {metadata_file}: {e}")
        
        # Lade Chunks
        for chunks_file in chunks_dir.glob('*.chunks.json'):
            try:
                with open(chunks_file, 'r', encoding='utf-8') as f:
                    doc_chunks = json.load(f)
                    # Füge Dokumentinformationen hinzu
                    doc_name = chunks_file.stem.replace('.chunks', '')
                    doc_metadata = next((doc for doc in self.documents if doc.get('filename', '').startswith(doc_name)), None)
                    
                    for chunk in doc_chunks:
                        if doc_metadata:
                            chunk['document'] = {
                                'filename': doc_metadata.get('filename', ''),
                                'category': doc_metadata.get('category', 'allgemein')
                            }
                        self.chunks.append(chunk)
            except Exception as e:
                logger.error(f"Fehler beim Laden der Chunks {chunks_file}: {e}")
    
    def _create_embeddings_index(self):
        """Erstellt Embeddings für alle Chunks und einen FAISS-Index"""
        if self.embedding_model is None:
            logger.warning("Embedding-Modell nicht verfügbar, überspringe Indexierung")
            return
        
        logger.info("Erstelle Embeddings für alle Chunks...")
        
        # Extrahiere Texte aus Chunks
        texts = [chunk.get('text', '') for chunk in self.chunks]
        
        if not texts:
            logger.warning("Keine Texte zum Einbetten gefunden")
            return
        
        try:
            # Erstelle Embeddings
            self.chunk_embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            
            # Erstelle FAISS-Index
            embedding_dim = self.chunk_embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatL2(embedding_dim)
            self.faiss_index.add(self.chunk_embeddings)
            
            logger.info(f"Embeddings-Index erstellt mit {len(texts)} Chunks")
        except Exception as e:
            logger.error(f"Fehler bei der Erstellung des Embeddings-Index: {e}")
            self.chunk_embeddings = None
            self.faiss_index = None
    
    def add_document(self, file_path):
        """
        Fügt ein einzelnes Dokument hinzu und indiziert es
        
        Args:
            file_path (str): Pfad zur Dokumentdatei
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.warning(f"Datei nicht gefunden: {file_path}")
            return False
        
        try:
            # Dokument verarbeiten
            document_meta = self._process_document(file_path)
            
            if not document_meta:
                logger.warning(f"Dokument konnte nicht verarbeitet werden: {file_path}")
                return False
            
            # Zu Dokumentenliste hinzufügen
            self.documents.append(document_meta)
            
            # Aktualisiere Embeddings und Index
            if self.embedding_model is not None and self.chunk_embeddings is not None and self.faiss_index is not None:
                # Hole neue Chunks
                new_chunks = [c for c in self.chunks if c.get('document', {}).get('filename', '') == file_path.name]
                
                if new_chunks:
                    new_texts = [chunk.get('text', '') for chunk in new_chunks]
                    new_embeddings = self.embedding_model.encode(new_texts, show_progress_bar=True)
                    
                    # Aktualisiere den Index
                    self.faiss_index.add(new_embeddings)
                    
                    # Aktualisiere das Embeddings-Array
                    if self.chunk_embeddings is not None:
                        self.chunk_embeddings = np.vstack([self.chunk_embeddings, new_embeddings])
                    else:
                        self.chunk_embeddings = new_embeddings
            
            logger.info(f"Dokument erfolgreich hinzugefügt und indiziert: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Hinzufügen des Dokuments {file_path}: {e}")
            return False
    
    def _process_document(self, file_path):
        """
        Verarbeitet ein einzelnes Dokument
        
        Args:
            file_path (Path): Pfad zur Dokumentdatei
        """
        if not file_path.exists():
            logger.warning(f"Datei nicht gefunden: {file_path}")
            return None
        
        try:
            # Verarbeite Dokument mit vorhandenem Prozessor
            return self.doc_processor.process_document(str(file_path))
        except Exception as e:
            logger.error(f"Fehler bei der Verarbeitung von {file_path}: {e}")
            return None
    
    def _search_relevant_chunks(self, question, top_k=5):
        """
        Sucht nach den relevantesten Chunks für eine Frage
        
        Args:
            question (str): Die Frage
            top_k (int): Anzahl der zurückzugebenden Chunks
        """
        relevant_chunks = []
        
        # Wenn Embedding-Modell verfügbar ist
        if self.embedding_model is not None and self.chunk_embeddings is not None and self.faiss_index is not None:
            try:
                # Erzeuge Embedding für die Frage
                question_embedding = self.embedding_model.encode([question])
                
                # Suche nach den nächsten Nachbarn
                distances, indices = self.faiss_index.search(question_embedding, min(top_k*2, len(self.chunks)))
                
                # Konvertiere Indizes zu Chunks und füge Distanzen hinzu
                for i, idx in enumerate(indices[0]):
                    if idx < len(self.chunks):
                        chunk = self.chunks[idx].copy()
                        # Niedrigere Distanz = höhere Relevanz
                        relevance_score = 1.0 / (1.0 + distances[0][i])
                        chunk['relevance_score'] = min(relevance_score, 0.95)  # Cap bei 0.95
                        relevant_chunks.append(chunk)
                
                # Reranking mit CrossEncoder falls verfügbar
                if self.cross_encoder is not None and len(relevant_chunks) > 0:
                    relevant_chunks = self._rerank_chunks(question, relevant_chunks, top_k)
                
                return relevant_chunks[:top_k]
            except Exception as e:
                logger.error(f"Fehler bei semantischer Suche: {e}")
                # Fallback auf Schlüsselwortsuche
        
        # Fallback: Einfache Keyword-basierte Suche
        query_keywords = set(self._extract_keywords(question.lower()))
        chunk_scores = []
        
        for i, chunk in enumerate(self.chunks):
            chunk_text = chunk.get('text', '').lower()
            score = 0
            
            # TF-IDF ähnlicher Score
            chunk_keywords = set(self._extract_keywords(chunk_text))
            common_keywords = query_keywords.intersection(chunk_keywords)
            
            if common_keywords:
                # Gewichteter Score basierend auf Anzahl der gemeinsamen Keywords
                score = len(common_keywords) / len(query_keywords) if query_keywords else 0
            
            if score > 0:
                chunk_scores.append((i, score))
        
        # Sortiere nach Score absteigend
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Extrahiere die Top-K Chunks
        for i, (chunk_idx, score) in enumerate(chunk_scores[:top_k]):
            chunk = self.chunks[chunk_idx].copy()
            chunk['relevance_score'] = min(score, 0.95)
            relevant_chunks.append(chunk)
        
        return relevant_chunks
    
    def _extract_keywords(self, text):
        """Extrahiert Keywords aus Text"""
        import re
        # Entferne Sonderzeichen und teile in Wörter
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Entferne Stoppwörter
        stopwords = {'der', 'die', 'das', 'ein', 'eine', 'und', 'oder', 'aber', 'ist', 'sind', 'war', 
                    'waren', 'in', 'an', 'auf', 'für', 'mit', 'durch', 'über', 'unter', 'neben',
                    'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were'}
        
        keywords = [word for word in words if word not in stopwords and len(word) > 2]
        return keywords
    
    def _rerank_chunks(self, question, chunks, top_k=5):
        """
        Ordnet Chunks mithilfe des Cross-Encoders neu
        
        Args:
            question (str): Die Frage
            chunks (list): Liste der zu bewertenden Chunks
            top_k (int): Anzahl der zurückzugebenden Chunks
        """
        if not chunks or not self.cross_encoder:
            return chunks
        
        try:
            # Bereite Paare für das Reranking vor
            chunk_texts = [chunk.get('text', '')[:500] for chunk in chunks]  # Begrenze auf 500 Zeichen
            chunk_pairs = [[question, text] for text in chunk_texts]
            
            # Bewerte Chunks
            scores = self.cross_encoder.predict(chunk_pairs)
            
            # Erstelle Liste von (chunk, score) Paaren
            chunk_score_pairs = list(zip(chunks, scores))
            
            # Sortiere nach Score absteigend
            chunk_score_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Aktualisiere Relevanzscores und gib neu geordnete Chunks zurück
            reranked_chunks = []
            for chunk, score in chunk_score_pairs[:top_k]:
                chunk_copy = chunk.copy()
                chunk_copy['relevance_score'] = min(float(score), 0.95)  # Cap bei 0.95
                reranked_chunks.append(chunk_copy)
            
            return reranked_chunks
        except Exception as e:
            logger.error(f"Fehler beim Reranking: {e}")
            return chunks[:top_k]  # Fallback auf originale Reihenfolge
    
    def answer_question(self, question, top_k=5, use_websearch=False):
        """
        Beantwortet eine Frage mithilfe der RAG-Architektur mit externen LLMs
        
        Args:
            question (str): Die zu beantwortende Frage
            top_k (int): Anzahl der zu berücksichtigenden Dokumente
            use_websearch (bool): Websearch für unbekannte Fragen verwenden
            
        Returns:
            dict: Antwort mit Metadaten
        """
        start_time = time.time()
        
        # Überprüfen, ob Dokumente geladen wurden
        if not self.chunks:
            return {
                "answer": "Bitte laden Sie zuerst Dokumente hoch, um Fragen beantworten zu können.",
                "sources": [],
                "processing_time": time.time() - start_time,
                "used_websearch": False,
                "used_llm": self.active_llm
            }
        
        # Suche relevante Dokumente
        relevant_chunks = self._search_relevant_chunks(question, top_k)
        
        # Wenn keine relevanten Chunks gefunden wurden und Websearch aktiviert ist
        websearch_used = False
        web_results = []
        
        if (not relevant_chunks or max([c.get('relevance_score', 0) for c in relevant_chunks], default=0) < 0.5) and use_websearch and self.enable_websearch:
            # Hier würde Websearch-Logik implementiert werden
            pass
        
        # Kombiniere gefundene Texte zu einem Kontext
        context = self._prepare_context(relevant_chunks, web_results)
        
        # Generiere Antwort mit dem aktiven LLM
        answer, sources = self._generate_answer_with_llm(question, context, relevant_chunks, web_results)
        
        # Wenn kein LLM verfügbar ist, fallback auf einfachere Methode
        if not answer and not sources:
            answer, sources = self._generate_extractive_answer(question, relevant_chunks)
        
        # Berechne Verarbeitungszeit
        processing_time = time.time() - start_time
        
        return {
            "answer": answer,
            "sources": sources,
            "processing_time": processing_time,
            "used_websearch": websearch_used,
            "used_llm": self.active_llm
        }
    
    def _prepare_context(self, chunks, web_results=None):
        """
        Bereitet den Kontext für das LLM vor
        
        Args:
            chunks (list): Relevante Dokumentchunks
            web_results (list): Ergebnisse der Websuche
        """
        context = []
        
        # Füge Dokument-Chunks hinzu
        for i, chunk in enumerate(chunks):
            if i >= 5:  # Begrenze auf 5 Chunks
                break
                
            chunk_text = chunk.get('text', '')
            document_info = chunk.get('document', {})
            filename = document_info.get('filename', 'Unbekannt')
            
            if chunk_text:
                context.append(f"[Dokument: {filename}]\n{chunk_text}")
        
        # Füge Websuchergebnisse hinzu falls vorhanden
        if web_results:
            for i, result in enumerate(web_results):
                if i >= 3:  # Begrenze auf 3 Webresultate
                    break
                    
                title = result.get('title', 'Webseite')
                snippet = result.get('snippet', '')
                url = result.get('url', '')
                
                if snippet:
                    context.append(f"[Webseite: {title}]\n{snippet}\nURL: {url}")
        
        return "\n\n".join(context)
    
    def _generate_answer_with_llm(self, question, context, chunks, web_results=None):
        """
        Generiert eine Antwort mithilfe des aktiven LLMs
        
        Args:
            question (str): Die Frage
            context (str): Der vorbereitete Kontext
            chunks (list): Die relevanten Dokumentchunks
            web_results (list): Die Websuche-Ergebnisse
        """
        if not self.active_llm:
            return None, None
        
        # Erstelle die Quellen für die Antwort
        sources = []
        
        # Füge Dokumentquellen hinzu
        for chunk in chunks:
            document_info = chunk.get('document', {})
            filename = document_info.get('filename', 'Unbekannt')
            relevance_score = chunk.get('relevance_score', 0.0)
            
            # Extrahiere relevante Sätze
            chunk_text = chunk.get('text', '')
            import re
            sentences = re.split(r'(?<=[.!?])\s+', chunk_text)
            matching_sentences = sentences[:2] if sentences else [chunk_text[:150] + "..."]
            
            sources.append({
                "source": f"Dokument: {filename}",
                "filename": filename,
                "section": "Relevanter Abschnitt",
                "relevanceScore": relevance_score,
                "matchingSentences": matching_sentences
            })
        
        # Füge Webquellen hinzu
        if web_results:
            for result in web_results:
                title = result.get('title', 'Webseite')
                url = result.get('url', '')
                snippet = result.get('snippet', '')[:150] + "..."
                
                sources.append({
                    "source": f"Webseite: {title}",
                    "url": url,
                    "section": "Websuche-Ergebnis",
                    "relevanceScore": 0.8,  # Standard-Relevanz für Webergebnisse
                    "matchingSentences": [snippet]
                })
        
        # Bereite System-Prompt vor
        system_prompt = """
Du bist ein hilfreicher Assistent für Frage-Antwort-Aufgaben. 
Beantworte die Frage basierend auf dem gegebenen Kontext. 
Wenn der Kontext nicht ausreicht, sage ehrlich, dass du die Antwort nicht finden kannst.
Beziehe dich in deiner Antwort auf die bereitgestellten Dokumente.
Gib strukturierte, informative und sachliche Antworten.
"""

        # Bereite User-Prompt vor
        user_prompt = f"""
Frage: {question}

Kontext:
{context}

Bitte beantworte die Frage basierend auf dem Kontext. Beziehe dich auf die relevanten Informationen und gib eine klare, strukturierte Antwort.
"""

        try:
            # Benutze das aktive LLM
            if self.active_llm == "openai" and self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",  # Oder ein anderes aktuelles Modell
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1000
                )
                answer = response.choices[0].message.content
                
            elif self.active_llm == "claude" and self.claude_client:
                response = self.claude_client.messages.create(
                    model="claude-3-opus-20240229",  # Oder ein anderes aktuelles Modell
                    max_tokens=1000,
                    temperature=0.3,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ]
                )
                answer = response.content[0].text
                
            else:
                return None, None
                
            return answer, sources
            
        except Exception as e:
            logger.error(f"Fehler bei der LLM-Antwortgenerierung: {e}")
            return None, None
    
    def _generate_extractive_answer(self, question, relevant_chunks):
        """
        Generiert eine extraktive Antwort ohne LLM
        
        Args:
            question (str): Die Frage
            relevant_chunks (list): Die relevantesten Chunks
        """
        sources = []
        relevant_texts = []
        
        for chunk in relevant_chunks:
            chunk_text = chunk.get('text', '')
            document_info = chunk.get('document', {})
            filename = document_info.get('filename', 'Unbekannt')
            relevance_score = chunk.get('relevance_score', 0.0)
            
            # Extrahiere relevante Sätze
            import re
            sentences = re.split(r'(?<=[.!?])\s+', chunk_text)
            matching_sentences = []
            
            query_keywords = set(self._extract_keywords(question.lower()))
            for sentence in sentences:
                sentence_keywords = set(self._extract_keywords(sentence.lower()))
                # Wenn die Überlappung groß genug ist
                if len(query_keywords.intersection(sentence_keywords)) >= 1 and len(sentence) > 20:
                    matching_sentences.append(sentence)
            
            # Nehme den ganzen Chunk, wenn keine Sätze gefunden wurden
            if not matching_sentences and chunk_text:
                matching_sentences = [chunk_text[:200] + "..."]
            
            # Füge Quelleninformation hinzu
            if matching_sentences:
                sources.append({
                    "source": f"Dokument: {filename}",
                    "filename": filename,
                    "section": "Relevanter Abschnitt",
                    "relevanceScore": relevance_score,
                    "matchingSentences": matching_sentences[:2]  # Limitiere auf 2 Sätze
                })
                
                relevant_texts.extend(matching_sentences)
        
        # Generiere Antwort basierend auf relevanten Texten
        if relevant_texts:
            # Für allgemeine Fragen
            answer = f"Basierend auf den verfügbaren Dokumenten zu '{question}':\n\n"
            for i, text in enumerate(relevant_texts[:5]):  # Top 5 relevante Textstellen
                answer += f"{i+1}. {text}\n\n"
        else:
            # Generische Antwort, wenn keine relevanten Textstellen gefunden wurden
            answer = f"Zu Ihrer Frage '{question}' konnte leider keine spezifische Information in den Dokumenten gefunden werden."
        
        return answer, sources

# Initialisiere Flask-App
app = Flask(__name__)
CORS(app)  # Aktiviere CORS für alle Routen
app.secret_key = os.environ.get("FLASK_SECRET_KEY", str(uuid.uuid4()))
app.config['UPLOAD_FOLDER'] = Path('data/uploads')
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB maximale Dateigröße
app.config['DOCS_FOLDER'] = Path('data/documents')
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'docx', 'md', 'html', 'xlsx', 'pptx', 'xml', 'csv', 'json'}

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
    "show_process_menu": True,
    "company_name": "SCODi Software AG",
    "app_version": "2.0",
    "current_year": datetime.now().year,
    # Design-Typ (modern)
    "design_type": "modern"
}

# Globale Variablen
qa_system = None
recent_questions = []

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def init_system():
    global qa_system
    
    if qa_system is None:
        # Verwende API-Schlüssel aus Umgebungsvariablen
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        claude_api_key = os.environ.get("ANTHROPIC_API_KEY")
        
        # Bestimme, welche LLMs verfügbar sind
        use_openai = OPENAI_AVAILABLE and openai_api_key
        use_claude = ANTHROPIC_AVAILABLE and claude_api_key
        
        # Initialisiere das verbesserte QA-System
        try:
            qa_system = EnhancedDocumentQA(
                docs_dir=app.config['DOCS_FOLDER'],
                embedding_model_name="sentence-transformers/all-mpnet-base-v2",
                use_openai=use_openai,
                use_claude=use_claude,
                openai_api_key=openai_api_key,
                claude_api_key=claude_api_key,
                use_gpu=True,
                enable_websearch=True
            )
            logger.info("QA-System erfolgreich initialisiert")
        except Exception as e:
            logger.error(f"Fehler bei der Initialisierung des QA-Systems: {e}")
            # Fallback auf existierendes System falls verfügbar
            if RAG_AVAILABLE:
                qa_system = RagDocumentQA()
                logger.info("Fallback auf vorhandenes RAG-System")
            elif LLM_AVAILABLE:
                qa_system = LlmDocumentQA()
                logger.info("Fallback auf vorhandenes LLM-System")
            else:
                qa_system = BaseDocumentQA()
                logger.info("Fallback auf Basis-QA-System")

# Modell-Training-Blueprint registrieren, wenn verfügbar
if MODEL_TRAINING_AVAILABLE:
    app.register_blueprint(model_training_bp)

# Routen für die Einheitsseite
@app.route('/')
def index():
    return render_template('unified_app.html', 
                          design=SCODI_DESIGN,
                          page_title="SCODi 4P - Dokumentenbasiertes QA-System")

# Modell-Training-Seite
@app.route('/model-training')
def model_training():
    if not MODEL_TRAINING_AVAILABLE:
        return redirect(url_for('index'))
    
    return render_template('model_training.html',
                          design=SCODI_DESIGN,
                          page_title="SCODi 4P - Lokales Modell-Training")

# API-Endpunkte
@app.route('/api/ask', methods=['POST'])
def ask_question():
    global qa_system, recent_questions
    
    # Initialisiere QA-System falls nötig
    init_system()
    
    # Hole Frage aus der Anfrage
    data = request.json
    question = data.get('question', '')
    use_websearch = data.get('useWebsearch', False)
    
    if not question:
        return jsonify({"error": "Keine Frage angegeben"}), 400
    
    # Zur Liste der kürzlich gestellten Fragen hinzufügen
    if question not in recent_questions:
        recent_questions.insert(0, question)
        recent_questions = recent_questions[:10]  # Maximal 10 Fragen speichern
    
    # Beantworte Frage
    try:
        result = qa_system.answer_question(question, use_websearch=use_websearch)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Fehler beim Beantworten der Frage: {str(e)}")
        return jsonify({"error": f"Fehler beim Beantworten der Frage: {str(e)}"}), 500

@app.route('/api/documents', methods=['GET'])
def get_documents():
    init_system()
    try:
        return jsonify(qa_system.documents if qa_system else [])
    except Exception as e:
        return jsonify({"error": f"Fehler beim Abrufen der Dokumente: {str(e)}"}), 500

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
        
        # Stelle sicher, dass die Verzeichnisse existieren
        os.makedirs(app.config['DOCS_FOLDER'], exist_ok=True)
        
        # Speichere Datei
        file_path = app.config['DOCS_FOLDER'] / filename
        file.save(file_path)
        
        # Verarbeite Dokument
        try:
            success = qa_system.add_document(str(file_path))
            
            if success:
                return jsonify({
                    "success": True,
                    "message": f"Dokument '{filename}' erfolgreich hochgeladen und verarbeitet.",
                    "document": {
                        "filename": filename,
                        "path": str(file_path),
                        "upload_date": datetime.now().isoformat()
                    }
                })
            else:
                return jsonify({"error": f"Dokument konnte nicht verarbeitet werden: {filename}"}), 500
        except Exception as e:
            return jsonify({"error": f"Fehler bei der Verarbeitung des Dokuments: {str(e)}"}), 500
    
    return jsonify({"error": "Dateiformat nicht erlaubt"}), 400

@app.route('/api/system/status', methods=['GET'])
def system_status():
    """Gibt den aktuellen Status des Systems zurück"""
    init_system()
    
    api_status = {
        "openai": OPENAI_AVAILABLE and bool(os.environ.get("OPENAI_API_KEY")),
        "claude": ANTHROPIC_AVAILABLE and bool(os.environ.get("ANTHROPIC_API_KEY")),
    }
    
    return jsonify({
        "active_llm": qa_system.active_llm if qa_system else None,
        "api_status": api_status,
        "documents_loaded": len(qa_system.documents) if qa_system else 0,
        "embedding_model": "sentence-transformers/all-mpnet-base-v2" if qa_system and qa_system.embedding_model else None,
        "websearch_enabled": qa_system.enable_websearch if qa_system else False,
        "model_training_available": MODEL_TRAINING_AVAILABLE,
        "scodi_version": SCODI_DESIGN["app_version"],
        "design_type": SCODI_DESIGN["design_type"]
    })

@app.route('/api/system/set_api_keys', methods=['POST'])
def set_api_keys():
    """Setzt API-Schlüssel für externe Dienste"""
    data = request.json
    openai_api_key = data.get('openai_api_key')
    claude_api_key = data.get('claude_api_key')
    
    # Setze API-Schlüssel in Umgebungsvariablen
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    
    if claude_api_key:
        os.environ["ANTHROPIC_API_KEY"] = claude_api_key
    
    # Setze API-Schlüssel für das bestehende System
    global qa_system
    if qa_system:
        if openai_api_key:
            qa_system.openai_api_key = openai_api_key
            try:
                qa_system.openai_client = OpenAI(api_key=openai_api_key)
                qa_system.active_llm = "openai"
            except Exception as e:
                logger.error(f"Fehler bei OpenAI API-Initialisierung: {e}")
        
        if claude_api_key:
            qa_system.claude_api_key = claude_api_key
            try:
                qa_system.claude_client = anthropic.Anthropic(api_key=claude_api_key)
                if not qa_system.active_llm:
                    qa_system.active_llm = "claude"
            except Exception as e:
                logger.error(f"Fehler bei Claude API-Initialisierung: {e}")
    
    return jsonify({"success": True, "message": "API-Schlüssel erfolgreich aktualisiert"})

if __name__ == '__main__':
    # Stelle sicher, dass die Upload-Verzeichnisse existieren
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['DOCS_FOLDER'], exist_ok=True)
    
    # Stelle sicher, dass das Modell-Verzeichnis existiert
    os.makedirs('models/local', exist_ok=True)
    
    print(f"SCODi 4P QA-System gestartet mit {SCODI_DESIGN['design_type']} Design")
    print(f"Modell-Training verfügbar: {'Ja' if MODEL_TRAINING_AVAILABLE else 'Nein'}")
    app.run(debug=True, host='0.0.0.0')
