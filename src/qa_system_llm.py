import os
import json
import time
import re
import numpy as np
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm  # Für Fortschrittsbalken

class DocumentQA:
    """
    Klasse für dokumentenbasiertes Frage-Antwort-System mit lokalem LLM
    """
    
    def __init__(self, embedding_model_name="paraphrase-multilingual-MiniLM-L12-v2", 
                 qa_model_name="deepset/minilm-uncased-squad2", use_gpu=False):
        """
        Initialisiert das QA-System mit lokalen LLM-Modellen
        
        Args:
            embedding_model_name (str): Name des Embedding-Modells von HuggingFace/SentenceTransformers
            qa_model_name (str): Name des QA-Modells von HuggingFace
            use_gpu (bool): Ob GPU verwendet werden soll (falls verfügbar)
        """
        self.documents = []
        self.chunks = []
        self.chunk_embeddings = None
        self.faiss_index = None
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Modelle initialisieren
        try:
            print(f"Lade Embedding-Modell: {embedding_model_name}")
            self.embedding_model = SentenceTransformer(embedding_model_name)
            if self.use_gpu:
                self.embedding_model = self.embedding_model.to(torch.device("cuda"))
            
            print(f"Lade QA-Modell: {qa_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
            self.qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
            
            if self.use_gpu:
                self.qa_model = self.qa_model.to(torch.device("cuda"))
            
            self.qa_pipeline = pipeline(
                "question-answering",
                model=self.qa_model,
                tokenizer=self.tokenizer,
                device=0 if self.use_gpu else -1
            )
            
            print("Modelle erfolgreich geladen")
        except Exception as e:
            print(f"Fehler beim Laden der Modelle: {e}")
            print("Fallback auf einfachere Methoden wenn Modelle nicht geladen werden können")
            self.embedding_model = None
            self.qa_model = None
            self.tokenizer = None
            self.qa_pipeline = None
    
    def process_documents(self, docs_dir):
        """
        Verarbeitet alle Dokumente im angegebenen Verzeichnis
        
        Args:
            docs_dir (str): Pfad zum Dokumentenverzeichnis
        """
        docs_path = Path(docs_dir)
        
        if not docs_path.exists():
            print(f"Warnung: Verzeichnis nicht gefunden: {docs_dir}")
            return
        
        # Lade alle Dokumente und Chunks
        self.documents = []
        self.chunks = []
        
        # Lade Metadaten, wenn bereits verarbeitet
        metadata_dir = docs_path / '.metadata'
        chunks_dir = docs_path / '.chunks'
        
        if metadata_dir.exists() and chunks_dir.exists():
            # Versuche vorhandene verarbeitete Daten zu laden
            try:
                self._load_processed_data(metadata_dir, chunks_dir)
            except Exception as e:
                print(f"Fehler beim Laden verarbeiteter Daten: {e}")
                print("Verarbeite Dokumente neu...")
                self._process_raw_documents(docs_path)
        else:
            # Erstelle .metadata und .chunks Verzeichnisse wenn sie nicht existieren
            metadata_dir.mkdir(exist_ok=True)
            chunks_dir.mkdir(exist_ok=True)
            
            # Verarbeite Dokumente neu
            self._process_raw_documents(docs_path)
            
        # Erstelle Embeddings und Index für semantische Suche
        if self.embedding_model is not None:
            self._create_embeddings_index()
        
        print(f"Geladen: {len(self.documents)} Dokumente, {len(self.chunks)} Chunks")
    
    def _load_processed_data(self, metadata_dir, chunks_dir):
        """Lädt bereits verarbeitete Dokumente und Chunks"""
        # Lade Metadaten
        for metadata_file in metadata_dir.glob('*.meta.json'):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    self.documents.append(metadata)
            except Exception as e:
                print(f"Fehler beim Laden der Metadaten {metadata_file}: {e}")
        
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
                print(f"Fehler beim Laden der Chunks {chunks_file}: {e}")
    
    def _process_raw_documents(self, docs_path):
        """Verarbeitet Rohdokumente und erstellt Chunks"""
        # Nach Dokumenten suchen und verarbeiten
        for file_path in docs_path.glob('*.*'):
            if file_path.suffix.lower() in ['.txt', '.md', '.pdf', '.docx', '.html']:
                try:
                    print(f"Verarbeite Dokument: {file_path.name}")
                    document_meta = self._process_document(file_path)
                    
                    if document_meta:
                        self.documents.append(document_meta)
                    
                except Exception as e:
                    print(f"Fehler bei der Verarbeitung von {file_path.name}: {e}")
    
    def _process_document(self, file_path):
        """
        Verarbeitet ein einzelnes Dokument
        
        Args:
            file_path (Path): Pfad zur Dokumentdatei
            
        Returns:
            dict: Metadaten zum Dokument
        """
        if not file_path.exists():
            print(f"Datei nicht gefunden: {file_path}")
            return None
            
        # Lese Dokumentinhalt
        content = self._read_document_content(file_path)
        
        if not content:
            print(f"Kein Inhalt in {file_path.name} gefunden oder nicht unterstütztes Format")
            return None
            
        # Erstelle Dokumentmetadaten
        metadata = {
            "id": str(hash(str(file_path))),
            "filename": file_path.name,
            "file_type": file_path.suffix.lower()[1:],
            "category": self._guess_category(file_path),
            "size": file_path.stat().st_size,
            "upload_date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(file_path.stat().st_mtime)),
            "processed_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "chunk_size": 1000,
            "overlap": 200
        }
        
        # Teile den Inhalt in Chunks
        chunks = self._create_chunks(content, chunk_size=1000, overlap=200)
        
        # Speichere Metadaten
        metadata_path = file_path.parent / '.metadata' / f"{file_path.stem}.meta.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
            
        # Speichere Chunks
        chunks_with_doc_info = []
        for chunk in chunks:
            chunk_with_doc = dict(chunk)
            chunk_with_doc['document'] = {
                'filename': file_path.name,
                'category': metadata['category']
            }
            chunks_with_doc_info.append(chunk_with_doc)
            self.chunks.append(chunk_with_doc)
            
        chunks_path = file_path.parent / '.chunks' / f"{file_path.stem}.chunks.json"
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
            
        return metadata
        
    def _read_document_content(self, file_path):
        """Liest den Inhalt eines Dokuments basierend auf dem Dateityp"""
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == '.txt' or suffix == '.md':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
                    
            elif suffix == '.html':
                # Einfache HTML-Extraktion - für bessere Ergebnisse sollte ein HTML-Parser verwendet werden
                with open(file_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    # Entferne HTML-Tags
                    return re.sub(r'<[^>]+>', ' ', html_content)
                    
            elif suffix == '.pdf':
                print("PDF-Extraktion erfordert pdfplumber oder PyPDF2, wird in dieser Version nicht unterstützt")
                return f"[PDF-Inhalt von {file_path.name}]"
                
            elif suffix == '.docx':
                print("DOCX-Extraktion erfordert python-docx, wird in dieser Version nicht unterstützt")
                return f"[DOCX-Inhalt von {file_path.name}]"
            
            return None
            
        except Exception as e:
            print(f"Fehler beim Lesen von {file_path.name}: {e}")
            return None
    
    def _create_chunks(self, text, chunk_size=1000, overlap=200):
        """
        Unterteilt Text in überlappende Chunks
        
        Args:
            text (str): Der zu unterteilende Text
            chunk_size (int): Größe der Chunks in Zeichen
            overlap (int): Überlappung zwischen Chunks in Zeichen
            
        Returns:
            list: Liste von Chunks
        """
        chunks = []
        
        if not text:
            return chunks
            
        # Text in Sätze aufteilen
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = ""
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size <= chunk_size:
                current_chunk += sentence + " "
                current_size += sentence_size + 1
            else:
                # Chunk speichern
                if current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "start_char": len(chunks) * (chunk_size - overlap) if chunks else 0,
                        "end_char": len(chunks) * (chunk_size - overlap) + len(current_chunk) if chunks else len(current_chunk)
                    })
                
                # Neuen Chunk beginnen, evtl. mit Überlappung
                if overlap > 0 and current_chunk:
                    # Teile den Text zum Überlappen auf
                    words = current_chunk.split()
                    overlap_word_count = min(len(words), int(overlap / 5))  # Ca. 5 Zeichen pro Wort im Durchschnitt
                    overlap_text = " ".join(words[-overlap_word_count:]) + " "
                    current_chunk = overlap_text + sentence + " "
                    current_size = len(current_chunk)
                else:
                    current_chunk = sentence + " "
                    current_size = sentence_size + 1
        
        # Letzten Chunk speichern
        if current_chunk:
            chunks.append({
                "text": current_chunk.strip(),
                "start_char": len(chunks) * (chunk_size - overlap) if chunks else 0,
                "end_char": len(chunks) * (chunk_size - overlap) + len(current_chunk) if chunks else len(current_chunk)
            })
            
        return chunks
    
    def _guess_category(self, file_path):
        """Versucht, die Kategorie eines Dokuments anhand des Namens zu erraten"""
        filename = file_path.name.lower()
        
        if 'churn' in filename:
            return 'churn'
        elif 'product' in filename or 'produkt' in filename:
            return 'product'
        elif 'company' in filename or 'unternehmen' in filename:
            return 'company'
        elif 'kunden' in filename or 'customer' in filename:
            return 'customer'
        else:
            return 'allgemein'
    
    def _create_embeddings_index(self):
        """Erstellt Embeddings für alle Chunks und einen FAISS-Index für schnelle Suche"""
        # Prüfe, ob Modell geladen wurde
        if self.embedding_model is None:
            print("Embedding-Modell nicht verfügbar, überspringe Indexierung")
            return
            
        print("Erstelle Embeddings für alle Chunks...")
        
        # Extrahiere Texte aus Chunks
        texts = [chunk.get('text', '') for chunk in self.chunks]
        
        if not texts:
            print("Keine Texts gefunden zum Einbetten")
            return
            
        # Erstelle Embeddings für alle Texte
        try:
            self.chunk_embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            
            # Erstelle FAISS-Index für schnelle Nearest-Neighbor-Suche
            embedding_dim = self.chunk_embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatL2(embedding_dim)
            self.faiss_index.add(self.chunk_embeddings)
            
            print(f"Embeddings-Index erstellt mit {len(texts)} Chunks")
        except Exception as e:
            print(f"Fehler bei der Erstellung des Embeddings-Index: {e}")
            self.chunk_embeddings = None
            self.faiss_index = None

    def add_document(self, file_path):
        """
        Fügt ein einzelnes Dokument hinzu
        
        Args:
            file_path (str): Pfad zur Dokumentdatei
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            print(f"Datei nicht gefunden: {file_path}")
            return
        
        try:
            # Dokument verarbeiten
            document_meta = self._process_document(file_path)
            
            if not document_meta:
                print(f"Dokument konnte nicht verarbeitet werden: {file_path}")
                return
                
            # Füge zu Dokumentenliste hinzu
            self.documents.append(document_meta)
            
            # Aktualisiere Embeddings und Index
            if self.embedding_model is not None and self.chunk_embeddings is not None and self.faiss_index is not None:
                # Hole nur die neu hinzugefügten Chunks
                new_chunks = [c for c in self.chunks if c.get('document', {}).get('filename', '') == file_path.name]
                
                if new_chunks:
                    new_texts = [chunk.get('text', '') for chunk in new_chunks]
                    new_embeddings = self.embedding_model.encode(new_texts, show_progress_bar=True)
                    
                    # Füge neue Embeddings zum Index hinzu
                    self.faiss_index.add(new_embeddings)
                    
                    # Aktualisiere das Embeddings-Array
                    if self.chunk_embeddings is not None:
                        self.chunk_embeddings = np.vstack([self.chunk_embeddings, new_embeddings])
                    else:
                        self.chunk_embeddings = new_embeddings
            
            print(f"Dokument erfolgreich hinzugefügt und indiziert: {file_path}")
        except Exception as e:
            print(f"Fehler beim Hinzufügen des Dokuments {file_path}: {e}")
    
    def answer_question(self, question, use_generation=True, top_k=5):
        """
        Beantwortet eine Frage mithilfe des LLM und der indizierten Dokumente
        
        Args:
            question (str): Die zu beantwortende Frage
            use_generation (bool): Ob eine generierte Antwort erstellt werden soll
            top_k (int): Anzahl der zu berücksichtigenden Dokumente
            
        Returns:
            str: Die Antwort auf die Frage
            list: Die verwendeten Quellen
        """
        # Überprüfen, ob Dokumente geladen wurden
        if not self.chunks:
            return "Bitte laden Sie zuerst Dokumente hoch, um Fragen beantworten zu können.", []
        
        # Suche relevante Dokumente
        relevant_chunks = self._search_relevant_chunks(question, top_k)
        
        if not relevant_chunks:
            return (f"Leider konnte keine passende Information zu '{question}' gefunden werden. "
                   "Bitte versuchen Sie eine andere Frage oder laden Sie relevante Dokumente hoch."), []
        
        # Versuche, die Frage mithilfe des LLM zu beantworten
        if use_generation and self.qa_pipeline is not None:
            return self._generate_llm_answer(question, relevant_chunks)
        else:
            return self._generate_extractive_answer(question, relevant_chunks)
    
    def _search_relevant_chunks(self, question, top_k=5):
        """
        Sucht nach den relevantesten Chunks für eine Frage
        
        Args:
            question (str): Die Frage
            top_k (int): Anzahl der zurückzugebenden Chunks
            
        Returns:
            list: Die relevantesten Chunks
        """
        relevant_chunks = []
        
        # Wenn semantic search verfügbar ist (mit Embeddings)
        if self.embedding_model is not None and self.chunk_embeddings is not None and self.faiss_index is not None:
            try:
                # Erzeuge Embedding für die Frage
                question_embedding = self.embedding_model.encode([question])
                
                # Suche nach den nächsten Nachbarn
                distances, indices = self.faiss_index.search(question_embedding, min(top_k, len(self.chunks)))
                
                # Konvertiere Indizes zu Chunks und füge Distanzen hinzu
                for i, idx in enumerate(indices[0]):
                    if idx < len(self.chunks):
                        chunk = self.chunks[idx].copy()
                        # Niedrigere Distanz = höhere Relevanz
                        relevance_score = 1.0 / (1.0 + distances[0][i])
                        chunk['relevance_score'] = min(relevance_score, 0.95)  # Cap bei 0.95
                        relevant_chunks.append(chunk)
                
                return relevant_chunks
            except Exception as e:
                print(f"Fehler bei semantischer Suche: {e}")
                # Fallback auf Schlüsselwortsuche
        
        # Fallback: Einfache Keyword-basierte Suche
        query_keywords = re.findall(r'\b\w+\b', question.lower())
        chunk_scores = []
        
        for i, chunk in enumerate(self.chunks):
            chunk_text = chunk.get('text', '').lower()
            score = 0
            
            for keyword in query_keywords:
                if keyword in chunk_text:
                    score += 1
            
            if score > 0:
                chunk_scores.append((i, score))
        
        # Sortiere nach Score absteigend
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Extrahiere die Top-K Chunks
        for i, (chunk_idx, score) in enumerate(chunk_scores[:top_k]):
            chunk = self.chunks[chunk_idx].copy()
            # Normalisiere Score auf 0-1 Bereich
            chunk['relevance_score'] = min(score / len(query_keywords), 0.95)
            relevant_chunks.append(chunk)
        
        return relevant_chunks
    
    def _generate_llm_answer(self, question, relevant_chunks):
        """
        Generiert eine Antwort mithilfe des QA-Modells
        
        Args:
            question (str): Die Frage
            relevant_chunks (list): Die relevantesten Chunks
            
        Returns:
            str: Generierte Antwort
            list: Quellen
        """
        # Extrahiere die Quellinformationen für die Antwort
        sources = []
        best_answers = []
        
        for chunk in relevant_chunks:
            chunk_text = chunk.get('text', '')
            document_info = chunk.get('document', {})
            filename = document_info.get('filename', 'Unbekannt')
            relevance_score = chunk.get('relevance_score', 0.0)
            
            # Extrahiere Antwort aus dem Chunk mit dem QA-Modell
            try:
                qa_result = self.qa_pipeline(
                    question=question,
                    context=chunk_text,
                    max_answer_len=150
                )
                
                answer_text = qa_result.get('answer', '')
                confidence = qa_result.get('score', 0.0)
                
                if answer_text and confidence > 0.01:  # Minimaler Confidence-Schwellenwert
                    best_answers.append({
                        'text': answer_text,
                        'confidence': confidence,
                        'chunk_text': chunk_text,
                        'filename': filename
                    })
                
                # Extrahiere den Kontext rund um die Antwort
                start_pos = max(0, chunk_text.lower().find(answer_text.lower()) - 100)
                end_pos = min(len(chunk_text), chunk_text.lower().find(answer_text.lower()) + len(answer_text) + 100)
                context = chunk_text[start_pos:end_pos]
                
                # Füge Quellinformation hinzu
                sources.append({
                    "source": f"data/churn_docs/{filename}",
                    "filename": filename,
                    "section": "Relevanter Abschnitt",
                    "relevanceScore": relevance_score,
                    "matchingSentences": [context if context else chunk_text[:150] + "..."]
                })
                
            except Exception as e:
                print(f"Fehler bei der QA-Verarbeitung: {e}")
                # Bei Fehler: Füge trotzdem Quelle hinzu ohne Antwortextraktion
                sources.append({
                    "source": f"data/churn_docs/{filename}",
                    "filename": filename,
                    "section": "Relevanter Abschnitt",
                    "relevanceScore": relevance_score,
                    "matchingSentences": [chunk_text[:150] + "..."]
                })
        
        # Sortiere Antworten nach Confidence
        best_answers.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Wenn keine spezifischen Antworten gefunden wurden
        if not best_answers:
            # Fallback auf extraktive Antwort
            return self._generate_extractive_answer(question, relevant_chunks)
        
        # Erstelle final Answer basierend auf den besten Antworten
        final_answer = f"Basierend auf den Dokumenten ist die Antwort auf Ihre Frage '{question}':\n\n"
        
        # Nehme die Antwort mit der höchsten Confidence
        main_answer = best_answers[0]['text']
        final_answer += main_answer + "\n\n"
        
        # Füge ergänzende Informationen aus anderen Antworten hinzu
        if len(best_answers) > 1:
            final_answer += "Zusätzliche Informationen aus den Dokumenten:\n"
            for i, answer in enumerate(best_answers[1:3]):  # Maximal 2 zusätzliche Antworten
                if answer['text'] != main_answer:
                    final_answer += f"- {answer['text']}\n"
        
        return final_answer, sources
    
    def _generate_extractive_answer(self, question, relevant_chunks):
        """
        Generiert eine extraktive Antwort ohne LLM
        
        Args:
            question (str): Die Frage
            relevant_chunks (list): Die relevantesten Chunks
            
        Returns:
            str: Extrahierte Antwort aus den Chunks
            list: Quellen
        """
        sources = []
        relevant_texts = []
        
        for chunk in relevant_chunks:
            chunk_text = chunk.get('text', '')
            document_info = chunk.get('document', {})
            filename = document_info.get('filename', 'Unbekannt')
            relevance_score = chunk.get('relevance_score', 0.0)
            
            # Extrahiere relevante Sätze
            sentences = re.split(r'(?<=[.!?])\s+', chunk_text)
            matching_sentences = []
            
            query_keywords = set(re.findall(r'\b\w+\b', question.lower()))
            for sentence in sentences:
                sentence_keywords = set(re.findall(r'\b\w+\b', sentence.lower()))
                # Wenn die Überlappung groß genug ist
                if len(query_keywords.intersection(sentence_keywords)) >= 1 and len(sentence) > 20:
                    matching_sentences.append(sentence)
            
            # Nehme den ganzen Chunk, wenn keine Sätze gefunden wurden
            if not matching_sentences and chunk_text:
                matching_sentences = [chunk_text[:200] + "..."]
            
            # Füge Quelleninformation hinzu
            if matching_sentences:
                sources.append({
                    "source": f"data/churn_docs/{filename}",
                    "filename": filename,
                    "section": "Relevanter Abschnitt",
                    "relevanceScore": relevance_score,
                    "matchingSentences": matching_sentences[:2]  # Limitiere auf 2 Sätze
                })
                
                relevant_texts.extend(matching_sentences)
        
        # Wenn Churn-bezogene Wörter in der Frage vorkommen, versuche eine spezifischere Antwort zu geben
        if any(keyword in question.lower() for keyword in ['churn', 'kunden', 'abwanderung', 'fluktuation']):
            answer = """Churn-Prediction ist ein Verfahren zur Vorhersage von Kundenabwanderung. 

Es gibt mehrere Faktoren, die als Warnsignale für möglichen Churn dienen:
1. Abnehmende Nutzungsintensität der Produkte oder Dienstleistungen
2. Vermehrte Beschwerden oder Support-Anfragen
3. Fehlende Reaktion auf Marketing- oder Kommunikationsmaßnahmen
4. Ausbleibende Verlängerung von Abonnements

Um Churn zu reduzieren, können folgende Maßnahmen ergriffen werden:
- Proaktiver Kundenservice
- Personalisierte Angebote
- Regelmäßige Check-ins und Kundenbefragungen
- Verbesserung der Produkt- oder Servicequalität"""
            
            # Wenn passende Sätze gefunden wurden, füge sie zur Antwort hinzu
            if relevant_texts:
                answer += "\n\nAus Ihren Dokumenten:\n"
                for i, text in enumerate(relevant_texts[:3]):  # Top 3 relevante Textstellen
                    answer += f"\n- {text}"
                    
        elif relevant_texts:
            # Für andere Fragen
            answer = f"Basierend auf den verfügbaren Dokumenten zu '{question}':\n\n"
            for i, text in enumerate(relevant_texts[:5]):  # Top 5 relevante Textstellen
                answer += f"{i+1}. {text}\n\n"
        else:
            # Generische Antwort, wenn keine relevanten Textstellen gefunden wurden
            answer = f"Zu Ihrer Frage '{question}' konnte leider keine spezifische Information in den Dokumenten gefunden werden. Die Dokumente enthalten hauptsächlich Informationen zu Churn-Prediction, Kundenanalyse und Maßnahmen zur Kundenbindung."
        
        return answer, sources