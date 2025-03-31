import os
import json
import time
import re
from pathlib import Path
import random
from collections import defaultdict
import math

class DocumentQA:
    """
    Klasse für dokumentenbasiertes Frage-Antwort-System
    """
    
    def __init__(self):
        """
        Initialisiert das QA-System
        """
        self.documents = []
        self.chunks = []
        self.document_index = defaultdict(list)  # Keyword -> [chunk_indices]
        self.keywords_importance = {}  # Keyword -> IDF (inverse document frequency)
        
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
        
        # Lade Metadaten
        metadata_dir = docs_path / '.metadata'
        if metadata_dir.exists():
            for metadata_file in metadata_dir.glob('*.meta.json'):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        self.documents.append(metadata)
                except Exception as e:
                    print(f"Fehler beim Laden der Metadaten {metadata_file}: {e}")
        
        # Lade Chunks
        chunks_dir = docs_path / '.chunks'
        if chunks_dir.exists():
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
        
        # Erstelle Index aus Chunks
        self._create_index()
        
        print(f"Geladen: {len(self.documents)} Dokumente, {len(self.chunks)} Chunks")
        
    def _create_index(self):
        """
        Erstellt einen Keyword-basierten Index aller Chunks
        """
        # Zurücksetzen des Index
        self.document_index = defaultdict(list)
        keyword_doc_count = defaultdict(int)  # Anzahl der Dokumente pro Keyword
        
        # Extrahiere Keywords aus allen Chunks und erstelle einen invertierten Index
        for i, chunk in enumerate(self.chunks):
            if 'text' in chunk:
                # Normalisiere und extrahiere Keywords
                text = chunk['text'].lower()
                # Einfache Tokenisierung (in einer echten Anwendung würde man NLP-Bibliotheken verwenden)
                keywords = self._extract_keywords(text)
                
                # Füge zum Index hinzu
                for keyword in set(keywords):  # Einzigartige Keywords pro Chunk
                    self.document_index[keyword].append(i)
                    keyword_doc_count[keyword] += 1
        
        # Berechne IDF (Inverse Document Frequency) für alle Keywords
        total_docs = len(self.chunks)
        for keyword, doc_count in keyword_doc_count.items():
            self.keywords_importance[keyword] = math.log(total_docs / (1 + doc_count))
    
    def _extract_keywords(self, text):
        """
        Extrahiert Keywords aus einem Text
        
        Args:
            text (str): Text, aus dem Keywords extrahiert werden sollen
            
        Returns:
            list: Liste mit Keywords
        """
        # Entferne Sonderzeichen und teile in Wörter auf
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Entferne Stoppwörter (in einer echten Anwendung würde man eine umfangreichere Liste verwenden)
        stopwords = {'der', 'die', 'das', 'ein', 'eine', 'und', 'oder', 'ist', 'sind', 'in', 'mit', 'zu', 'für', 'von', 'auf'}
        keywords = [word for word in words if word not in stopwords and len(word) > 2]
        
        return keywords
    
    def add_document(self, file_path):
        """
        Fügt ein einzelnes Dokument hinzu
        
        Args:
            file_path (str): Pfad zur Dokumentdatei
        """
        # In einer vollständigen Implementierung würde hier die 
        # Extraktion und Indizierung des Dokuments erfolgen
        
        # Für Demo-Zwecke fügen wir ein Beispiel-Dokument hinzu
        file_path = Path(file_path)
        if file_path.exists():
            try:
                # Ein einfaches Dokument erstellen
                document = {
                    "filename": file_path.name,
                    "file_type": file_path.suffix.lower()[1:],
                    "category": "hinzugefügt",
                    "size": file_path.stat().st_size,
                    "upload_date": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Dokument zu Dokumentenliste hinzufügen
                self.documents.append(document)
                
                # Erstelle einen Beispiel-Chunk aus dem Dokument
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                chunk = {
                    "text": content[:1000],  # Nur die ersten 1000 Zeichen für den Demo-Chunk
                    "start_char": 0,
                    "end_char": min(1000, len(content)),
                    "document": {
                        "filename": file_path.name,
                        "category": "hinzugefügt"
                    }
                }
                
                # Chunk zur Chunks-Liste hinzufügen
                self.chunks.append(chunk)
                
                # Index aktualisieren
                self._create_index()
                
                print(f"Dokument hinzugefügt und indiziert: {file_path}")
            except Exception as e:
                print(f"Fehler beim Hinzufügen des Dokuments {file_path}: {e}")
        else:
            print(f"Datei nicht gefunden: {file_path}")
    
    def _search_chunks(self, query, top_k=5):
        """
        Sucht nach den relevantesten Chunks für eine Abfrage
        
        Args:
            query (str): Die Suchabfrage
            top_k (int): Anzahl der zurückzugebenden Chunks
            
        Returns:
            list: Liste der relevantesten Chunk-Indizes und ihrer Scores
        """
        # Extrahiere Keywords aus der Abfrage
        query_keywords = self._extract_keywords(query)
        
        if not query_keywords:
            return []
        
        # Berechne Relevanz-Scores für alle Chunks
        chunk_scores = defaultdict(float)
        
        for keyword in query_keywords:
            # Finde Chunks, die dieses Keyword enthalten
            if keyword in self.document_index:
                for chunk_idx in self.document_index[keyword]:
                    # TF-IDF-ähnlicher Score: Keywordgewicht * Wichtigkeit des Keywords
                    keyword_importance = self.keywords_importance.get(keyword, 1.0)
                    chunk_scores[chunk_idx] += keyword_importance
        
        # Sortiere Chunks nach Relevanz
        scored_chunks = [(idx, score) for idx, score in chunk_scores.items()]
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Gib die Top-K Chunks zurück
        return scored_chunks[:top_k]
    
    def _generate_answer(self, query, relevant_chunks):
        """
        Generiert eine Antwort auf Basis der relevanten Chunks
        
        Args:
            query (str): Die Abfrage
            relevant_chunks (list): Liste von relevanten Chunks
            
        Returns:
            str: Eine generierte Antwort
            list: Verwendete Quellen
        """
        if not relevant_chunks:
            return (
                f"Leider konnte keine passende Information zu '{query}' gefunden werden. "
                "Bitte versuchen Sie eine andere Frage oder laden Sie relevante Dokumente hoch.",
                []
            )
        
        # Extrahiere die relevanten Textstellen und ihre Dokumente
        sources = []
        relevant_texts = []
        
        for chunk_idx, score in relevant_chunks:
            if chunk_idx < len(self.chunks):
                chunk = self.chunks[chunk_idx]
                chunk_text = chunk.get('text', '')
                
                # Relevante Sätze suchen
                sentences = re.split(r'(?<=[.!?])\s+', chunk_text)
                matching_sentences = []
                
                query_keywords = set(self._extract_keywords(query))
                for sentence in sentences:
                    sentence_keywords = set(self._extract_keywords(sentence))
                    # Wenn die Überlappung groß genug ist
                    if len(query_keywords.intersection(sentence_keywords)) >= 1 and len(sentence) > 20:
                        matching_sentences.append(sentence)
                
                # Nehme den ganzen Chunk, wenn keine Sätze gefunden wurden
                if not matching_sentences and chunk_text:
                    matching_sentences = [chunk_text[:200] + "..."]
                
                # Füge Quelleninformation hinzu
                if 'document' in chunk and matching_sentences:
                    document_info = chunk['document']
                    filename = document_info.get('filename', 'Unbekannt')
                    
                    sources.append({
                        "source": f"data/churn_docs/{filename}",
                        "filename": filename,
                        "section": "Relevanter Abschnitt",
                        "relevanceScore": min(score / 5.0, 0.95),  # Normalisiere den Score
                        "matchingSentences": matching_sentences[:2]  # Limitiere auf 2 Sätze
                    })
                
                relevant_texts.extend(matching_sentences)
        
        # Die Antwort hängt von den gefundenen Textstellen ab
        
        # Wenn Churn-bezogene Wörter in der Frage vorkommen, versuche eine spezifischere Antwort zu geben
        if any(keyword in query.lower() for keyword in ['churn', 'kunden', 'abwanderung', 'fluktuation']):
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
            answer = f"Basierend auf den verfügbaren Dokumenten zu '{query}':\n\n"
            for i, text in enumerate(relevant_texts[:3]):  # Top 3 relevante Textstellen
                answer += f"{i+1}. {text}\n\n"
        else:
            # Generische Antwort, wenn keine relevanten Textstellen gefunden wurden
            answer = f"Zu Ihrer Frage '{query}' konnte leider keine spezifische Information in den Dokumenten gefunden werden. Die Dokumente enthalten hauptsächlich Informationen zu Churn-Prediction, Kundenanalyse und Maßnahmen zur Kundenbindung."
        
        return answer, sources
    
    def answer_question(self, question, use_generation=True, top_k=5):
        """
        Beantwortet eine Frage auf Basis der Dokumente
        
        Args:
            question (str): Die zu beantwortende Frage
            use_generation (bool): Ob eine KI-generierte Antwort erstellt werden soll
            top_k (int): Anzahl der zu berücksichtigenden Dokumente
            
        Returns:
            str: Die Antwort auf die Frage
            list: Die verwendeten Quellen
        """
        # Überprufen, ob Dokumente geladen wurden
        if not self.chunks:
            # Wenn keine Dokumente vorhanden sind, gib eine Hinweismeldung zurück
            return "Bitte laden Sie zuerst Dokumente hoch, um Fragen beantworten zu können.", []
        
        # Suche nach relevanten Chunks
        relevant_chunks = self._search_chunks(question, top_k)
        
        # Generiere eine Antwort auf Basis der relevanten Chunks
        if use_generation:
            return self._generate_answer(question, relevant_chunks)
        else:
            # Einfacher Modus, der nur die relevanten Textstellen zurückgibt
            sources = []
            relevant_texts = []
            
            for chunk_idx, score in relevant_chunks:
                if chunk_idx < len(self.chunks):
                    chunk = self.chunks[chunk_idx]
                    chunk_text = chunk.get('text', '')
                    
                    # Füge Quelleninformation hinzu
                    if 'document' in chunk:
                        document_info = chunk['document']
                        filename = document_info.get('filename', 'Unbekannt')
                        
                        sources.append({
                            "source": f"data/churn_docs/{filename}",
                            "filename": filename,
                            "section": "Relevanter Abschnitt",
                            "relevanceScore": min(score / 5.0, 0.95),  # Normalisiere den Score
                            "matchingSentences": [chunk_text[:100] + "..."]
                        })
                    
                    relevant_texts.append(chunk_text[:300] + "...")
            
            # Rückgabe der relevanten Textstellen ohne Generierung
            answer = "Relevante Informationen aus den Dokumenten:\n\n" + "\n\n".join(relevant_texts)
            return answer, sources
