import os
import json
import time
from pathlib import Path
import random

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
        
        print(f"Geladen: {len(self.documents)} Dokumente, {len(self.chunks)} Chunks")
    
    def add_document(self, file_path):
        """
        Fügt ein einzelnes Dokument hinzu
        
        Args:
            file_path (str): Pfad zur Dokumentdatei
        """
        # In einer vollständigen Implementierung würde hier die 
        # Extraktion und Indizierung des Dokuments erfolgen
        
        print(f"Dokument hinzugefügt: {file_path}")
    
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
        # Simuliere Verarbeitungszeit
        time.sleep(1.5)
        
        # In einer vollständigen Implementierung würde hier eine 
        # Ähnlichkeitssuche und Antwortgenerierung erfolgen
        
        # Mock-Antwort für Demo-Zwecke
        if 'churn' in question.lower() or 'kunden' in question.lower() or 'abwanderung' in question.lower():
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
        else:
            answer = f"""Basierend auf den verfügbaren Dokumenten kann Ihre Frage "{question}" wie folgt beantwortet werden:

Diese Antwort würde in einer vollständigen Implementierung basierend auf den relevanten Dokumenten generiert werden. In dieser Demo-Version werden Platzhalterantworten verwendet.

Die Dokumente enthalten Informationen zu Churn-Prediction, Kundenanalyse und Maßnahmen zur Kundenbindung."""
        
        # Simuliere relevante Quellen
        sources = []
        source_docs = ["churn_prediction_methods.pdf", "customer_retention_guide.txt", "ml_for_churn.docx"]
        
        for i in range(min(top_k, 3)):
            sources.append({
                "source": f"data/churn_docs/{source_docs[i]}",
                "filename": source_docs[i],
                "section": f"Abschnitt {i+1}",
                "relevanceScore": random.uniform(0.6, 0.95),
                "matchingSentences": [
                    f"Dies ist ein passender Satz aus dem Dokument {source_docs[i]}."
                ]
            })
        
        # In einer vollständigen Implementierung würde hier ein 
        # vollständiges Antwort-Objekt zurückgegeben werden
        return answer, sources
