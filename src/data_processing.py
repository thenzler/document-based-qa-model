import os
import json
from datetime import datetime
from pathlib import Path
import re

class DocumentProcessor:
    """
    Klasse zur Verarbeitung und Verwaltung von Dokumenten für das QA-System
    """
    
    def __init__(self, chunk_size=1000, overlap=200):
        """
        Initialisiert den DocumentProcessor
        
        Args:
            chunk_size (int): Größe der Text-Chunks in Zeichen
            overlap (int): Überlappung zwischen benachbarten Chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def list_documents(self, docs_dir):
        """
        Gibt eine Liste aller Dokumente im angegebenen Verzeichnis zurück
        
        Args:
            docs_dir (str): Pfad zum Dokumentenverzeichnis
            
        Returns:
            list: Liste mit Dokumentinformationen
        """
        result = []
        docs_path = Path(docs_dir)
        
        if not docs_path.exists():
            return result
        
        # Liste alle Dateien auf und filtere nach unterstützten Dateitypen
        for file_path in docs_path.glob('*.*'):
            if file_path.suffix.lower() in ['.pdf', '.docx', '.txt', '.md', '.html']:
                # Erstelle Metadaten für das Dokument
                doc_info = self._get_document_info(file_path)
                result.append(doc_info)
        
        return result
    
    def process_document(self, file_path, category='allgemein'):
        """
        Verarbeitet ein einzelnes Dokument
        
        Args:
            file_path (str): Pfad zur Dokumentdatei
            category (str): Kategorie des Dokuments
            
        Returns:
            dict: Metadaten und Verarbeitungsergebnis
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Datei nicht gefunden: {file_path}")
        
        # Extrahiere Text aus dem Dokument
        text = self._extract_text(file_path)
        
        # Unterteile Text in Chunks
        chunks = self._create_chunks(text)
        
        # Extrahiere Metadaten
        metadata = {
            "filename": file_path.name,
            "file_type": file_path.suffix.lower()[1:],
            "category": category,
            "size": file_path.stat().st_size,
            "upload_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "processed_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "num_chunks": len(chunks)
        }
        
        # Speichere Metadaten und Chunks
        self._save_document_data(file_path, metadata, chunks)
        
        return metadata
    
    def process_all_documents(self, docs_dir, force_reprocess=False):
        """
        Verarbeitet alle Dokumente im angegebenen Verzeichnis
        
        Args:
            docs_dir (str): Pfad zum Dokumentenverzeichnis
            force_reprocess (bool): Wenn True, werden auch bereits verarbeitete Dokumente erneut verarbeitet
            
        Returns:
            dict: Ergebnisse der Verarbeitung
        """
        docs_path = Path(docs_dir)
        
        if not docs_path.exists():
            raise FileNotFoundError(f"Verzeichnis nicht gefunden: {docs_dir}")
        
        results = {
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "documents": []
        }
        
        # Liste alle Dateien auf und verarbeite sie
        for file_path in docs_path.glob('*.*'):
            if file_path.suffix.lower() in ['.pdf', '.docx', '.txt', '.md', '.html']:
                # Prüfe, ob das Dokument bereits verarbeitet wurde
                metadata_path = self._get_metadata_path(file_path)
                
                if metadata_path.exists() and not force_reprocess:
                    results["skipped"] += 1
                    continue
                
                try:
                    # Verarbeite das Dokument
                    category = self._guess_category(file_path)
                    metadata = self.process_document(file_path, category)
                    results["processed"] += 1
                    results["documents"].append({
                        "filename": file_path.name,
                        "status": "success"
                    })
                except Exception as e:
                    results["failed"] += 1
                    results["documents"].append({
                        "filename": file_path.name,
                        "status": "error",
                        "error": str(e)
                    })
        
        return results
    
    def _extract_text(self, file_path):
        """
        Extrahiert Text aus einer Dokumentdatei
        
        Args:
            file_path (Path): Pfad zur Dokumentdatei
            
        Returns:
            str: Extrahierter Text
        """
        file_type = file_path.suffix.lower()
        
        # Für einfache Implementierung nur TXT-Dateien unterstützen
        if file_type == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif file_type == '.md':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif file_type == '.html':
            # Hier würde in einer vollständigen Implementierung HTML-Parsing stehen
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
                # Entferne HTML-Tags mit einem einfachen Regex
                return re.sub(r'<[^>]+>', '', html_content)
        elif file_type == '.pdf':
            # Hier müsste in einer vollständigen Implementierung PDF-Parsing stehen
            return f"[Platzhalter für PDF-Text aus {file_path.name}]"
        elif file_type == '.docx':
            # Hier müsste in einer vollständigen Implementierung DOCX-Parsing stehen
            return f"[Platzhalter für DOCX-Text aus {file_path.name}]"
        else:
            raise ValueError(f"Nicht unterstützter Dateityp: {file_type}")
    
    def _create_chunks(self, text):
        """
        Unterteilt Text in überlappende Chunks
        
        Args:
            text (str): Der zu unterteilende Text
            
        Returns:
            list: Liste mit Text-Chunks
        """
        chunks = []
        
        if not text:
            return chunks
        
        # Unterteile Text in Chunks mit Überlappung
        for i in range(0, len(text), self.chunk_size - self.overlap):
            chunk_text = text[i:i + self.chunk_size]
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "start_char": i,
                    "end_char": i + len(chunk_text)
                })
        
        return chunks
    
    def _get_document_info(self, file_path):
        """
        Erstellt Dokumentmetadaten für ein Dokument
        
        Args:
            file_path (Path): Pfad zur Dokumentdatei
            
        Returns:
            dict: Dokumentmetadaten
        """
        # Versuche, gespeicherte Metadaten zu lesen
        metadata_path = self._get_metadata_path(file_path)
        
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        
        # Erstelle Basis-Metadaten falls keine gespeicherten Daten vorhanden sind
        return {
            "id": str(hash(str(file_path))),
            "filename": file_path.name,
            "file_type": file_path.suffix.lower()[1:],
            "category": self._guess_category(file_path),
            "size": file_path.stat().st_size,
            "upload_date": datetime.fromtimestamp(file_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "processed": False
        }
    
    def _guess_category(self, file_path):
        """
        Versucht, die Kategorie eines Dokuments anhand des Namens zu erraten
        
        Args:
            file_path (Path): Pfad zur Dokumentdatei
            
        Returns:
            str: Vermutete Kategorie
        """
        filename = file_path.name.lower()
        
        if 'churn' in filename:
            return 'churn'
        elif 'product' in filename or 'produkt' in filename:
            return 'product'
        elif 'company' in filename or 'unternehmen' in filename:
            return 'company'
        else:
            return 'allgemein'
    
    def _get_metadata_path(self, file_path):
        """
        Gibt den Pfad für die Metadaten-Datei eines Dokuments zurück
        
        Args:
            file_path (Path): Pfad zur Dokumentdatei
            
        Returns:
            Path: Pfad zur Metadaten-Datei
        """
        # Erstelle Metadaten-Verzeichnis falls nötig
        metadata_dir = file_path.parent / '.metadata'
        metadata_dir.mkdir(exist_ok=True)
        
        return metadata_dir / f"{file_path.stem}.meta.json"
    
    def _save_document_data(self, file_path, metadata, chunks):
        """
        Speichert Metadaten und Chunks für ein Dokument
        
        Args:
            file_path (Path): Pfad zur Dokumentdatei
            metadata (dict): Dokumentmetadaten
            chunks (list): Liste mit Text-Chunks
        """
        # Speichere Metadaten
        metadata_path = self._get_metadata_path(file_path)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # Speichere Chunks
        chunks_dir = file_path.parent / '.chunks'
        chunks_dir.mkdir(exist_ok=True)
        
        chunks_path = chunks_dir / f"{file_path.stem}.chunks.json"
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
