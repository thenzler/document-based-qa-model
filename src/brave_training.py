"""
Training mit BraveSearch-Integration
====================================

Dieses Skript automatisiert die Sammlung von Trainingsdaten mithilfe von BraveSearch
und trainiert dann das Modell.
"""

import os
import json
import logging
import requests
import argparse
from pathlib import Path
from datetime import datetime

# Importiere den lokalen Modell-Trainer
from local_model_trainer import LocalModelTrainer
from data_processing import DocumentProcessor

# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("brave-training")

class BraveSearchDataCollector:
    """Klasse zum Sammeln von Trainingsdaten mit der BraveSearch API"""
    
    def __init__(self, brave_api_key=None):
        self.api_key = brave_api_key or os.environ.get("BRAVE_API_KEY")
        self.api_url = "https://api.search.brave.com/res/v1/web/search"
        self.headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key
        }
    
    def search(self, query, count=10):
        """Führt eine Suche mit BraveSearch durch"""
        params = {"q": query, "count": min(count, 20)}
        
        try:
            response = requests.get(self.api_url, headers=self.headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            return data.get("web", {}).get("results", [])
        except Exception as e:
            logger.error(f"Fehler bei der BraveSearch-Anfrage: {e}")
            return []
    
    def fetch_content(self, url):
        """Holt den Inhalt einer Webseite"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Webseite {url}: {e}")
            return None
    
    def collect_training_data(self, queries, output_dir, count_per_query=5):
        """Sammelt Trainingsdaten für eine Liste von Suchanfragen"""
        output_dir.mkdir(parents=True, exist_ok=True)
        collected_files = []
        
        for i, query in enumerate(queries):
            logger.info(f"Sammle Daten für Anfrage {i+1}/{len(queries)}: '{query}'")
            
            # Führe Suche durch
            results = self.search(query, count=count_per_query)
            
            # Verarbeite jedes Ergebnis
            for j, result in enumerate(results):
                url = result.get("url")
                title = result.get("title", "").replace('/', '_')
                
                if not url:
                    continue
                
                # Hole Inhalt
                content = self.fetch_content(url)
                if not content:
                    continue
                
                # Erstelle Dateinamen
                safe_title = "".join(c if c.isalnum() or c in [' ', '-', '_'] else '_' for c in title[:50])
                filename = f"{i+1}_{j+1}_{safe_title}.txt"
                file_path = output_dir / filename
                
                # Speichere Inhalt
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(f"Title: {title}\n")
                        f.write(f"URL: {url}\n")
                        f.write(f"Query: {query}\n")
                        f.write(f"Date: {datetime.now().isoformat()}\n\n")
                        f.write(content)
                    
                    collected_files.append(file_path)
                    logger.info(f"Datei gespeichert: {file_path}")
                except Exception as e:
                    logger.error(f"Fehler beim Speichern der Datei {file_path}: {e}")
        
        return collected_files

def train_with_brave_data(args):
    """Hauptfunktion zum Sammeln von Daten und Training des Modells"""
    # Verzeichnisse vorbereiten
    base_dir = Path(args.output_dir)
    raw_data_dir = base_dir / "brave_raw_data"
    processed_docs_dir = base_dir / "processed_documents"
    model_output_dir = base_dir / "trained_model"
    
    for dir_path in [base_dir, raw_data_dir, processed_docs_dir, model_output_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # BraveSearch-Datensammler initialisieren
    collector = BraveSearchDataCollector(args.brave_api_key)
    
    # Standard-Abfragen für NLP/ML
    queries = [
        "question answering system implementation",
        "document retrieval techniques",
        "natural language processing best practices",
        "knowledge base construction methods",
        "semantic search implementation"
    ]
    
    # Daten sammeln
    logger.info(f"Sammle Daten für {len(queries)} Abfragen...")
    collected_files = collector.collect_training_data(
        queries=queries,
        output_dir=raw_data_dir,
        count_per_query=args.results_per_query
    )
    
    # Dokumente verarbeiten
    doc_processor = DocumentProcessor()
    
    for file_path in collected_files:
        try:
            logger.info(f"Verarbeite Dokument: {file_path}")
            doc_processor.process_document(
                str(file_path),
                output_dir=str(processed_docs_dir)
            )
        except Exception as e:
            logger.error(f"Fehler bei der Verarbeitung von {file_path}: {e}")
    
    # Trainer initialisieren
    trainer = LocalModelTrainer(
        base_model_name=args.base_model,
        model_type=args.model_type,
        docs_dir=processed_docs_dir,
        output_dir=model_output_dir
    )
    
    # Modell laden und trainieren
    logger.info("Lade Basis-Modell...")
    trainer.load_base_model()
    
    logger.info("Bereite Trainingsdaten vor...")
    trainer.prepare_training_data()
    
    logger.info(f"Starte Training mit {args.epochs} Epochen...")
    trainer.train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # Modell optimieren
    if args.optimize:
        logger.info("Optimiere Modell...")
        trainer.optimize_model(quantize=True, onnx_export=True)
    
    # Modell verpacken
    logger.info("Verpacke Modell...")
    package_path = trainer.package_model()
    
    if package_path:
        logger.info(f"Training abgeschlossen. Modell verfügbar unter: {package_path}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Training mit BraveSearch-Integration")
    
    # BraveSearch Parameter
    parser.add_argument("--brave-api-key", type=str, help="BraveSearch API-Schlüssel")
    parser.add_argument("--results-per-query", type=int, default=5, help="Anzahl der Ergebnisse pro Anfrage")
    
    # Ausgabeparameter
    parser.add_argument("--output-dir", type=str, default="data/brave_training", help="Ausgabeverzeichnis")
    
    # Trainingsparameter
    parser.add_argument("--base-model", type=str, default="distilbert-base-uncased", help="Name des Basis-Modells")
    parser.add_argument("--model-type", type=str, default="qa", choices=["qa", "causal_lm", "seq2seq_lm"], help="Modelltyp")
    parser.add_argument("--epochs", type=int, default=3, help="Anzahl der Trainings-Epochen")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch-Größe")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Lernrate")
    parser.add_argument("--optimize", action="store_true", help="Modell nach dem Training optimieren")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_with_brave_data(args)
