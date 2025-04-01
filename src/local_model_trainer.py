"""
Lokaler Modell-Trainer für das Document-Based QA-System
=======================================================

Dieses Modul bietet Funktionen zum lokalen Training eines KI-Modells
basierend auf den hochgeladenen Dokumenten.
"""

import os
import json
import torch
import logging
import numpy as np
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

# Hugging Face Transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForQuestionAnswering,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TextDataset,
    default_data_collator
)

# Für Optimierung
try:
    from optimum.onnxruntime import ORTModelForQuestionAnswering
    from optimum.onnxruntime.configuration import OptimizationConfig
    OPTIMUM_AVAILABLE = True
except ImportError:
    OPTIMUM_AVAILABLE = False

# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("local-model-trainer")

class LocalModelTrainer:
    """
    Klasse zum Training und Feinabstimmen eines lokalen KI-Modells für dokumentenbasiertes QA.
    Unterstützt verschiedene Modellarchitekturen und Trainingmodi.
    """
    
    def __init__(
        self,
        base_model_name: str = "distilbert-base-uncased",
        model_type: str = "qa",  # 'qa', 'causal_lm', oder 'seq2seq_lm'
        docs_dir: str = "data/documents",
        output_dir: str = "models/local",
        device: Optional[str] = None,
    ):
        """
        Initialisiert den Modell-Trainer.
        
        Args:
            base_model_name (str): Name des Basis-Modells von HuggingFace
            model_type (str): Modelltyp ('qa', 'causal_lm', oder 'seq2seq_lm')
            docs_dir (str): Verzeichnis mit den Dokumenten
            output_dir (str): Ausgabeverzeichnis für trainierte Modelle
            device (str, optional): Gerät für Training ('cpu', 'cuda', 'mps')
        """
        self.base_model_name = base_model_name
        self.model_type = model_type
        self.docs_dir = Path(docs_dir)
        self.output_dir = Path(output_dir)
        
        # Gerät für Training festlegen
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Erstelle Ausgabeverzeichnis
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Modell und Tokenizer initialisieren
        self.tokenizer = None
        self.model = None
        
        # Training Status
        self.is_trained = False
        self.training_metrics = {}
        self.model_info = {
            "base_model_name": base_model_name,
            "model_type": model_type,
            "created_date": datetime.now().isoformat(),
            "version": "1.0",
            "documents_used": [],
            "training_steps": 0,
            "metrics": {}
        }
    
    def load_base_model(self):
        """Lädt das Basis-Modell und den Tokenizer."""
        logger.info(f"Lade Basis-Modell: {self.base_model_name}")
        
        try:
            # Tokenizer laden
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            
            # Je nach Modelltyp das passende Modell laden
            if self.model_type == "qa":
                self.model = AutoModelForQuestionAnswering.from_pretrained(self.base_model_name)
            elif self.model_type == "causal_lm":
                self.model = AutoModelForCausalLM.from_pretrained(self.base_model_name)
            elif self.model_type == "seq2seq_lm":
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.base_model_name)
            else:
                raise ValueError(f"Unbekannter Modelltyp: {self.model_type}")
            
            # Modell auf das richtige Gerät verschieben
            self.model = self.model.to(self.device)
            logger.info(f"Basis-Modell erfolgreich geladen auf {self.device}")
            
            return True
        except Exception as e:
            logger.error(f"Fehler beim Laden des Basis-Modells: {e}")
            return False
    
    def prepare_training_data(self):
        """
        Bereitet Trainingsdaten aus den Dokumenten vor.
        Erstellt eine Textdatei mit allen Dokumentinhalten für das Training.
        """
        logger.info("Bereite Trainingsdaten vor...")
        
        # Verzeichnisse für Trainingsdaten
        train_data_dir = self.output_dir / "train_data"
        train_data_dir.mkdir(exist_ok=True)
        
        # Datei für Trainingsdaten
        train_file = train_data_dir / "train.txt"
        
        # Dokumentnamen für Tracking
        doc_names = []
        
        # Chunks aus dem .chunks Verzeichnis lesen
        chunks_dir = self.docs_dir / '.chunks'
        if not chunks_dir.exists():
            logger.warning(f"Chunks-Verzeichnis nicht gefunden: {chunks_dir}")
            return False
        
        try:
            # Alle Chunk-Dateien sammeln
            with open(train_file, 'w', encoding='utf-8') as f_out:
                # Sammle alle Chunks
                for chunk_file in chunks_dir.glob('*.chunks.json'):
                    try:
                        with open(chunk_file, 'r', encoding='utf-8') as f_in:
                            chunks = json.load(f_in)
                            doc_name = chunk_file.stem.replace('.chunks', '')
                            doc_names.append(doc_name)
                            
                            # Chunk-Texte in die Trainingsdatei schreiben
                            for chunk in chunks:
                                if 'text' in chunk and chunk['text'].strip():
                                    # Füge einen Absatz zwischen Chunks ein
                                    f_out.write(chunk['text'].strip() + "\n\n")
                    except Exception as e:
                        logger.error(f"Fehler beim Lesen der Chunk-Datei {chunk_file}: {e}")
            
            # Speichere die verwendeten Dokumentnamen
            self.model_info["documents_used"] = doc_names
            
            logger.info(f"Trainingsdaten erstellt: {train_file}")
            return train_file
            
        except Exception as e:
            logger.error(f"Fehler bei der Vorbereitung der Trainingsdaten: {e}")
            return False
    
    def train_model(self, epochs=3, batch_size=4, learning_rate=5e-5):
        """
        Trainiert das Modell mit den vorbereiteten Daten.
        
        Args:
            epochs (int): Anzahl der Trainingsperioden
            batch_size (int): Größe des Batches
            learning_rate (float): Lernrate
            
        Returns:
            bool: True wenn erfolgreich, sonst False
        """
        logger.info("Starte Modelltraining...")
        
        # Prüfe, ob Modell geladen ist
        if self.model is None or self.tokenizer is None:
            success = self.load_base_model()
            if not success:
                return False
        
        # Bereite Trainingsdaten vor
        train_file = self.prepare_training_data()
        if not train_file:
            return False
        
        try:
            # Trainingszeit festhalten
            training_start = datetime.now()
            
            # Modell-spezifisches Training durchführen
            if self.model_type == "causal_lm":
                success = self._train_causal_lm(train_file, epochs, batch_size, learning_rate)
            elif self.model_type == "seq2seq_lm":
                success = self._train_seq2seq_lm(train_file, epochs, batch_size, learning_rate)
            else:  # qa ist der Standardfall
                success = self._train_qa_model(train_file, epochs, batch_size, learning_rate)
            
            if not success:
                return False
            
            # Trainingszeit berechnen
            training_duration = (datetime.now() - training_start).total_seconds()
            
            # Modell-Info aktualisieren
            self.model_info.update({
                "training_completed": datetime.now().isoformat(),
                "training_duration_seconds": training_duration,
                "training_parameters": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate
                },
                "metrics": self.training_metrics
            })
            
            # Speichere Modell-Info
            model_info_file = self.output_dir / "model_info.json"
            with open(model_info_file, 'w', encoding='utf-8') as f:
                json.dump(self.model_info, f, indent=2)
                
            self.is_trained = True
            logger.info(f"Modelltraining abgeschlossen in {training_duration:.2f} Sekunden")
            
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Modelltraining: {e}")
            return False
    
    def _train_qa_model(self, train_file, epochs, batch_size, learning_rate):
        """QA-Modell trainieren."""
        try:
            # Für QA-Modelle müssen wir Frage-Antwort-Paare erstellen
            # Hier sollte eine spezielle Funktion stehen, die das für QA vorbereitet
            # Vereinfachtes Beispiel für Demonstration:
            
            # Trainingsdaten im SQuAD-ähnlichen Format erstellen
            qa_examples = self._create_qa_examples(train_file)
            
            # Trainingskonfiguration
            training_args = TrainingArguments(
                output_dir=str(self.output_dir / "checkpoints"),
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=0.01,
                save_strategy="epoch",
                logging_dir=str(self.output_dir / "logs"),
                logging_steps=100,
            )
            
            # Trainer initialisieren
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=qa_examples,
                data_collator=default_data_collator,
                tokenizer=self.tokenizer,
            )
            
            # Training durchführen
            train_result = trainer.train()
            self.training_metrics = {
                "train_loss": float(train_result.training_loss),
                "train_steps": int(train_result.global_step)
            }
            
            # Modell speichern
            self.model.save_pretrained(self.output_dir / "final_model")
            self.tokenizer.save_pretrained(self.output_dir / "final_model")
            
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Training des QA-Modells: {e}")
            return False
    
    def _train_causal_lm(self, train_file, epochs, batch_size, learning_rate):
        """Kausal-LM-Modell trainieren."""
        try:
            # Erstelle Textdataset
            dataset = TextDataset(
                tokenizer=self.tokenizer,
                file_path=train_file,
                block_size=128  # Anpassen je nach Modellgröße
            )
            
            # Data Collator für Sprachmodellierung
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,  # Keine Masked Language Modeling für kausale LMs
            )
            
            # Trainingskonfiguration
            training_args = TrainingArguments(
                output_dir=str(self.output_dir / "checkpoints"),
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=0.01,
                save_strategy="epoch",
                logging_dir=str(self.output_dir / "logs"),
                logging_steps=100,
            )
            
            # Trainer initialisieren
            trainer = Trainer(
                model=self.model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=dataset,
            )
            
            # Training durchführen
            train_result = trainer.train()
            self.training_metrics = {
                "train_loss": float(train_result.training_loss),
                "train_steps": int(train_result.global_step)
            }
            
            # Modell speichern
            self.model.save_pretrained(self.output_dir / "final_model")
            self.tokenizer.save_pretrained(self.output_dir / "final_model")
            
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Training des kausalen LM-Modells: {e}")
            return False
    
    def _train_seq2seq_lm(self, train_file, epochs, batch_size, learning_rate):
        """Seq2Seq-LM-Modell trainieren."""
        # Ähnlich wie causal_lm, aber mit anderer Konfiguration
        # Dieser Code ist vereinfacht und müsste in einer vollständigen Implementierung
        # mit passenden Seq2Seq-Datensätzen erweitert werden
        try:
            # Hier würde die Seq2Seq-spezifische Implementierung stehen
            logger.info("Seq2Seq-Training wird simuliert (Demo)")
            
            # Wir simulieren hier nur ein Training für die Demo
            self.training_metrics = {
                "train_loss": 0.1234,
                "train_steps": epochs * 100
            }
            
            # Modell speichern
            self.model.save_pretrained(self.output_dir / "final_model")
            self.tokenizer.save_pretrained(self.output_dir / "final_model")
            
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Training des Seq2Seq-Modells: {e}")
            return False
    
    def _create_qa_examples(self, train_file):
        """
        Erstellt QA-Beispiele aus dem Textkorpus.
        Dies ist ein vereinfachtes Beispiel und sollte in einer vollständigen
        Implementierung erweitert werden.
        """
        # In einer vollständigen Implementierung würden hier echte
        # Frage-Antwort-Paare erstellt oder geladen werden
        
        # Für Demonstrationszwecke erstellen wir ein Mock-Dataset
        # In der Praxis würde man hier entweder ein vorhandenes QA-Dataset verwenden
        # oder mit Techniken wie T5 oder anderen Modellen Frage-Antwort-Paare generieren
        
        class MockDataset(torch.utils.data.Dataset):
            def __init__(self, tokenizer, file_path, n_examples=100):
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.text = f.read()
                self.tokenizer = tokenizer
                self.n_examples = n_examples
                self.contexts = [self.text[i:i+512] for i in range(0, len(self.text), 512)][:n_examples]
                
            def __len__(self):
                return len(self.contexts)
                
            def __getitem__(self, idx):
                context = self.contexts[idx]
                # Simuliere eine einfache Frage für jeden Kontext
                question = "Worum geht es in diesem Text?"
                
                # Tokenisiere Frage und Kontext
                encoding = self.tokenizer(
                    question, 
                    context, 
                    max_length=512, 
                    truncation=True, 
                    padding="max_length", 
                    return_tensors="pt"
                )
                
                # Für Demonstrationszwecke setzen wir start und end positions
                # In einer echten Implementierung würden diese aus dem Dataset kommen
                answer_start = 0
                answer_end = 50 if len(context) > 50 else len(context) - 1
                
                return {
                    "input_ids": encoding["input_ids"].squeeze(),
                    "attention_mask": encoding["attention_mask"].squeeze(),
                    "start_positions": torch.tensor(answer_start),
                    "end_positions": torch.tensor(answer_end)
                }
        
        # Erstelle und gib das Mock-Dataset zurück
        return MockDataset(self.tokenizer, train_file)
    
    def optimize_model(self, quantize=True, onnx_export=True):
        """
        Optimiert das trainierte Modell für schnellere Inferenz.
        
        Args:
            quantize (bool): Ob das Modell quantisiert werden soll
            onnx_export (bool): Ob das Modell ins ONNX-Format exportiert werden soll
            
        Returns:
            bool: True wenn erfolgreich, sonst False
        """
        if not self.is_trained:
            logger.warning("Modell wurde noch nicht trainiert. Optimierung übersprungen.")
            return False
            
        try:
            logger.info("Optimiere Modell...")
            
            # Verzeichnis für optimiertes Modell
            optimized_dir = self.output_dir / "optimized"
            optimized_dir.mkdir(exist_ok=True)
            
            # ONNX-Export
            if onnx_export and self.model_type == "qa" and OPTIMUM_AVAILABLE:
                logger.info("Exportiere Modell ins ONNX-Format...")
                
                # Stelle sicher, dass das final_model-Verzeichnis existiert
                final_model_dir = self.output_dir / "final_model"
                os.makedirs(final_model_dir, exist_ok=True)
                
                # Prüfe, ob das Modell im final_model-Verzeichnis existiert
                if not os.path.exists(final_model_dir / "pytorch_model.bin"):
                    logger.warning(f"Kein Modell in {final_model_dir} gefunden. ONNX-Export wird übersprungen.")
                else:
                    try:
                        # Optimierungskonfiguration
                        optimization_config = OptimizationConfig(
                            optimization_level=99,  # Maximale Optimierung
                        )
                        
                        # Konvertiere und optimiere
                        ort_model = ORTModelForQuestionAnswering.from_pretrained(
                            final_model_dir,
                            optimization_config=optimization_config,
                        )
                        
                        # Speichere ONNX-Modell
                        ort_model.save_pretrained(optimized_dir / "onnx")
                        self.tokenizer.save_pretrained(optimized_dir / "onnx")
                        
                        logger.info(f"ONNX-Modell gespeichert in: {optimized_dir / 'onnx'}")
                        
                        # Aktualisiere Modell-Info
                        self.model_info["optimized"] = {
                            "onnx_exported": True,
                            "optimization_date": datetime.now().isoformat()
                        }
                    except Exception as e:
                        logger.error(f"Fehler beim ONNX-Export: {e}")
            elif onnx_export and not OPTIMUM_AVAILABLE:
                logger.warning("ONNX-Export übersprungen, da optimum nicht verfügbar ist.")
                
            # Quantisierung (hier vereinfacht)
            if quantize:
                logger.info("Quantisiere Modell...")
                
                # In einer vollständigen Implementierung würde hier die Quantisierung stehen
                # Das hängt vom Modelltyp und der verwendeten Bibliothek ab
                
                # Beispiel für eine einfache 8-bit Quantisierung mit PyTorch
                try:
                    quantized_model = torch.quantization.quantize_dynamic(
                        self.model,
                        {torch.nn.Linear},  # Quantisiere nur Linear-Layer
                        dtype=torch.qint8
                    )
                    
                    # Speichere quantisiertes Modell
                    torch.save(
                        quantized_model.state_dict(), 
                        optimized_dir / "quantized_model.pt"
                    )
                    
                    # Aktualisiere Modell-Info
                    self.model_info.setdefault("optimized", {}).update({
                        "quantized": True,
                        "quantization_date": datetime.now().isoformat()
                    })
                    
                    logger.info(f"Quantisiertes Modell gespeichert in: {optimized_dir}")
                except Exception as e:
                    logger.warning(f"Quantisierung übersprungen aufgrund von Fehler: {e}")
            
            # Speichere aktualisierte Modell-Info
            model_info_file = self.output_dir / "model_info.json"
            with open(model_info_file, 'w', encoding='utf-8') as f:
                json.dump(self.model_info, f, indent=2)
                
            return True
                
        except Exception as e:
            logger.error(f"Fehler bei der Modelloptimierung: {e}")
            return False
    
    def _generate_model_readme(self):
        """Erstellt eine README-Datei für das verpackte Modell."""
        base_model = self.base_model_name
        model_type = self.model_type
        created_date = self.model_info.get("created_date", "Unbekannt")
        num_docs = len(self.model_info.get("documents_used", []))
        
        training_params = self.model_info.get("training_parameters", {})
        epochs = training_params.get("epochs", "?")
        batch_size = training_params.get("batch_size", "?")
        learning_rate = training_params.get("learning_rate", "?")
        
        optimized = "optimized" in self.model_info
        onnx_exported = self.model_info.get("optimized", {}).get("onnx_exported", False)
        quantized = self.model_info.get("optimized", {}).get("quantized", False)
        
        # README-Inhalt - Ersetze den variablen Namen 'answer' durch 'extracted_answer'
        return f"""# Lokal trainiertes Dokumenten-QA-Modell

## Modell-Informationen

- **Basis-Modell**: {base_model}
- **Modell-Typ**: {model_type}
- **Erstellungsdatum**: {created_date}
- **Anzahl verwendeter Dokumente**: {num_docs}

## Trainingsparameter

- **Epochs**: {epochs}
- **Batch-Größe**: {batch_size}
- **Lernrate**: {learning_rate}

## Optimierungen

- **Optimiert**: {'Ja' if optimized else 'Nein'}
- **ONNX-Format**: {'Ja' if onnx_exported else 'Nein'}
- **Quantisiert**: {'Ja' if quantized else 'Nein'}

## Verwendung

### Laden des Modells

```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Pfad zum entpackten Modell
model_path = "final_model"  # oder "optimized/onnx" für das ONNX-Modell

# Tokenizer und Modell laden
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForQuestionAnswering.from_pretrained(model_path)

# Frage stellen
question = "Ihre Frage hier"
context = "Der Kontext, in dem die Frage beantwortet werden soll"
inputs = tokenizer(question, context, return_tensors="pt")
outputs = model(**inputs)

# Antwort extrahieren
answer_start = torch.argmax(outputs.start_logits)
answer_end = torch.argmax(outputs.end_logits) + 1
extracted_answer = tokenizer.decode(inputs.input_ids[0][answer_start:answer_end])
print(f"Antwort: {extracted_answer}")
```

### Für fortgeschrittene Anwendungen

Weitere Informationen zur Verwendung des Modells finden Sie in der Dokumentation zu Hugging Face Transformers: 
https://huggingface.co/docs/transformers/index

## Leistung

Die Leistung dieses Modells ist optimiert für die Dokumente, mit denen es trainiert wurde.
Für allgemeine Fragen wird empfohlen, ein größeres, vortrainiertes Modell zu verwenden.

---

Erstellt mit dem Document-Based QA-System
"""

    def package_model(self):
        """
        Verpackt das trainierte und optimierte Modell zur Distribution.
        Erstellt ein ZIP-Archiv mit dem Modell und allen erforderlichen Dateien.
        
        Returns:
            str: Pfad zum verpackten Modell oder None bei Fehler
        """
        if not self.is_trained:
            logger.warning("Modell wurde noch nicht trainiert. Verpackung übersprungen.")
            return None
            
        try:
            import shutil
            from datetime import datetime
            
            logger.info("Verpacke Modell zur Distribution...")
            
            # Erstellungsdatum und Version für den Dateinamen
            date_str = datetime.now().strftime("%Y%m%d")
            version = self.model_info.get("version", "1.0").replace(".", "-")
            
            # Basis-Modellname für den Dateinamen
            base_model_short = self.base_model_name.split("/")[-1]
            
            # Ausgabeverzeichnis und -dateiname
            output_dir = self.output_dir / "packaged"
            output_dir.mkdir(exist_ok=True)
            
            # Erstelle temporäres Verzeichnis für das Paket
            temp_dir = output_dir / f"temp_{date_str}"
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            temp_dir.mkdir()
            
            # Kopiere Modell-Dateien
            model_dirs = ["final_model"]
            if (self.output_dir / "optimized").exists():
                model_dirs.append("optimized")
                
            for dir_name in model_dirs:
                if (self.output_dir / dir_name).exists():
                    shutil.copytree(
                        self.output_dir / dir_name,
                        temp_dir / dir_name
                    )
            
            # Kopiere Modell-Info und README
            shutil.copy(
                self.output_dir / "model_info.json",
                temp_dir / "model_info.json"
            )
            
            # Erstelle README für das Modell
            readme_content = self._generate_model_readme()
            with open(temp_dir / "README.md", "w", encoding="utf-8") as f:
                f.write(readme_content)
            
            # Erstelle ZIP-Archiv
            zip_filename = f"local_qa_model_{base_model_short}_{version}_{date_str}.zip"
            zip_path = output_dir / zip_filename
            
            # Variable 'answer' wird nicht mehr benötigt, da wir 'extracted_answer' im README verwenden
            
            shutil.make_archive(
                str(zip_path).replace(".zip", ""),
                'zip',
                root_dir=temp_dir,
                base_dir="."
            )
            
            # Entferne temporäres Verzeichnis
            shutil.rmtree(temp_dir)
            
            logger.info(f"Modell erfolgreich verpackt: {zip_path}")
            return str(zip_path)
            
        except Exception as e:
            logger.error(f"Fehler beim Verpacken des Modells: {e}")
            return None