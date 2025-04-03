"""
Model Training System for Document-Based QA
===========================================

This module provides functionality to train custom models for the document-based QA system.
Trained models are automatically used as the new standard for answering questions.
"""

import os
import json
import logging
import shutil
import time
from pathlib import Path
from datetime import datetime
import torch
from tqdm import tqdm
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ModelTraining")

# Try to import necessary libraries
try:
    from sentence_transformers import SentenceTransformer, losses, InputExample, evaluation
    from sentence_transformers.cross_encoder import CrossEncoder
    from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("SentenceTransformers not available. Training will be limited.")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from datasets import Dataset
    from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForQuestionAnswering
    from transformers import DefaultDataCollator
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("Transformers/datasets not available. Training will be limited.")
    TRANSFORMERS_AVAILABLE = False

class ModelTrainer:
    """
    Class for training document-based QA models that automatically become
    the new standard after training completes
    """
    
    def __init__(
        self,
        base_model_name="sentence-transformers/all-mpnet-base-v2",
        cross_encoder_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        qa_model_name="deepset/roberta-base-squad2",
        models_dir="models",
        use_gpu=True
    ):
        """
        Initialize the model trainer
        
        Args:
            base_model_name (str): Base embedding model name
            cross_encoder_model_name (str): Base cross-encoder model name
            qa_model_name (str): Base QA model name
            models_dir (str): Directory to store trained models
            use_gpu (bool): Whether to use GPU if available
        """
        self.base_model_name = base_model_name
        self.cross_encoder_model_name = cross_encoder_model_name
        self.qa_model_name = qa_model_name
        self.models_dir = Path(models_dir)
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Create models directory
        self.models_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories for different model types
        (self.models_dir / "embedding").mkdir(exist_ok=True)
        (self.models_dir / "cross-encoder").mkdir(exist_ok=True)
        (self.models_dir / "qa").mkdir(exist_ok=True)
        
        # Configuration file for currently active models
        self.config_file = self.models_dir / "active_models.json"
        
        # Initialize model availability flags
        self.st_available = SENTENCE_TRANSFORMERS_AVAILABLE
        self.transformers_available = TRANSFORMERS_AVAILABLE
        
        logger.info(f"Model trainer initialized with storage directory: {self.models_dir}")
        logger.info(f"SentenceTransformers available: {self.st_available}")
        logger.info(f"Transformers available: {self.transformers_available}")
    
    def train_embedding_model(self, training_data, validation_data=None, epochs=3, batch_size=16):
        """
        Train a custom embedding model
        
        Args:
            training_data (list): List of training examples (text pairs with scores)
            validation_data (list): List of validation examples
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            str: Path to the trained model
        """
        if not self.st_available:
            logger.error("SentenceTransformers not available. Cannot train embedding model.")
            return None
        
        try:
            # Create timestamp for model version
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_save_path = self.models_dir / "embedding" / f"embedding_model_{timestamp}"
            
            # Load base model
            logger.info(f"Loading base embedding model: {self.base_model_name}")
            model = SentenceTransformer(self.base_model_name)
            
            if self.use_gpu:
                model = model.to(torch.device("cuda"))
            
            # Prepare training data
            train_examples = []
            for item in training_data:
                if len(item) >= 3:  # Need at least two texts and a score
                    train_examples.append(InputExample(
                        texts=[item[0], item[1]], 
                        label=float(item[2])
                    ))
            
            # Prepare validation data if provided
            if validation_data:
                val_examples = []
                for item in validation_data:
                    if len(item) >= 3:
                        val_examples.append(InputExample(
                            texts=[item[0], item[1]], 
                            label=float(item[2])
                        ))
                
                evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
                    val_examples, batch_size=batch_size
                )
            else:
                evaluator = None
            
            # Configure training
            train_dataloader = torch.utils.data.DataLoader(
                train_examples, shuffle=True, batch_size=batch_size
            )
            train_loss = losses.CosineSimilarityLoss(model)
            
            # Train the model
            logger.info(f"Starting embedding model training for {epochs} epochs")
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=epochs,
                evaluation_steps=100,
                evaluator=evaluator,
                output_path=str(model_save_path)
            )
            
            logger.info(f"Embedding model successfully trained and saved to {model_save_path}")
            
            # Update active model configuration
            self._update_active_model_config("embedding_model", str(model_save_path))
            
            return str(model_save_path)
            
        except Exception as e:
            logger.error(f"Error training embedding model: {e}")
            return None
    
    def train_cross_encoder(self, training_data, validation_data=None, epochs=3, batch_size=16):
        """
        Train a custom cross-encoder model
        
        Args:
            training_data (list): List of training examples (text pairs with scores)
            validation_data (list): List of validation examples
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            str: Path to the trained model
        """
        if not self.st_available:
            logger.error("SentenceTransformers not available. Cannot train cross-encoder model.")
            return None
        
        try:
            # Create timestamp for model version
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_save_path = self.models_dir / "cross-encoder" / f"cross_encoder_{timestamp}"
            
            # Load base model
            logger.info(f"Loading base cross-encoder model: {self.cross_encoder_model_name}")
            model = CrossEncoder(self.cross_encoder_model_name, num_labels=1)
            
            # Prepare training data
            train_samples = []
            for item in training_data:
                if len(item) >= 3:  # Need at least two texts and a score
                    train_samples.append([item[0], item[1], float(item[2])])
            
            # Prepare validation data if provided
            if validation_data:
                val_samples = []
                for item in validation_data:
                    if len(item) >= 3:
                        val_samples.append([item[0], item[1], float(item[2])])
                
                evaluator = CECorrelationEvaluator(val_samples)
            else:
                evaluator = None
            
            # Train the model
            logger.info(f"Starting cross-encoder model training for {epochs} epochs")
            model.fit(
                train_dataloader=train_samples,
                epochs=epochs,
                batch_size=batch_size,
                evaluation_steps=100,
                evaluator=evaluator,
                output_path=str(model_save_path)
            )
            
            logger.info(f"Cross-encoder model successfully trained and saved to {model_save_path}")
            
            # Update active model configuration
            self._update_active_model_config("cross_encoder_model", str(model_save_path))
            
            return str(model_save_path)
            
        except Exception as e:
            logger.error(f"Error training cross-encoder model: {e}")
            return None
    
    def train_qa_model(self, training_data, validation_data=None, epochs=3, batch_size=8):
        """
        Train a custom QA model
        
        Args:
            training_data (list): List of QA training examples
            validation_data (list): List of QA validation examples
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            str: Path to the trained model
        """
        if not self.transformers_available:
            logger.error("Transformers not available. Cannot train QA model.")
            return None
        
        try:
            # Create timestamp for model version
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_save_path = self.models_dir / "qa" / f"qa_model_{timestamp}"
            
            # Load base model and tokenizer
            logger.info(f"Loading base QA model: {self.qa_model_name}")
            tokenizer = AutoTokenizer.from_pretrained(self.qa_model_name)
            model = AutoModelForQuestionAnswering.from_pretrained(self.qa_model_name)
            
            # Process training data into HF dataset format
            processed_train_data = []
            for item in training_data:
                if len(item) >= 3:  # Need question, context, and answer
                    question, context, answer_text = item[0], item[1], item[2]
                    
                    # Find answer position in context
                    answer_start = context.find(answer_text)
                    if answer_start == -1:
                        # Skip examples where answer is not in context
                        continue
                    
                    processed_train_data.append({
                        "question": question,
                        "context": context,
                        "answer": answer_text,
                        "start_positions": answer_start,
                        "end_positions": answer_start + len(answer_text)
                    })
            
            # Create datasets
            train_dataset = Dataset.from_list(processed_train_data)
            
            # Process validation data if provided
            if validation_data:
                processed_val_data = []
                for item in validation_data:
                    if len(item) >= 3:
                        question, context, answer_text = item[0], item[1], item[2]
                        answer_start = context.find(answer_text)
                        if answer_start == -1:
                            continue
                        
                        processed_val_data.append({
                            "question": question,
                            "context": context,
                            "answer": answer_text,
                            "start_positions": answer_start,
                            "end_positions": answer_start + len(answer_text)
                        })
                
                val_dataset = Dataset.from_list(processed_val_data)
            else:
                val_dataset = None
            
            # Tokenize datasets
            def preprocess_function(examples):
                questions = [q for q in examples["question"]]
                contexts = [c for c in examples["context"]]
                
                # Tokenize
                tokenized_examples = tokenizer(
                    questions,
                    contexts,
                    truncation="only_second",
                    max_length=384,
                    stride=128,
                    return_overflowing_tokens=False,
                    return_offsets_mapping=True,
                    padding="max_length",
                )
                
                # Map from token indices to character positions
                offset_mapping = tokenized_examples.pop("offset_mapping")
                
                # Find start and end token positions
                tokenized_examples["start_positions"] = []
                tokenized_examples["end_positions"] = []
                
                for i, offsets in enumerate(offset_mapping):
                    start_char = examples["start_positions"][i]
                    end_char = examples["end_positions"][i]
                    
                    # Find token that contains the answer start
                    start_token = None
                    for j, (start, end) in enumerate(offsets):
                        if start <= start_char < end:
                            start_token = j
                            break
                    
                    # Find token that contains the answer end
                    end_token = None
                    for j, (start, end) in enumerate(offsets):
                        if start < end_char <= end:
                            end_token = j
                            break
                    
                    # If the answer is not fully contained in the context, use start token as end token
                    if start_token is None:
                        start_token = 0
                    if end_token is None:
                        end_token = start_token
                    
                    tokenized_examples["start_positions"].append(start_token)
                    tokenized_examples["end_positions"].append(end_token)
                
                return tokenized_examples
            
            # Apply preprocessing
            tokenized_train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=train_dataset.column_names
            )
            
            if val_dataset:
                tokenized_val_dataset = val_dataset.map(
                    preprocess_function,
                    batched=True,
                    remove_columns=val_dataset.column_names
                )
            else:
                tokenized_val_dataset = None
            
            # Define training arguments
            training_args = TrainingArguments(
                output_dir=str(model_save_path),
                evaluation_strategy="epoch" if tokenized_val_dataset else "no",
                learning_rate=3e-5,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=epochs,
                weight_decay=0.01,
                save_strategy="epoch",
            )
            
            # Initialize Trainer
            data_collator = DefaultDataCollator()
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train_dataset,
                eval_dataset=tokenized_val_dataset,
                data_collator=data_collator,
                tokenizer=tokenizer,
            )
            
            # Train the model
            logger.info(f"Starting QA model training for {epochs} epochs")
            trainer.train()
            
            # Save the model
            model.save_pretrained(str(model_save_path))
            tokenizer.save_pretrained(str(model_save_path))
            
            logger.info(f"QA model successfully trained and saved to {model_save_path}")
            
            # Update active model configuration
            self._update_active_model_config("qa_model", str(model_save_path))
            
            return str(model_save_path)
            
        except Exception as e:
            logger.error(f"Error training QA model: {e}")
            return None
    
    def _update_active_model_config(self, model_type, model_path):
        """
        Update the active model configuration
        
        Args:
            model_type (str): Type of model ('embedding_model', 'cross_encoder_model', or 'qa_model')
            model_path (str): Path to the trained model
        """
        # Read existing config if it exists
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
            except Exception:
                config = {}
        else:
            config = {}
        
        # Update config
        config[model_type] = model_path
        config["last_updated"] = datetime.now().isoformat()
        
        # Write updated config
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Updated active model configuration: {model_type} -> {model_path}")
    
    def get_active_models(self):
        """
        Get the currently active models configuration
        
        Returns:
            dict: Dictionary of active model paths
        """
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading active models config: {e}")
                return {}
        else:
            return {}
    
    def generate_training_data_from_documents(self, documents, num_examples=1000):
        """
        Generate synthetic training data from document chunks
        
        Args:
            documents (list): List of document texts
            num_examples (int): Number of examples to generate
            
        Returns:
            tuple: (embedding_training_data, cross_encoder_training_data, qa_training_data)
        """
        # This is a simplified implementation - in a real system, this would be more sophisticated
        
        embedding_data = []
        cross_encoder_data = []
        qa_data = []
        
        try:
            # Split documents into paragraphs
            all_paragraphs = []
            for doc in documents:
                paragraphs = [p for p in doc.split('\n\n') if len(p) > 50]
                all_paragraphs.extend(paragraphs)
            
            if not all_paragraphs:
                logger.warning("No usable paragraphs found in documents")
                return [], [], []
            
            # For embedding and cross-encoder models: generate similar and dissimilar pairs
            import random
            
            # Generate similar pairs (paragraphs from same document)
            for _ in range(min(num_examples // 2, 500)):
                doc_idx = random.randint(0, len(documents) - 1)
                doc_paragraphs = [p for p in documents[doc_idx].split('\n\n') if len(p) > 50]
                
                if len(doc_paragraphs) >= 2:
                    p1, p2 = random.sample(doc_paragraphs, 2)
                    similarity = 0.7 + random.random() * 0.3  # Random similarity between 0.7 and 1.0
                    
                    embedding_data.append([p1, p2, similarity])
                    cross_encoder_data.append([p1, p2, similarity])
            
            # Generate dissimilar pairs (paragraphs from different documents)
            for _ in range(min(num_examples // 2, 500)):
                if len(documents) >= 2:
                    doc1_idx, doc2_idx = random.sample(range(len(documents)), 2)
                    
                    doc1_paragraphs = [p for p in documents[doc1_idx].split('\n\n') if len(p) > 50]
                    doc2_paragraphs = [p for p in documents[doc2_idx].split('\n\n') if len(p) > 50]
                    
                    if doc1_paragraphs and doc2_paragraphs:
                        p1 = random.choice(doc1_paragraphs)
                        p2 = random.choice(doc2_paragraphs)
                        similarity = random.random() * 0.3  # Random similarity between 0 and 0.3
                        
                        embedding_data.append([p1, p2, similarity])
                        cross_encoder_data.append([p1, p2, similarity])
            
            # For QA model: generate question-answer pairs
            # In a real implementation, this would use more sophisticated methods
            for _ in range(min(num_examples, 500)):
                para = random.choice(all_paragraphs)
                sentences = [s for s in para.split('.') if len(s) > 20]
                
                if sentences:
                    # Select a sentence to base the question on
                    answer_sentence = random.choice(sentences)
                    
                    # Extract a potential answer - this is highly simplified
                    words = answer_sentence.split()
                    if len(words) > 5:
                        start_idx = random.randint(0, len(words) - 3)
                        span_length = random.randint(1, min(3, len(words) - start_idx))
                        answer = ' '.join(words[start_idx:start_idx + span_length])
                        
                        # Generate a simple question - very basic implementation
                        question_type = random.choice(["What is", "Describe", "Explain", "How does"])
                        question = f"{question_type} {answer}?"
                        
                        qa_data.append([question, para, answer])
            
            logger.info(f"Generated training data - Embedding: {len(embedding_data)}, "
                        f"Cross-encoder: {len(cross_encoder_data)}, QA: {len(qa_data)}")
            
            return embedding_data, cross_encoder_data, qa_data
            
        except Exception as e:
            logger.error(f"Error generating training data: {e}")
            return [], [], []

# Helper function to get the latest trained models
def get_latest_trained_models(models_dir="models"):
    """
    Get the latest trained models from the models directory
    
    Args:
        models_dir (str): Directory containing trained models
        
    Returns:
        dict: Dictionary with paths to the latest models
    """
    models_path = Path(models_dir)
    
    # Check if active_models.json exists
    config_file = models_path / "active_models.json"
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading active models config: {e}")
    
    # If no config file or error reading it, try to find the latest models by timestamp
    result = {}
    
    # Find latest embedding model
    embedding_dir = models_path / "embedding"
    if embedding_dir.exists():
        embedding_models = list(embedding_dir.glob("embedding_model_*"))
        if embedding_models:
            latest = max(embedding_models, key=lambda p: p.stat().st_mtime)
            result["embedding_model"] = str(latest)
    
    # Find latest cross-encoder model
    cross_encoder_dir = models_path / "cross-encoder"
    if cross_encoder_dir.exists():
        cross_encoder_models = list(cross_encoder_dir.glob("cross_encoder_*"))
        if cross_encoder_models:
            latest = max(cross_encoder_models, key=lambda p: p.stat().st_mtime)
            result["cross_encoder_model"] = str(latest)
    
    # Find latest QA model
    qa_dir = models_path / "qa"
    if qa_dir.exists():
        qa_models = list(qa_dir.glob("qa_model_*"))
        if qa_models:
            latest = max(qa_models, key=lambda p: p.stat().st_mtime)
            result["qa_model"] = str(latest)
    
    return result
