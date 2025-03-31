"""
Model Training Module for Document-based QA System

This module handles the training of models for:
- Document retrieval (ranking documents by relevance)
- Question answering (extracting or generating answers from documents)
- Churn prediction (as an application example)
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Tuple, Optional, Any
from pathlib import Path
import pickle
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, Trainer, TrainingArguments

import shap
import torch
from torch.utils.data import Dataset, DataLoader


class ChurnModel:
    """Model for churn prediction using structured data"""
    
    def __init__(self, model_type: str = "RandomForest"):
        """
        Initialize the churn prediction model.
        
        Args:
            model_type: Type of model to use ('RandomForest', 'GradientBoosting', or 'LogisticRegression')
        """
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        self.pipeline = None
        
        # Initialize the appropriate model
        if model_type == "RandomForest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=42
            )
        elif model_type == "GradientBoosting":
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            )
        elif model_type == "LogisticRegression":
            self.model = LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def preprocess_data(self, data: pd.DataFrame, target_col: str = "Churn") -> Tuple[Pipeline, pd.DataFrame, pd.Series]:
        """
        Preprocess data for training.
        
        Args:
            data: DataFrame containing the data
            target_col: Name of the target column
            
        Returns:
            Tuple of (preprocessing_pipeline, X, y)
        """
        # Split features and target
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Create preprocessing pipeline
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        preprocessing = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        return preprocessing, X, y
    
    def train(self, data: pd.DataFrame, target_col: str = "Churn", test_size: float = 0.2):
        """
        Train the churn prediction model.
        
        Args:
            data: DataFrame containing the training data
            target_col: Name of the target column
            test_size: Proportion of data to use for testing
        
        Returns:
            Dictionary with training results
        """
        # Preprocess data
        preprocessing, X, y = self.preprocess_data(data, target_col)
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Create and train the pipeline
        self.pipeline = Pipeline([
            ('preprocessing', preprocessing),
            ('model', self.model)
        ])
        
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.pipeline.predict(X_test)
        y_prob = self.pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = self.pipeline.score(X_test, y_test)
        roc_auc = roc_auc_score(y_test, y_prob)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Generate feature importance
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = abs(self.model.coef_[0])
        else:
            importances = None
        
        # Store results
        results = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'report': report,
            'confusion_matrix': cm,
            'feature_importances': importances,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
        
        print(f"Model training completed with accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}")
        
        return results
    
    def optimize_hyperparameters(self, data: pd.DataFrame, target_col: str = "Churn", cv: int = 5):
        """
        Optimize model hyperparameters using GridSearchCV.
        
        Args:
            data: DataFrame containing the training data
            target_col: Name of the target column
            cv: Number of cross-validation folds
            
        Returns:
            Best model with optimized hyperparameters
        """
        # Preprocess data
        preprocessing, X, y = self.preprocess_data(data, target_col)
        
        # Define parameter grid based on model type
        if self.model_type == "RandomForest":
            param_grid = {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [None, 10, 20],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4],
                'model__class_weight': ['balanced', 'balanced_subsample']
            }
        elif self.model_type == "GradientBoosting":
            param_grid = {
                'model__n_estimators': [50, 100, 200],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__max_depth': [3, 5, 7],
                'model__min_samples_split': [2, 5, 10],
                'model__subsample': [0.8, 0.9, 1.0]
            }
        elif self.model_type == "LogisticRegression":
            param_grid = {
                'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
                'model__penalty': ['l2'],
                'model__solver': ['liblinear', 'sag', 'saga'],
                'model__class_weight': ['balanced', None]
            }
        
        # Create pipeline for grid search
        pipe = Pipeline([
            ('preprocessing', preprocessing),
            ('model', self.model)
        ])
        
        # Run grid search
        grid_search = GridSearchCV(
            pipe,
            param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        # Update model with best estimator
        self.pipeline = grid_search.best_estimator_
        self.model = grid_search.best_estimator_.named_steps['model']
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best ROC AUC: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def explain_model(self, data: Optional[pd.DataFrame] = None, X_test: Optional[pd.DataFrame] = None):
        """
        Explain model predictions using SHAP values.
        
        Args:
            data: Optional DataFrame to use for explanations
            X_test: Optional test data to use for explanations
            
        Returns:
            SHAP explainer object
        """
        if self.pipeline is None:
            raise ValueError("Model must be trained before explanation")
        
        # Get test data
        if X_test is None:
            if data is not None:
                _, X, _ = self.preprocess_data(data)
                _, X_test, _, _ = train_test_split(X, data["Churn"], test_size=0.2, random_state=42)
            else:
                raise ValueError("Either data or X_test must be provided")
        
        # Preprocess the test data
        X_processed = self.pipeline.named_steps['preprocessing'].transform(X_test)
        
        # Create SHAP explainer based on model type
        if self.model_type in ["RandomForest", "GradientBoosting"]:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_processed)
            
            # For binary classification, shap_values is a list where index 1 is the positive class
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            # For LogisticRegression
            explainer = shap.KernelExplainer(
                self.model.predict_proba, 
                X_processed[:100]  # Use subset for kernel explainer
            )
            shap_values = explainer.shap_values(X_processed)[1]
        
        # Generate summary plot
        plt.figure(figsize=(10, 8))
        feature_names = self.feature_names
        shap.summary_plot(
            shap_values, 
            X_processed, 
            feature_names=feature_names,
            show=False
        )
        plt.tight_layout()
        plt.savefig('models/shap_summary.png')
        plt.close()
        
        # Return explainer for further use
        return explainer
    
    def save_model(self, save_path: Union[str, Path]):
        """
        Save the trained model to a file.
        
        Args:
            save_path: Path to save the model
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'pipeline': self.pipeline,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {save_path}")
    
    def load_model(self, load_path: Union[str, Path]):
        """
        Load a trained model from a file.
        
        Args:
            load_path: Path to load the model from
        """
        with open(load_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.pipeline = model_data['pipeline']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.model = self.pipeline.named_steps['model']
        
        print(f"Model loaded from {load_path}")
    
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on new data.
        
        Args:
            data: DataFrame containing the new data
            
        Returns:
            DataFrame with predictions
        """
        if self.pipeline is None:
            raise ValueError("Model must be trained or loaded before prediction")
        
        # Make predictions
        probabilities = self.pipeline.predict_proba(data)[:, 1]
        predictions = self.pipeline.predict(data)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'churn_probability': probabilities,
            'churn_prediction': predictions
        })
        
        # Add risk categories
        results['risk_category'] = pd.cut(
            results['churn_probability'],
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
        
        return results


class QAModelTrainer:
    """Trainer for question answering models"""
    
    def __init__(self, model_name: str = "deepset/gbert-base"):
        """
        Initialize the QA model trainer.
        
        Args:
            model_name: Name of the pre-trained model to fine-tune
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def prepare_qa_data(self, qa_examples: List[Dict]) -> Dataset:
        """
        Prepare QA data for training.
        
        Args:
            qa_examples: List of QA examples with context, question, and answers
            
        Returns:
            PyTorch Dataset for training
        """
        # Create a custom dataset
        class QADataset(Dataset):
            def __init__(self, tokenizer, examples):
                self.examples = examples
                self.tokenizer = tokenizer
                
            def __len__(self):
                return len(self.examples)
                
            def __getitem__(self, idx):
                example = self.examples[idx]
                
                # Tokenize inputs
                inputs = self.tokenizer(
                    example["question"],
                    example["context"],
                    truncation="only_second",
                    max_length=384,
                    stride=128,
                    return_overflowing_tokens=True,
                    return_offsets_mapping=True,
                    padding="max_length",
                    return_tensors="pt"
                )
                
                # Find start and end positions
                answer_start = example["context"].find(example["answer"])
                answer_end = answer_start + len(example["answer"]) - 1
                
                # Convert to token positions
                offset_mapping = inputs.pop("offset_mapping")[0]
                
                start_positions = []
                end_positions = []
                
                for i, offset in enumerate(offset_mapping):
                    if offset[0] <= answer_start < offset[1]:
                        start_positions.append(i)
                    if offset[0] <= answer_end < offset[1]:
                        end_positions.append(i)
                
                # Use the first valid start and end positions
                start_position = start_positions[0] if start_positions else 0
                end_position = end_positions[0] if end_positions else 0
                
                # Create labels
                return {
                    "input_ids": inputs["input_ids"][0],
                    "attention_mask": inputs["attention_mask"][0],
                    "start_positions": torch.tensor(start_position),
                    "end_positions": torch.tensor(end_position)
                }
        
        return QADataset(self.tokenizer, qa_examples)
    
    def train_qa_model(self, train_dataset: Dataset, output_dir: str, epochs: int = 3, batch_size: int = 8):
        """
        Train the QA model.
        
        Args:
            train_dataset: Dataset for training
            output_dir: Directory to save the model
            epochs: Number of training epochs
            batch_size: Training batch size
        """
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            save_strategy="epoch"
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )
        
        trainer.train()
        
        # Save model and tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"QA model training completed and saved to {output_dir}")
    
    def create_qa_examples_from_documents(self, documents: List[Dict], questions: List[Dict]) -> List[Dict]:
        """
        Create QA examples from documents and questions.
        
        Args:
            documents: List of document dictionaries
            questions: List of question dictionaries with answers
            
        Returns:
            List of QA examples
        """
        qa_examples = []
        
        # Map document texts by source
        doc_texts = {}
        for doc in documents:
            doc_texts[doc["source"]] = doc["text"]
        
        for question in questions:
            source = question.get("source")
            if source and source in doc_texts:
                context = doc_texts[source]
                
                # Create example
                example = {
                    "question": question["question"],
                    "context": context,
                    "answer": question["answer"]
                }
                
                # Verify answer is in context
                if example["answer"] in context:
                    qa_examples.append(example)
                else:
                    print(f"Warning: Answer '{example['answer']}' not found in context for question '{example['question']}'")
        
        print(f"Created {len(qa_examples)} QA examples")
        return qa_examples


if __name__ == "__main__":
    # Example usage for churn model
    data = pd.read_csv("data/customer_data.csv")
    churn_model = ChurnModel("RandomForest")
    results = churn_model.train(data)
    churn_model.save_model("models/churn_model.pkl")
    
    # Example for QA model
    # qa_trainer = QAModelTrainer()
    # Load documents and questions
    # Create QA examples
    # Prepare dataset
    # Train model
