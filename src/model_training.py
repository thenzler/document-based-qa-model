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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, Trainer, TrainingArguments
import re

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
        self.categorical_features = None
        self.numerical_features = None
        
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
        Preprocess data for training with enhanced feature handling.
        
        Args:
            data: DataFrame containing the data
            target_col: Name of the target column
            
        Returns:
            Tuple of (preprocessing_pipeline, X, y)
        """
        # Split features and target
        X = data.drop(columns=[target_col]) if target_col in data.columns else data.copy()
        y = data[target_col] if target_col in data.columns else None
        
        # Identify feature types
        self.numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Create preprocessing pipeline
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessing = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features if self.categorical_features else [])
            ]
        )
        
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
        
        # Get feature names after preprocessing
        if hasattr(self.pipeline.named_steps['preprocessing'], 'get_feature_names_out'):
            feature_names = self.pipeline.named_steps['preprocessing'].get_feature_names_out()
        else:
            feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]
        
        shap.summary_plot(
            shap_values, 
            X_processed, 
            feature_names=feature_names,
            show=False
        )
        plt.tight_layout()
        
        # Ensure directory exists
        os.makedirs('models', exist_ok=True)
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
            'model_type': self.model_type,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features
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
        self.numerical_features = model_data.get('numerical_features', [])
        self.categorical_features = model_data.get('categorical_features', [])
        
        print(f"Model loaded from {load_path}")
    
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on new data with additional risk insights.
        
        Args:
            data: DataFrame containing the new data
            
        Returns:
            DataFrame with predictions and risk insights
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
        
        # Add risk categories in German
        results['risk_category'] = pd.cut(
            results['churn_probability'],
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Niedriges Risiko', 'Mittleres Risiko', 'Hohes Risiko']
        )
        
        # Add risk factors if model supports feature importance
        if hasattr(self.model, 'feature_importances_') or hasattr(self.model, 'coef_'):
            # Get the preprocessor to transform the data
            preprocessor = self.pipeline.named_steps['preprocessing']
            X_processed = preprocessor.transform(data)
            
            # Get feature names after preprocessing
            if hasattr(preprocessor, 'get_feature_names_out'):
                feature_names = preprocessor.get_feature_names_out()
            else:
                feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]
            
            # For each customer, identify the top risk factors
            for i in range(len(data)):
                if probabilities[i] > 0.3:  # Only for medium/high risk
                    # Get feature contributions using either feature_importances_ or coef_
                    if hasattr(self.model, 'feature_importances_'):
                        # For tree-based models, use a simple heuristic
                        # Multiply feature values by importance
                        contributions = X_processed[i] * self.model.feature_importances_
                    else:
                        # For logistic regression
                        contributions = X_processed[i] * self.model.coef_[0]
                    
                    # Get top 3 contributing features
                    top_idx = np.argsort(contributions)[-3:]
                    top_features = [feature_names[idx] for idx in top_idx]
                    
                    # Add to results
                    results.at[i, 'top_risk_factors'] = ', '.join(top_features)
        
        return results


class DocumentEnhancedChurnModel(ChurnModel):
    """Churn prediction model enhanced with document-based features"""
    
    def __init__(self, model_type: str = "RandomForest", doc_embeddings: Optional[np.ndarray] = None, 
                 doc_mapping: Optional[Dict] = None):
        """
        Initialize the document-enhanced churn prediction model.
        
        Args:
            model_type: Type of model ('RandomForest', 'GradientBoosting', or 'LogisticRegression')
            doc_embeddings: Optional document embeddings for recommendations
            doc_mapping: Optional mapping from embedding indices to documents
        """
        super().__init__(model_type=model_type)
        self.doc_embeddings = doc_embeddings
        self.doc_mapping = doc_mapping
        self.doc_intervention_map = {
            'Niedriges Risiko': ['standard_betreuung.txt', 'check_ins.txt'],
            'Mittleres Risiko': ['schulungen.txt', 'feature_demo.txt', 'erfolgsgeschichten.txt'],
            'Hohes Risiko': ['rabatte.txt', 'premium_support.txt', 'bedarfsanalyse.txt']
        }
    
    def set_document_embeddings(self, embeddings: np.ndarray, doc_mapping: Dict):
        """
        Set document embeddings for recommendation.
        
        Args:
            embeddings: Document embeddings matrix
            doc_mapping: Mapping from embedding indices to documents
        """
        self.doc_embeddings = embeddings
        self.doc_mapping = doc_mapping
    
    def predict_with_interventions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions with intervention recommendations based on documents.
        
        Args:
            data: DataFrame containing customer data
            
        Returns:
            DataFrame with predictions and intervention recommendations
        """
        # Get standard predictions
        predictions = self.predict(data)
        
        # Add intervention recommendations based on risk category
        recommendations = []
        for i, row in predictions.iterrows():
            risk_level = row['risk_category']
            
            # Get relevant documents for this risk level
            relevant_docs = self.doc_intervention_map.get(risk_level, [])
            
            if relevant_docs:
                # Simple mapping-based recommendation
                recommendations.append(', '.join(relevant_docs))
            else:
                recommendations.append('')
        
        predictions['recommended_interventions'] = recommendations
        
        # Add document sources if available
        if self.doc_embeddings is not None and self.doc_mapping is not None:
            # For now, just use a placeholder - in a real implementation, you'd match
            # customer features to document relevance
            doc_sources = []
            for i, row in predictions.iterrows():
                if row['risk_category'] == 'Hohes Risiko':
                    doc_sources.append('premium_support.txt, rabatte.txt')
                elif row['risk_category'] == 'Mittleres Risiko':
                    doc_sources.append('schulungen.txt, feature_demo.txt')
                else:
                    doc_sources.append('standard_betreuung.txt')
            
            predictions['document_sources'] = doc_sources
        
        return predictions
    
    def explain_prediction_with_documents(self, customer_data: pd.Series) -> Dict:
        """
        Provide a document-backed explanation for a customer prediction.
        
        Args:
            customer_data: Series with a single customer's data
            
        Returns:
            Dictionary with explanation and document references
        """
        # Make prediction for this customer
        df = pd.DataFrame([customer_data])
        prediction = self.predict(df).iloc[0]
        
        # Prepare explanation
        risk_level = prediction['risk_category']
        probability = prediction['churn_probability']
        
        # Create explanation dictionary
        explanation = {
            'risk_level': risk_level,
            'probability': probability,
            'prediction': bool(prediction['churn_prediction']),
            'risk_factors': prediction.get('top_risk_factors', '').split(', '),
            'document_references': []
        }
        
        # Add document references based on risk level
        if risk_level == 'Hohes Risiko':
            explanation['document_references'] = [
                {'filename': 'use_case.txt', 'section': 'Hochrisiko-Kunden (>70%)'},
                {'filename': 'implementation.txt', 'section': 'Hyperparameter-Optimierung'}
            ]
        elif risk_level == 'Mittleres Risiko':
            explanation['document_references'] = [
                {'filename': 'use_case.txt', 'section': 'Mittleres Risiko (30-70%)'},
                {'filename': 'churn_theory.txt', 'section': 'Feature Engineering für Churn Prediction'}
            ]
        else:
            explanation['document_references'] = [
                {'filename': 'use_case.txt', 'section': 'Niedriges Risiko (<30%)'}
            ]
        
        return explanation


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
    
    def extract_qa_pairs_from_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Extract potential QA pairs from documents using patterns.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of QA examples
        """
        qa_examples = []
        
        # Pattern for QA format like "## Question: What is X? **Answer:** Y"
        qa_pattern = re.compile(r'(?:##?\s*)?(?:Frage|Question)\s*(?:\d+)?:?\s*([^\n]+)\s*(?:\n+[^\n]*(?:Antwort|Answer)[^\n]*:?\s*)(.*?)(?=\n\s*(?:##?\s*)?(?:Frage|Question)|$)', re.DOTALL)
        
        for doc in documents:
            text = doc["text"]
            
            # Search for QA patterns
            matches = qa_pattern.findall(text)
            
            for match in matches:
                question = match[0].strip()
                answer = match[1].strip()
                
                # Clean up formatting
                answer = re.sub(r'\*\*|\*', '', answer)  # Remove Markdown formatting
                
                # Add to examples if both question and answer are present
                if question and answer:
                    example = {
                        "question": question,
                        "context": text,
                        "answer": answer,
                        "source": doc["source"]
                    }
                    qa_examples.append(example)
        
        print(f"Extracted {len(qa_examples)} QA pairs from documents")
        return qa_examples


class DocumentBasedQATrainer:
    """Specialized trainer for document-based QA models"""
    
    def __init__(self, qa_trainer: QAModelTrainer):
        """
        Initialize the document-based QA trainer.
        
        Args:
            qa_trainer: QA model trainer for the base QA capabilities
        """
        self.qa_trainer = qa_trainer
        self.training_examples = []
        self.document_sources = {}
    
    def prepare_training_data_from_docs(self, documents: List[Dict], doc_processor=None):
        """
        Prepare training data from documents.
        
        Args:
            documents: List of document dictionaries
            doc_processor: Optional document processor for additional processing
            
        Returns:
            Number of examples created
        """
        # Extract QA pairs from documents
        qa_examples = self.qa_trainer.extract_qa_pairs_from_documents(documents)
        
        # Store document sources for each example
        for example in qa_examples:
            self.document_sources[example["question"]] = example["source"]
        
        # Add to training examples
        self.training_examples.extend(qa_examples)
        
        # If we have a document processor, add document keywords
        if doc_processor:
            for example in self.training_examples:
                # Extract keywords using the document processor
                keywords = doc_processor.extract_keywords(example["question"])
                example["keywords"] = list(keywords)
        
        return len(qa_examples)
    
    def create_training_dataset(self):
        """
        Create a PyTorch dataset from the training examples.
        
        Returns:
            PyTorch Dataset for training
        """
        return self.qa_trainer.prepare_qa_data(self.training_examples)
    
    def train_model(self, output_dir: str, epochs: int = 3, batch_size: int = 8):
        """
        Train the QA model on the document-based examples.
        
        Args:
            output_dir: Directory to save the model
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Trained model
        """
        if not self.training_examples:
            raise ValueError("No training examples available. Call prepare_training_data_from_docs first.")
        
        # Create dataset
        train_dataset = self.create_training_dataset()
        
        # Train model
        self.qa_trainer.train_qa_model(train_dataset, output_dir, epochs, batch_size)
        
        return self.qa_trainer.model
    
    def get_document_source(self, question: str) -> Optional[str]:
        """
        Get the document source for a question.
        
        Args:
            question: The question text
            
        Returns:
            Document source if available, None otherwise
        """
        return self.document_sources.get(question)


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
