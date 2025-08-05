import pandas as pd
import numpy as np
import os
import pickle
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from core.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

class ChurnModelTrainer:
    """
    A comprehensive model trainer for credit card churn prediction using RandomForestClassifier.
    """
    
    def __init__(self, models_dir='models', random_state=42):
        """
        Initialize the model trainer.
        
        Args:
            models_dir (str): Directory to save models and results
            random_state (int): Random state for reproducibility
        """
        self.models_dir = models_dir
        self.random_state = random_state
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.training_results = {}
        
        # Create models directory
        os.makedirs(self.models_dir, exist_ok=True)
        logger.info(f"Model trainer initialized with models directory: {self.models_dir}")
    

    def load_processed_data(self, data_path='data/processed_features.csv'):
        """
        Load the processed data for training.
        
        Args:
            data_path (str): Path to the processed data file
        
        Returns:
            tuple: (X, y) features and target
        """
        logger.info(f"Loading processed data from: {data_path}")
        
        try:
            df = pd.read_csv(data_path)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            
            # Separate features and target
            if 'Churn' in df.columns:
                X = df.drop('Churn', axis=1)
                y = df['Churn']
                
                self.feature_names = list(X.columns)
                logger.info(f"Features: {len(X.columns)}")
                logger.info(f"Target distribution:\n{y.value_counts()}")
                
                return X, y
            else:
                logger.error("Target column 'Churn' not found in the dataset")
                return None, None
                
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None, None


    def split_data(self, X, y, test_size=0.2, stratify=True):
        """
        Split data into training and testing sets.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            test_size (float): Proportion of test set
            stratify (bool): Whether to stratify the split
        """
        logger.info("Splitting data into train and test sets...")
        
        stratify_param = y if stratify else None
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=stratify_param
        )
        
        logger.info(f"Training set shape: {self.X_train.shape}")
        logger.info(f"Test set shape: {self.X_test.shape}")
        logger.info(f"Training target distribution:\n{self.y_train.value_counts()}")
        logger.info(f"Test target distribution:\n{self.y_test.value_counts()}")


    def train_basic_model(self, **kwargs):
        """
        Train a basic RandomForestClassifier with default parameters.
        
        Args:
            **kwargs: Additional parameters for RandomForestClassifier
        """
        logger.info("Training basic RandomForestClassifier...")
        
        # Default parameters
        default_params = {
            'n_estimators': 100,
            'random_state': self.random_state,
            'n_jobs': -1
        }
        default_params.update(kwargs)
        
        self.model = RandomForestClassifier(**default_params)
        self.model.fit(self.X_train, self.y_train)
        
        logger.info("Basic model training completed")
        
        # Evaluate basic model
        train_score = self.model.score(self.X_train, self.y_train)
        test_score = self.model.score(self.X_test, self.y_test)
        
        logger.info(f"Basic model - Training accuracy: {train_score:.4f}")
        logger.info(f"Basic model - Test accuracy: {test_score:.4f}")
        
        return self.model

    
    def evaluate_model(self, model=None, save_plots=True):
        """
        Comprehensive model evaluation.
        
        Args:
            model: Model to evaluate (uses best_model if None)
            save_plots (bool): Whether to save evaluation plots
        
        Returns:
            dict: Evaluation metrics
        """
        logger.info("Starting comprehensive model evaluation...")
        
        if model is None:
            model = self.model

        if model is None:
            logger.error("No model available for evaluation")
            return None
        
        # Make predictions
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)
        y_test_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'train_accuracy': accuracy_score(self.y_train, y_train_pred),
            'test_accuracy': accuracy_score(self.y_test, y_test_pred),
            'precision': precision_score(self.y_test, y_test_pred),
            'recall': recall_score(self.y_test, y_test_pred),
            'f1_score': f1_score(self.y_test, y_test_pred),
            'roc_auc': roc_auc_score(self.y_test, y_test_proba)
        }
        
        # Log metrics
        logger.info("Model Evaluation Results:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='f1')
        metrics['cv_f1_mean'] = cv_scores.mean()
        metrics['cv_f1_std'] = cv_scores.std()
        
        logger.info(f"Cross-validation F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Classification report
        class_report = classification_report(self.y_test, y_test_pred, output_dict=True)
        metrics['classification_report'] = class_report
        
        logger.info("Classification Report:")
        logger.info(f"\n{classification_report(self.y_test, y_test_pred)}")
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_test_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        if save_plots:
            self._save_evaluation_plots(model, y_test_pred, y_test_proba, cm)
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            metrics['feature_importance'] = feature_importance.to_dict('records')
            
            logger.info("Top 10 Feature Importances:")
            for idx, row in feature_importance.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
            
            if save_plots:
                self._save_feature_importance_plot(feature_importance)
        
        self.training_results['evaluation_metrics'] = metrics
        return metrics


    def _save_evaluation_plots(self, model, y_test_pred, y_test_proba, cm):
        """Save evaluation plots including confusion matrix and ROC curve."""
        logger.info("Saving evaluation plots...")
        
        # Create plots directory in root (not inside models)
        plots_dir = 'evaluation_plots'
        os.makedirs(plots_dir, exist_ok=True)
        
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Churn', 'Churn'],
                    yticklabels=['No Churn', 'Churn'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, y_test_proba)
        roc_auc = roc_auc_score(self.y_test, y_test_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Evaluation plots saved to {plots_dir}")


    def _save_feature_importance_plot(self, feature_importance):
        """Save feature importance plot."""
        plots_dir = 'evaluation_plots'
        
        plt.figure(figsize=(10, 8))
        top_features = feature_importance.head(15)
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title('Top 15 Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Feature importance plot saved")


    def save_model(self, model=None, model_name=None):
        """
        Save the trained model and metadata.
        
        Args:
            model: Model to save (uses best_model if None)
            model_name (str): Name for the saved model
        """
        logger.info("Saving trained model...")
        
        if model is None:
            model = self.model
        
        if model is None:
            logger.error("No model available to save")
            return None
        
        # Generate model name if not provided
        if model_name is None:
            model_name = "random_forest_churn_model"
        
        # Save model
        model_file = os.path.join(self.models_dir, f"{model_name}.pkl")
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        
        # Save model metadata
        metadata = {
            'model_name': model_name,
            'model_type': 'RandomForestClassifier',
            'training_date': datetime.now().isoformat(),
            'feature_names': self.feature_names,
            'training_shape': self.X_train.shape,
            'test_shape': self.X_test.shape,
            'model_parameters': model.get_params(),
            'training_results': self.training_results
        }
        
        metadata_file = os.path.join(self.models_dir, f"{model_name}_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Model saved to: {model_file}")
        logger.info(f"Metadata saved to: {metadata_file}")
        
        return model_file, metadata_file


    def train_complete_pipeline(self, data_path='data/processed_features.csv', save_model_flag=True):
        """
        Run the complete training pipeline.
        
        Args:
            data_path (str): Path to processed data
            save_model_flag (bool): Whether to save the final model
        
        Returns:
            dict: Training results and metrics
        """
        logger.info("="*60)
        logger.info("STARTING COMPLETE MODEL TRAINING PIPELINE")
        logger.info("="*60)
        
        # Load data
        X, y = self.load_processed_data(data_path)
        if X is None or y is None:
            logger.error("Failed to load data. Aborting training.")
            return None
        
        # Split data
        self.split_data(X, y)
        
        # Train basic model
        self.train_basic_model()
        
        # Evaluate model
        logger.info("\n" + "-"*40)
        logger.info("MODEL EVALUATION")
        logger.info("-"*40)
        metrics = self.evaluate_model()
        
        # Save model
        if save_model_flag:
            logger.info("\n" + "-"*40)
            logger.info("SAVING MODEL")
            logger.info("-"*40)
            self.save_model()
        
        logger.info("\n" + "="*60)
        logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        
        return {
            'model': self.model,
            'metrics': metrics,
            'training_results': self.training_results
        }


def train_churn_model(data_path=None, data=None, models_dir='models'):
    """
    Convenience function to train a churn prediction model.
    
    Args:
        data_path (str, optional): Path to processed data CSV file
        data (pd.DataFrame, optional): Pre-loaded DataFrame with features and target
        models_dir (str): Directory to save models
    
    Returns:
        ChurnModelTrainer: Trained model trainer instance
        
    Raises:
        ValueError: If neither data_path nor data is provided, or if both are provided
    """
    if data_path is None and data is None:
        raise ValueError("Either data_path or data must be provided")
    
    if data_path is not None and data is not None:
        raise ValueError("Provide either data_path or data, not both")
    
    trainer = ChurnModelTrainer(models_dir=models_dir)
    
    if data is not None:
        # Use provided DataFrame
        logger.info("Using provided DataFrame for training")
        if 'Churn' not in data.columns:
            raise ValueError("DataFrame must contain 'Churn' column as target")
        
        X = data.drop('Churn', axis=1)
        y = data['Churn']
        trainer.feature_names = list(X.columns)
        
        # Split data and train
        trainer.split_data(X, y)
        trainer.train_basic_model()
        metrics = trainer.evaluate_model()
        trainer.save_model()
        
        results = {
            'model': trainer.best_model if trainer.best_model else trainer.model,
            'metrics': metrics,
            'training_results': trainer.training_results
        }
    else:
        # Use file path
        results = trainer.train_complete_pipeline(data_path)
    
    return trainer, results


if __name__ == "__main__":
    logger.info("Model trainer module loaded successfully!")
    
    # Example usage:
    # trainer, results = train_churn_model()
    # print("Training completed successfully!")
