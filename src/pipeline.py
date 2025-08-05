import pandas as pd
import numpy as np
import pickle
import json
import os
from pathlib import Path
from typing import Union, Dict, List, Optional
from core.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

class ChurnPredictionPipeline:
    """
    Complete prediction pipeline for credit card churn prediction.
    Loads saved models and transformers to preprocess data and make predictions.
    """
    
    def __init__(self, models_dir='models'):
        """
        Initialize the prediction pipeline.
        
        Args:
            models_dir (str): Directory containing saved models and transformers
        """
        self.models_dir = models_dir
        self.model = None
        self.encoders = {}
        self.scaler = None
        self.feature_names = None
        self.model_metadata = None
        
        logger.info(f"Prediction pipeline initialized with models directory: {self.models_dir}")
    
    def load_model(self, model_name='random_forest_churn_model'):
        """
        Load the trained model and its metadata.
        
        Args:
            model_name (str): Name of the model file (without .pkl extension)
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            # Load model
            model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
            
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"Model loaded successfully from: {model_path}")
            
            # Load metadata
            metadata_path = os.path.join(self.models_dir, f"{model_name}_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
                self.feature_names = self.model_metadata.get('feature_names', [])
                logger.info(f"Model metadata loaded. Expected features: {len(self.feature_names)}")
            else:
                logger.warning(f"Metadata file not found: {metadata_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def load_transformers(self):
        """
        Load all saved transformers (encoding info and scalers).
        
        Returns:
            bool: True if transformers loaded successfully, False otherwise
        """
        try:
            # Load encoding info (not individual encoders)
            encoding_file = os.path.join(self.models_dir, 'encoding_info.pkl')
            if os.path.exists(encoding_file):
                with open(encoding_file, 'rb') as f:
                    self.encoders = pickle.load(f)
                logger.info(f"Encoding info loaded: {self.encoders.get('original_columns', [])}")
            else:
                logger.warning(f"Encoding info file not found: {encoding_file}")
                return False
            
            # Load scalers
            scalers_path = os.path.join(self.models_dir, 'all_scalers.pkl')
            if os.path.exists(scalers_path):
                with open(scalers_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info(f"Scalers loaded: {list(self.scaler.keys())}")
            else:
                logger.warning(f"Scalers file not found: {scalers_path}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading transformers: {str(e)}")
            return False
    
    def validate_input_data(self, data: pd.DataFrame) -> bool:
        """
        Validate input data format and required columns.
        
        Args:
            data (pd.DataFrame): Input data to validate
        
        Returns:
            bool: True if data is valid, False otherwise
        """
        required_columns = [
            'CustomerID', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
            'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
        ]
        
        # Check if all required columns are present
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Check data types and ranges
        validation_rules = {
            'Age': (lambda x: (x >= 18) & (x <= 100), "Age should be between 18 and 100"),
            'Balance': (lambda x: x >= 0, "Balance should be non-negative"),
            'NumOfProducts': (lambda x: x >= 1, "NumOfProducts should be between 1 and 10"),
            'HasCrCard': (lambda x: x.isin([0, 1]), "HasCrCard should be 0 or 1"),
            'IsActiveMember': (lambda x: x.isin([0, 1]), "IsActiveMember should be 0 or 1"),
            'Tenure': (lambda x: x >= 0, "Tenure should be between 0 and 20"),
            'EstimatedSalary': (lambda x: x >= 0, "EstimatedSalary should be non-negative"),
            'Gender': (lambda x: x.astype(str).str.strip().str.title().isin(['Male', 'Female']), "Gender should be 'Male' or 'Female'")
        }
        
        for column, (rule, message) in validation_rules.items():
            if column in data.columns:
                try:
                    validation_result = rule(data[column])
                    if not validation_result.all():
                        logger.error(f"Validation failed for {column}: {message}")
                        return False
                except Exception as e:
                    logger.error(f"Error validating {column}: {str(e)}")
                    return False
        
        logger.info("Input data validation passed")
        return True
    
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering steps used during training.
        
        Args:
            data (pd.DataFrame): Preprocessed data
        
        Returns:
            pd.DataFrame: Data with engineered features
        """
        logger.info("Starting feature engineering...")
        
        df = data.copy()
        
        # Create engineered features (same as in feature_engineering.py)
        try:
            # Create age bins (same as feature_engineering.py)
            df['AgeCategory'] = pd.cut(df['Age'], 
                                     bins=[0, 25, 35, 50, 100], 
                                     labels=['Young', 'Adult', 'Middle', 'Senior'])
            
            # Create balance categories (same as feature_engineering.py)
            df['BalanceCategory'] = pd.cut(df['Balance'], 
                                         bins=[-1, 0, 50000, 100000, np.inf], 
                                         labels=['Zero', 'Low', 'Medium', 'High'])
            
            # Balance per product (same as feature_engineering.py)
            df['BalancePerProduct'] = df['Balance'] / (df['NumOfProducts'] + 1)
            
            # Tenure-age ratio (same as feature_engineering.py)
            df['TenureAgeRatio'] = df['Tenure'] / (df['Age'] + 1)
            
            logger.info("Feature engineering completed successfully")
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            raise
        
        return df
    
    def apply_encoding(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the same encoding transformations used during training.
        
        Args:
            data (pd.DataFrame): Data with engineered features
        
        Returns:
            pd.DataFrame: Encoded data
        """
        logger.info("Applying encoding transformations...")
        
        if not self.encoders:
            logger.error("Encoding info not loaded. Call load_transformers() first.")
            return data
        
        df = data.copy()
        
        try:
            # Apply one-hot encoding using pandas get_dummies (same as training)
            categorical_columns = self.encoders.get('original_columns', ['Gender', 'AgeCategory', 'BalanceCategory'])
            
            # Apply get_dummies with drop_first=True (same as training)
            df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
            
            # Ensure we have the same columns as during training
            expected_columns = self.encoders.get('all_columns_after_encoding', [])
            
            # Add missing columns with zeros
            for col in expected_columns:
                if col not in df_encoded.columns:
                    df_encoded[col] = 0
                    logger.warning(f"Added missing column {col} with default value 0")
            
            # Remove extra columns
            for col in df_encoded.columns:
                if col not in expected_columns:
                    df_encoded = df_encoded.drop(col, axis=1)
                    logger.warning(f"Removed extra column {col}")
            
            # Reorder columns to match training
            df_encoded = df_encoded[expected_columns]
            
            logger.info("Encoding transformations completed")
            return df_encoded
            
        except Exception as e:
            logger.error(f"Error in encoding: {str(e)}")
            raise
    
    def apply_scaling(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the same scaling transformations used during training.
        
        Args:
            data (pd.DataFrame): Encoded data
        
        Returns:
            pd.DataFrame: Scaled data
        """
        logger.info("Applying scaling transformations...")
        
        if self.scaler is None:
            logger.error("Scalers not loaded. Call load_transformers() first.")
            return data
        
        try:
            df = data.copy()
            
            # Apply scaling using individual scalers for each column
            scaled_columns = ['Balance', 'EstimatedSalary', 'Age', 'Tenure', 
                            'NumOfProducts', 'BalancePerProduct', 'TenureAgeRatio']
            
            for col in scaled_columns:
                if col in df.columns and col in self.scaler:
                    scaler = self.scaler[col]
                    df[col] = scaler.transform(df[[col]]).flatten()
                    logger.info(f"Applied scaling for {col}")
                elif col in df.columns:
                    logger.warning(f"No scaler found for column {col}")
            
            logger.info("Scaling transformations completed")
            return df
            
        except Exception as e:
            logger.error(f"Error in scaling: {str(e)}")
            raise
    
    def ensure_feature_order(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure features are in the same order as during training.
        
        Args:
            data (pd.DataFrame): Processed data
        
        Returns:
            pd.DataFrame: Data with correctly ordered features
        """
        if self.feature_names is None:
            logger.warning("Feature names not available. Using current column order.")
            return data
        
        try:
            # Remove target column from feature names if present
            feature_names = [col for col in self.feature_names if col != 'Churn']
            
            # Check for missing features
            missing_features = set(feature_names) - set(data.columns)
            if missing_features:
                logger.error(f"Missing features required by model: {missing_features}")
                # Add missing features with zeros
                for feature in missing_features:
                    data[feature] = 0
                    logger.warning(f"Added missing feature {feature} with default value 0")
            
            # Check for extra features
            extra_features = set(data.columns) - set(feature_names)
            if extra_features:
                logger.warning(f"Extra features not used by model: {extra_features}")
                data = data.drop(columns=extra_features)
            
            # Reorder columns to match training (excluding target)
            data = data[feature_names]
            logger.info("Feature order aligned with training data")
            
        except Exception as e:
            logger.error(f"Error ensuring feature order: {str(e)}")
            raise
        
        return data
    
    def predict(self, data: Union[pd.DataFrame, str], return_probabilities: bool = False) -> Dict:
        """
        Make predictions on new data.
        
        Args:
            data (pd.DataFrame or str): Input data or path to CSV file
            return_probabilities (bool): Whether to return prediction probabilities
        
        Returns:
            Dict: Prediction results including predictions, probabilities, and input info
        """
        logger.info("="*60)
        logger.info("STARTING CHURN PREDICTION")
        logger.info("="*60)
        
        try:
            # Load data if path is provided
            if isinstance(data, str):
                logger.info(f"Loading data from: {data}")
                df = pd.read_csv(data)
            else:
                df = data.copy()
            
            logger.info(f"Input data shape: {df.shape}")
            
            # Store original customer IDs if present
            customer_ids = df['CustomerID'].tolist() if 'CustomerID' in df.columns else list(range(len(df)))
            
            # Validate input data
            if not self.validate_input_data(df):
                raise ValueError("Input data validation failed")
            
            # Check if models are loaded
            if self.model is None:
                logger.error("Model not loaded. Call load_model() first.")
                raise ValueError("Model not loaded")
            
            if not self.encoders or self.scaler is None:
                logger.error("Transformers not loaded. Call load_transformers() first.")
                raise ValueError("Transformers not loaded")
            
            # Apply preprocessing pipeline
            logger.info("\n" + "-"*40)
            logger.info("PREPROCESSING PIPELINE")
            logger.info("-"*40)
            
            # Remove CustomerID column for processing
            if 'CustomerID' in df.columns:
                df = df.drop('CustomerID', axis=1)
                logger.info("Removed CustomerID column for processing")
            
            # Step 1: Feature engineering
            df_engineered = self.engineer_features(df)
            
            # Step 2: Encoding
            df_encoded = self.apply_encoding(df_engineered)
            
            # Step 3: Scaling
            df_scaled = self.apply_scaling(df_encoded)
            
            # Step 4: Ensure feature order
            df_final = self.ensure_feature_order(df_scaled)
            
            logger.info(f"Final processed data shape: {df_final.shape}")
            
            # Make predictions
            logger.info("\n" + "-"*40)
            logger.info("MAKING PREDICTIONS")
            logger.info("-"*40)
            
            predictions = self.model.predict(df_final)
            probabilities = self.model.predict_proba(df_final)
            
            # Create results
            results = {
                'customer_ids': customer_ids,
                'predictions': predictions.tolist(),
                'prediction_labels': ['No Churn' if p == 0 else 'Churn' for p in predictions],
                'churn_probability': probabilities[:, 1].tolist(),
                'no_churn_probability': probabilities[:, 0].tolist(),
                'total_customers': len(predictions),
                'predicted_churn_count': int(sum(predictions)),
                'predicted_churn_rate': float(sum(predictions)) / len(predictions) * 100
            }
            
            if return_probabilities:
                results['probabilities'] = probabilities.tolist()
            
            # Log results summary
            logger.info(f"Predictions completed for {len(predictions)} customers")
            logger.info(f"Predicted churn count: {results['predicted_churn_count']}")
            logger.info(f"Predicted churn rate: {results['predicted_churn_rate']:.2f}%")
            
            logger.info("\n" + "="*60)
            logger.info("PREDICTION COMPLETED SUCCESSFULLY")
            logger.info("="*60)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in prediction pipeline: {str(e)}")
            raise

    
    def load_pipeline(self, model_name='random_forest_churn_model'):
        """
        Load complete pipeline (model + transformers).
        
        Args:
            model_name (str): Name of the model to load
        
        Returns:
            bool: True if pipeline loaded successfully
        """
        logger.info("Loading complete prediction pipeline...")
        
        model_loaded = self.load_model(model_name)
        transformers_loaded = self.load_transformers()
        
        if model_loaded and transformers_loaded:
            logger.info("Complete pipeline loaded successfully!")
            return True
        else:
            logger.error("Failed to load complete pipeline")
            return False

def create_prediction_pipeline(models_dir='models', model_name='random_forest_churn_model'):
    """
    Convenience function to create and load a complete prediction pipeline.
    
    Args:
        models_dir (str): Directory containing saved models
        model_name (str): Name of the model to load
    
    Returns:
        ChurnPredictionPipeline: Ready-to-use prediction pipeline
    """
    pipeline = ChurnPredictionPipeline(models_dir=models_dir)
    
    if pipeline.load_pipeline(model_name):
        logger.info("Prediction pipeline ready for use!")
        return pipeline
    else:
        logger.error("Failed to create prediction pipeline")
        return None

# Example usage and testing
if __name__ == "__main__":
    logger.info("Churn Prediction Pipeline module loaded successfully!")
    
    # Example usage:
    # pipeline = create_prediction_pipeline()
    # if pipeline:
    #     # Single customer prediction
    #     customer = {
    #         'CustomerID': 'CUST12345',
    #         'Gender': 'Male',
    #         'Age': 35,
    #         'Tenure': 5,
    #         'Balance': 120000,
    #         'NumOfProducts': 2,
    #         'HasCrCard': 1,
    #         'IsActiveMember': 1,
    #         'EstimatedSalary': 80000
    #     }
    #     print(f"Prediction: {result['prediction_label']}")
    #     print(f"Churn Probability: {result['churn_probability']:.2%}")
