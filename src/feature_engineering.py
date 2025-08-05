"""
Feature Engineering, Encoding, and Scaling module for Credit Card Churn Prediction.
This module handles feature creation, categorical encoding, and numerical scaling
while saving all transformers for later use in prediction.
"""

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
from core.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

def create_engineered_features(df):
    """
    Create new features from existing columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        pd.DataFrame: Dataframe with new engineered features
    """
    logger.info("Starting feature engineering...")
    
    # Create a copy to avoid modifying original data
    df_feat_eng = df.copy()
    
    # Create age bins
    logger.info("Creating age categories...")
    df_feat_eng['AgeCategory'] = pd.cut(
        df_feat_eng['Age'], 
        bins=[0, 25, 35, 50, 100], 
        labels=['Young', 'Adult', 'Middle', 'Senior']
    )
    age_dist = df_feat_eng['AgeCategory'].value_counts()
    logger.info(f"Age category distribution:\n{age_dist}")
    
    # Create balance categories
    logger.info("Creating balance categories...")
    df_feat_eng['BalanceCategory'] = pd.cut(
        df_feat_eng['Balance'], 
        bins=[-1, 0, 50000, 100000, np.inf], 
        labels=['Zero', 'Low', 'Medium', 'High']
    )
    balance_dist = df_feat_eng['BalanceCategory'].value_counts()
    logger.info(f"Balance category distribution:\n{balance_dist}")
    
    # Create balance per product
    logger.info("Creating balance per product feature...")
    df_feat_eng['BalancePerProduct'] = df_feat_eng['Balance'] / (df_feat_eng['NumOfProducts'] + 1)
    logger.info(f"BalancePerProduct stats:\n{df_feat_eng['BalancePerProduct'].describe()}")
    
    # Create tenure-age ratio (Loyalty relative to life stage)
    logger.info("Creating tenure-age ratio feature...")
    df_feat_eng['TenureAgeRatio'] = df_feat_eng['Tenure'] / (df_feat_eng['Age'] + 1)
    logger.info(f"TenureAgeRatio stats:\n{df_feat_eng['TenureAgeRatio'].describe()}")
    
    logger.info(f"Feature engineering completed. New shape: {df_feat_eng.shape}")
    return df_feat_eng

def apply_one_hot_encoding(df, categorical_columns=None, models_dir='models'):
    """
    Apply one-hot encoding to categorical columns and save the encoding information.
    
    Args:
        df (pd.DataFrame): Input dataframe
        categorical_columns (list): List of categorical columns to encode
        models_dir (str): Directory to save encoding models
    
    Returns:
        tuple: (encoded_dataframe, encoding_info_dict)
    """
    if categorical_columns is None:
        categorical_columns = ['Gender', 'AgeCategory', 'BalanceCategory']
    
    logger.info(f"Applying one-hot encoding to columns: {categorical_columns}")
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Store original columns before encoding
    original_columns = df.columns.tolist()
    
    # Apply one-hot encoding
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    
    # Get the new columns created by encoding
    new_columns = [col for col in df_encoded.columns if col not in original_columns]
    
    # Store encoding information
    encoding_info = {
        'original_columns': categorical_columns,
        'encoded_columns': new_columns,
        'all_columns_after_encoding': df_encoded.columns.tolist(),
        'drop_first': True
    }
    
    # Save encoding information
    encoding_file = os.path.join(models_dir, 'encoding_info.pkl')
    with open(encoding_file, 'wb') as f:
        pickle.dump(encoding_info, f)
    logger.info(f"Encoding information saved to {encoding_file}")
    
    # Log encoding results
    logger.info(f"One-hot encoding completed:")
    for orig_col in categorical_columns:
        related_cols = [col for col in new_columns if col.startswith(orig_col)]
        logger.info(f"  {orig_col} -> {related_cols}")
    
    logger.info(f"Shape after encoding: {df_encoded.shape}")
    return df_encoded, encoding_info

def apply_scaling(df, columns_to_scale=None, models_dir='models'):
    """
    Apply MinMax scaling to numerical columns and save scalers.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns_to_scale (list): List of columns to scale
        models_dir (str): Directory to save scaler models
    
    Returns:
        tuple: (scaled_dataframe, scalers_dict)
    """
    if columns_to_scale is None:
        columns_to_scale = [
            'Balance', 'EstimatedSalary', 'Age', 'Tenure', 
            'NumOfProducts', 'BalancePerProduct', 'TenureAgeRatio'
        ]
    
    logger.info(f"Applying MinMax scaling to columns: {columns_to_scale}")
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Create a copy to avoid modifying original data
    df_scaled = df.copy()
    scalers = {}
    
    # Apply scaling to each column
    for column in columns_to_scale:
        if column in df_scaled.columns:
            logger.info(f"Scaling column: {column}")
            
            # Create and fit scaler
            scaler = MinMaxScaler()
            df_scaled[column] = scaler.fit_transform(df_scaled[[column]])
            
            # Store scaler
            scalers[column] = scaler
            
            # Save individual scaler
            scaler_file = os.path.join(models_dir, f'scaler_{column.lower()}.pkl')
            with open(scaler_file, 'wb') as f:
                pickle.dump(scaler, f)
            logger.info(f"Scaler for {column} saved to {scaler_file}")
            
            # Log scaling statistics
            logger.info(f"  Original {column} range: [{df[column].min():.2f}, {df[column].max():.2f}]")
            logger.info(f"  Scaled {column} range: [{df_scaled[column].min():.2f}, {df_scaled[column].max():.2f}]")
        else:
            logger.warning(f"Column {column} not found in dataframe. Skipping.")
    
    # Save all scalers together
    scalers_file = os.path.join(models_dir, 'all_scalers.pkl')
    with open(scalers_file, 'wb') as f:
        pickle.dump(scalers, f)
    logger.info(f"All scalers saved to {scalers_file}")
    
    logger.info(f"Scaling completed for {len(scalers)} columns")
    return df_scaled, scalers

def prepare_target_variable(df, target_column='Churn'):
    """
    Prepare the target variable for modeling.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_column (str): Name of target column
    
    Returns:
        pd.DataFrame: Dataframe with prepared target variable
    """
    logger.info(f"Preparing target variable: {target_column}")
    
    df_target = df.copy()
    
    if target_column in df_target.columns:
        # Convert to categorical
        df_target[target_column] = df_target[target_column].astype('category')
        
        # Log target distribution
        target_dist = df_target[target_column].value_counts()
        logger.info(f"Target variable distribution:\n{target_dist}")
        
        # Calculate class balance
        class_percentages = df_target[target_column].value_counts(normalize=True) * 100
        logger.info(f"Target variable percentages:\n{class_percentages}")
        
    else:
        logger.error(f"Target column {target_column} not found in dataframe")
    
    return df_target

def feature_engineering_pipeline(df, output_path=None, models_dir='models'):
    """
    Complete feature engineering pipeline including feature creation, encoding, and scaling.
    
    Args:
        df (pd.DataFrame): Input dataframe
        output_path (str): Path to save the processed dataframe
        models_dir (str): Directory to save all models and transformers
    
    Returns:
        pd.DataFrame: Fully processed dataframe ready for modeling
    """
    logger.info("Starting complete feature engineering pipeline...")
    logger.info(f"Input dataframe shape: {df.shape}")
    
    # Step 1: Create engineered features
    df_features = create_engineered_features(df)
    
    # Step 2: Apply one-hot encoding
    df_encoded, encoding_info = apply_one_hot_encoding(df_features, models_dir=models_dir)
    
    # Step 3: Prepare target variable
    df_target = prepare_target_variable(df_encoded)
    
    # Step 4: Apply scaling
    df_final, scalers = apply_scaling(df_target, models_dir=models_dir)
    
    # Save the final processed dataframe
    if output_path:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_final.to_csv(output_path, index=False)
        logger.info(f"Processed dataframe saved to {output_path}")
    
    # Save processing metadata
    metadata = {
        'original_shape': df.shape,
        'final_shape': df_final.shape,
        'features_added': ['AgeCategory', 'BalanceCategory', 'BalancePerProduct', 'TenureAgeRatio'],
        'encoded_columns': encoding_info['original_columns'],
        'scaled_columns': list(scalers.keys()),
        'models_directory': models_dir
    }
    
    metadata_file = os.path.join(models_dir, 'processing_metadata.pkl')
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    logger.info(f"Processing metadata saved to {metadata_file}")
    
    logger.info(f"Feature engineering pipeline completed!")
    logger.info(f"Final dataframe shape: {df_final.shape}")
    logger.info(f"All models and transformers saved in: {models_dir}")
    
    return df_final

def load_transformers(models_dir='models'):
    """
    Load all saved transformers for use in prediction.
    
    Args:
        models_dir (str): Directory containing saved models
    
    Returns:
        dict: Dictionary containing all loaded transformers
    """
    logger.info(f"Loading transformers from {models_dir}")
    
    transformers = {}
    
    try:
        # Load encoding info
        encoding_file = os.path.join(models_dir, 'encoding_info.pkl')
        if os.path.exists(encoding_file):
            with open(encoding_file, 'rb') as f:
                transformers['encoding_info'] = pickle.load(f)
            logger.info("Encoding info loaded successfully")
        
        # Load all scalers
        scalers_file = os.path.join(models_dir, 'all_scalers.pkl')
        if os.path.exists(scalers_file):
            with open(scalers_file, 'rb') as f:
                transformers['scalers'] = pickle.load(f)
            logger.info(f"Scalers loaded successfully: {list(transformers['scalers'].keys())}")
        
        # Load metadata
        metadata_file = os.path.join(models_dir, 'processing_metadata.pkl')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'rb') as f:
                transformers['metadata'] = pickle.load(f)
            logger.info("Processing metadata loaded successfully")
        
        logger.info("All transformers loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading transformers: {str(e)}")
    
    return transformers

if __name__ == "__main__":
    logger.info("Feature engineering module loaded successfully!")
    
    # Example usage
    df = pd.read_csv('data/cleaned_data.csv')
    processed_df = feature_engineering_pipeline(
        df, 
        output_path='data/processed_data.csv',
        models_dir='models'
    )
