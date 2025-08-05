import pandas as pd
import os
from core.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

def detect_outliers_iqr(df, column):
    """
    Detect outliers using the Interquartile Range (IQR) method.
    
    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Column name to check for outliers
    
    Returns:
        pd.DataFrame: Rows containing outliers in the specified column
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

def load_data_from_source(source_path, source_type='csv', **kwargs):
    """
    Load data from various sources (CSV, database, etc.).
    
    Args:
        source_path (str): Path to the data source
        source_type (str): Type of data source ('csv', 'sql', 'json', etc.)
        **kwargs: Additional arguments for specific data loading functions
    
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    try:
        logger.info(f"Loading data from {source_path} with type {source_type}")
        
        if source_type.lower() == 'csv':
            df = pd.read_csv(source_path, **kwargs)
        elif source_type.lower() == 'json':
            df = pd.read_json(source_path, **kwargs)
        elif source_type.lower() == 'excel':
            df = pd.read_excel(source_path, **kwargs)
        elif source_type.lower() == 'sql':
            # For SQL, source_path should be the query and 'con' should be in kwargs
            df = pd.read_sql(source_path, **kwargs)
        elif source_type.lower() == 'parquet':
            df = pd.read_parquet(source_path, **kwargs)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
        
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading data from {source_path}: {str(e)}")
        return None

def preprocess_data(input_path=None, output_path=None, skip_if_exists=True, source_type='csv', **kwargs):
    """
    Preprocess the credit card churn data.
    
    Args:
        input_path (str): Path to the input data source. If None, uses default path.
        output_path (str): Path to save the cleaned CSV file. If None, uses default path.
        skip_if_exists (bool): If True, skips preprocessing if cleaned data already exists.
        source_type (str): Type of data source ('csv', 'sql', 'json', etc.)
        **kwargs: Additional arguments for data loading functions
    
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    
    # Define default file paths if not provided
    if input_path is None:
        input_path = '../data/exl_credit_card_churn_data.csv'
    if output_path is None:
        output_path = '../data/exl_credit_card_churn_data_cleaned.csv'
    
    logger.info(f"Starting preprocessing with input: {input_path}, output: {output_path}")
    
    # Check if cleaned data already exists
    if skip_if_exists and os.path.exists(output_path):
        logger.info(f"Cleaned data already exists at {output_path}. Skipping preprocessing.")
        return pd.read_csv(output_path)
    
    logger.info("Starting data preprocessing...")
    
    # Load the dataset using the flexible data loader
    df = load_data_from_source(input_path, source_type, **kwargs)
    if df is None:
        logger.error(f"Could not load data from {input_path}")
        return None
    
    logger.info(f"Dataset loaded successfully! Shape: {df.shape}")
    
    # Remove CustomerID column (not needed for analysis)
    if 'CustomerID' in df.columns:
        df.drop(columns=['CustomerID'], inplace=True)
        logger.info("CustomerID column removed")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    missing_count = missing_values[missing_values > 0]
    if len(missing_count) > 0:
        logger.info(f"Missing values before cleaning:\n{missing_count}")
    else:
        logger.info("No missing values found")
    
    # Fill null values based on data type
    # Categorical columns: fill with mode
    categorical_cols = ['Gender', 'Churn', 'HasCrCard', 'IsActiveMember']
    for col in categorical_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            mode_value = df[col].mode()[0]
            df[col] = df[col].fillna(mode_value)
            logger.info(f"Filled {col} null values with mode: {mode_value}")
    
    # Numerical columns: fill with mean
    numerical_cols = ['Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    for col in numerical_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            mean_value = df[col].mean()
            df[col] = df[col].fillna(mean_value)
            logger.info(f"Filled {col} null values with mean: {mean_value:.2f}")
    
    # Check for null values after filling
    null_values = df.isnull().sum()
    logger.info(f"Null values after filling: {null_values.sum()}")
    
    # Fix object columns
    logger.info("Fixing object columns...")
    
    # Fix Gender column - use title case
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].str.strip().str.title()
        logger.info("Gender column fixed with title case")
    
    # Fix HasCrCard column - convert to 0 or 1
    if 'HasCrCard' in df.columns:
        df['HasCrCard'] = df['HasCrCard'].apply(
            lambda x: 1 if pd.notna(x) and str(x).isdigit() and float(x) > 0 else 0
        )
        logger.info("HasCrCard column converted to binary (0/1)")
    
    # Fix IsActiveMember column - convert to 0 or 1
    if 'IsActiveMember' in df.columns:
        df['IsActiveMember'] = df['IsActiveMember'].apply(
            lambda x: 1 if pd.notna(x) and str(x).isdigit() and float(x) > 0 else 0
        )
        logger.info("IsActiveMember column converted to binary (0/1)")
    
    # Fix Churn column - map values and convert to numeric
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({
            '0.0': 0, '1.0': 1, '2.0': 2, '2': 2, 'Maybe': 2,
            0.0: 0, 1.0: 1, 2.0: 2, 0: 0, 1: 1, 2: 2
        })
        logger.info("Churn column mapped to numeric values")
    
    # Check Churn distribution
    churn_dist_before = df['Churn'].value_counts()
    logger.info(f"Churn distribution before removing class 2:\n{churn_dist_before}")
    
    # Remove rows with Churn = 2 (very few samples)
    rows_before = len(df)
    df = df[df['Churn'] != 2]
    rows_removed = rows_before - len(df)
    logger.info(f"Removed {rows_removed} rows with Churn = 2")
    
    churn_dist_after = df['Churn'].value_counts()
    logger.info(f"Churn distribution after removing class 2:\n{churn_dist_after}")
    
    # Check using IQR for outliers in numerical columns
    logger.info("Detecting outliers using IQR method...")
    
    numerical_outlier_cols = ['Balance', 'EstimatedSalary', 'Age', 'Tenure']
    outliers_info = {}
    all_outlier_indices = set()
    
    for col in numerical_outlier_cols:
        if col in df.columns:
            outliers = detect_outliers_iqr(df, col)
            outliers_info[col] = outliers
            all_outlier_indices.update(outliers.index)
            logger.info(f"Outliers in {col}: {len(outliers)} rows")
            if len(outliers) > 0:
                logger.debug(f"Outlier indices for {col}: {list(outliers.index)}")
    
    logger.info(f"Total number of unique outlier indices: {len(all_outlier_indices)}")
    
    # Log detailed outlier information
    for col, outliers in outliers_info.items():
        if len(outliers) > 0:
            logger.info(f"Outliers in {col}:\n{outliers[[col]].describe()}")
    
    # Remove outliers
    rows_before_outlier_removal = len(df)
    df = df.drop(index=all_outlier_indices)
    rows_removed_outliers = rows_before_outlier_removal - len(df)
    logger.info(f"Removed {rows_removed_outliers} outlier rows")
    
    # Print final data info
    logger.info(f"Final dataset shape after outlier removal: {df.shape}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the cleaned dataset
    df.to_csv(output_path, index=False)
    logger.info(f"Cleaned data saved to {output_path}")
    
    return df

if __name__ == "__main__":
    logger.info("Libraries imported successfully!")
    df = preprocess_data()
    if df is not None:
        logger.info("Preprocessing completed successfully!")
        logger.info(f"Final dataset info:")
        print(df.info())
    else:
        logger.error("Preprocessing failed!")