import pandas as pd
import os

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
        if source_type.lower() == 'csv':
            return pd.read_csv(source_path, **kwargs)
        elif source_type.lower() == 'json':
            return pd.read_json(source_path, **kwargs)
        elif source_type.lower() == 'excel':
            return pd.read_excel(source_path, **kwargs)
        elif source_type.lower() == 'sql':
            # For SQL, source_path should be the query and 'con' should be in kwargs
            return pd.read_sql(source_path, **kwargs)
        elif source_type.lower() == 'parquet':
            return pd.read_parquet(source_path, **kwargs)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
    except Exception as e:
        print(f"Error loading data from {source_path}: {str(e)}")
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
    
    # Check if cleaned data already exists
    if skip_if_exists and os.path.exists(output_path):
        print(f"Cleaned data already exists at {output_path}. Skipping preprocessing.")
        return pd.read_csv(output_path)
    
    print("Starting data preprocessing...")
    
    # Load the dataset using the flexible data loader
    df = load_data_from_source(input_path, source_type, **kwargs)
    if df is None:
        print(f"Error: Could not load data from {input_path}")
        return None
    
    print(f"Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    
    # Remove CustomerID column (not needed for analysis)
    if 'CustomerID' in df.columns:
        df.drop(columns=['CustomerID'], inplace=True)
        print("CustomerID column removed")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    print(f"\nMissing values before cleaning:")
    print(missing_values[missing_values > 0])
    
    # Fill null values based on data type
    # Categorical columns: fill with mode
    categorical_cols = ['Gender', 'Churn', 'HasCrCard', 'IsActiveMember']
    for col in categorical_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # Numerical columns: fill with mean
    numerical_cols = ['Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    for col in numerical_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mean())
    
    # Check for null values after filling
    null_values = df.isnull().sum()
    print(f"\nNull values after filling: {null_values.sum()}")
    
    # Fix object columns
    print("\nFixing object columns...")
    
    # Fix Gender column - use title case
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].str.strip().str.title()
    
    # Fix HasCrCard column - convert to 0 or 1
    if 'HasCrCard' in df.columns:
        df['HasCrCard'] = df['HasCrCard'].apply(
            lambda x: 1 if pd.notna(x) and str(x).isdigit() and float(x) > 0 else 0
        )
    
    # Fix IsActiveMember column - convert to 0 or 1
    if 'IsActiveMember' in df.columns:
        df['IsActiveMember'] = df['IsActiveMember'].apply(
            lambda x: 1 if pd.notna(x) and str(x).isdigit() and float(x) > 0 else 0
        )
    
    # Fix Churn column - map values and convert to numeric
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({
            '0.0': 0, '1.0': 1, '2.0': 2, '2': 2, 'Maybe': 2,
            0.0: 0, 1.0: 1, 2.0: 2, 0: 0, 1: 1, 2: 2
        })
    
    # Check Churn distribution
    print(f"\nChurn distribution before removing class 2:")
    print(df['Churn'].value_counts())
    
    # Remove rows with Churn = 2 (very few samples)
    df = df[df['Churn'] != 2]
    
    print(f"\nChurn distribution after removing class 2:")
    print(df['Churn'].value_counts())
    
    # Print final data info
    print(f"\nFinal dataset shape: {df.shape}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the cleaned dataset
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    
    return df

if __name__ == "__main__":
    print("Libraries imported successfully!")
    df = preprocess_data()
    if df is not None:
        print("\nPreprocessing completed successfully!")
        print(f"Final dataset info:")
        print(df.info())