"""
Credit Card Churn Prediction - Pipeline Runner

This script provides command-line interface for running different parts of the ML pipeline:
- eda: Generate comprehensive EDA report and visualizations
- train: Train the churn prediction model with feature engineering
- test_using_pipeline: Test the prediction pipeline with sample data

Usage:
    python first_run.py eda
    python first_run.py train
    python first_run.py test_using_pipeline
"""

import argparse
import os
import sys
import pandas as pd
from pathlib import Path

from src.eda import generate_comprehensive_eda_report, create_correlation_heatmap
from src.feature_engineering import feature_engineering_pipeline, load_transformers
from src.data_preprocessing import preprocess_data, load_data_from_source
from src.model_trainer import train_churn_model
from src.pipeline import create_prediction_pipeline


def check_required_artifacts():
    """Check if all required model artifacts are present for testing."""
    required_files = [
        'models/random_forest_churn_model.pkl',
        'models/encoding_info.pkl',
        'models/all_scalers.pkl'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("Missing required model artifacts:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease run 'python first_run.py train' first to generate the required artifacts.")
        return False
    
    print("All required model artifacts found.")
    return True


def run_eda():
    """Run exploratory data analysis."""
    print("Starting Exploratory Data Analysis...")
    
    # Load preprocessed data or preprocess if needed
    if os.path.exists('data/temp.csv'):
        print("Loading existing preprocessed data...")
        df = pd.read_csv('data/temp.csv')
    else:
        print("Preprocessing data for EDA...")
        df = preprocess_data(
            input_path='data/exl_credit_card_churn_data.csv',
            output_path='data/temp.csv',
            skip_if_exists=False
        )
    
    if df is not None:
        print(f"Dataset loaded successfully! Shape: {df.shape}")
        
        # Generate EDA report
        print("Generating comprehensive EDA report...")
        generate_comprehensive_eda_report(df)
        
        # Create correlation heatmap
        print("Creating correlation heatmap...")
        create_correlation_heatmap(df)
        
        print("EDA completed successfully!")
        print("Check the 'visualisations/' directory for generated plots.")
    else:
        print("Failed to load data for EDA.")
        sys.exit(1)


def run_training():
    """Run the complete training pipeline."""
    print("Starting Model Training Pipeline...")
    
    # Preprocess data
    print("Step 1: Data Preprocessing...")
    df = preprocess_data(
        input_path='data/exl_credit_card_churn_data.csv',
        output_path='data/temp.csv',
        skip_if_exists=False
    )
    
    if df is None:
        print("Data preprocessing failed!")
        sys.exit(1)
    
    print(f"Preprocessing successful! Dataset shape: {df.shape}")
    
    # Feature engineering
    print("Step 2: Feature Engineering...")
    df_engineered = feature_engineering_pipeline(df)
    print("Feature engineering completed!")
    
    # Model training
    print("Step 3: Model Training...")
    train_churn_model(None, df_engineered)
    print("Model training completed!")
    
    print("Training pipeline completed successfully!")
    print("Model artifacts saved in 'models/' directory.")


def run_test_pipeline():
    """Test the prediction pipeline with sample data."""
    print("Testing Prediction Pipeline...")
    
    # Check if required artifacts exist
    if not check_required_artifacts():
        sys.exit(1)
    
    # Check if test data exists
    test_data_path = 'data/test_customers.csv'
    if not os.path.exists(test_data_path):
        print(f"Test data file not found: {test_data_path}")
        print("Please ensure test data is available before running tests.")
        sys.exit(1)
    
    try:
        # Load test data
        print(f"Loading test data from {test_data_path}...")
        test_data = pd.read_csv(test_data_path)
        print(f"Test data loaded successfully! Shape: {test_data.shape}")
        
        # Create prediction pipeline
        print("Initializing prediction pipeline...")
        pipeline = create_prediction_pipeline()
        print("Pipeline initialized successfully!")
        
        # Make predictions
        print("Making predictions...")
        results = pipeline.predict(test_data)
        
        if results is not None and len(results) > 0:
            print("Predictions completed successfully!")
            print(f"Processed {len(results)} test samples")
            
            # Display sample results
            print("\nSample prediction results:")
            for i, result in enumerate(results[:3]):  # Show first 3 results
                print(f"   Sample {i+1}: Churn={result['churn_prediction']}, "
                      f"Confidence={result['prediction_confidence']:.2%}")
            
            if len(results) > 3:
                print(f"   ... and {len(results)-3} more results")
                
        else:
            print("No predictions generated!")
            
    except Exception as e:
        print(f"Error during pipeline testing: {str(e)}")
        sys.exit(1)


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Credit Card Churn Prediction Pipeline Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python first_run.py eda                    # Run exploratory data analysis
  python first_run.py train                  # Train the churn prediction model
  python first_run.py test_using_pipeline    # Test prediction pipeline
        """
    )
    
    parser.add_argument(
        'command',
        choices=['eda', 'train', 'test_using_pipeline'],
        help='Pipeline command to execute'
    )
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('visualisations', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    print(f"Executing command: {args.command}")
    print("=" * 50)
    
    # Execute the requested command
    if args.command == 'eda':
        run_eda()
    elif args.command == 'train':
        run_training()
    elif args.command == 'test_using_pipeline':
        run_test_pipeline()


if __name__ == "__main__":
    main()
