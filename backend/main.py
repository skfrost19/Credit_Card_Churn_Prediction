from flask import Flask, request, jsonify
import pandas as pd
import sys
import os
from datetime import datetime
import traceback

# Add the parent directory to the path to import from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline import create_prediction_pipeline
from core.logger import get_logger

# Initialize Flask app
app = Flask(__name__)

# Initialize logger
logger = get_logger(__name__)

# Global prediction pipeline
prediction_pipeline = None

def initialize_pipeline():
    """Initialize the prediction pipeline on startup."""
    global prediction_pipeline
    try:
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        prediction_pipeline = create_prediction_pipeline(models_dir=models_dir)
        if prediction_pipeline:
            logger.info("Prediction pipeline initialized successfully")
            return True
        else:
            logger.error("Failed to initialize prediction pipeline")
            return False
    except Exception as e:
        logger.error(f"Error initializing pipeline: {str(e)}")
        return False

@app.route('/', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    
    Returns:
        JSON response with API status and timestamp
    """
    try:
        status = {
            'status': 'healthy',
            'service': 'Credit Card Churn Prediction API',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0',
            'model_loaded': prediction_pipeline is not None
        }
        
        logger.info("Health check requested")
        return jsonify(status), 200
        
    except Exception as e:
        logger.error(f"Error in health check: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Health check failed',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/predict', methods=['POST'])
def predict_churn():
    """
    Predict churn for a customer.
    
    Expected JSON payload:
    {
        "CustomerID": "string",
        "Gender": "Male" or "Female",
        "Age": integer (18-100),
        "Tenure": integer (0-20),
        "Balance": float (>= 0),
        "NumOfProducts": integer (1-10),
        "HasCrCard": integer (0 or 1),
        "IsActiveMember": integer (0 or 1),
        "EstimatedSalary": float (>= 0)
    }
    
    Returns:
        JSON response with prediction results
    """
    try:
        # Check if pipeline is loaded
        if prediction_pipeline is None:
            logger.error("Prediction pipeline not loaded")
            return jsonify({
                'error': 'Prediction service not available',
                'message': 'Model not loaded'
            }), 503
        
        # Get JSON data from request
        if not request.is_json:
            return jsonify({
                'error': 'Invalid request format',
                'message': 'Request must be JSON'
            }), 400
        
        data = request.get_json()
        
        # Validate required fields
        required_fields = [
            'CustomerID', 'Gender', 'Age', 'Tenure', 'Balance', 
            'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'error': 'Missing required fields',
                'missing_fields': missing_fields,
                'required_fields': required_fields
            }), 400
        
        # Convert to DataFrame
        customer_df = pd.DataFrame([data])
        
        logger.info(f"Prediction requested for customer: {data.get('CustomerID', 'Unknown')}")
        
        # Make prediction
        results = prediction_pipeline.predict(customer_df, return_probabilities=True)
        
        # Extract results for single customer
        customer_result = {
            'customer_id': results['customer_ids'][0],
            'churn_prediction': int(results['predictions'][0]),
            'churn_prediction_label': results['prediction_labels'][0],
            'confidence': {
                'churn_probability': round(results['churn_probability'][0], 4),
                'no_churn_probability': round(results['no_churn_probability'][0], 4)
            },
            'prediction_confidence': round(max(results['churn_probability'][0], 
                                              results['no_churn_probability'][0]), 4),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Prediction completed for customer {customer_result['customer_id']}: "
                   f"{customer_result['churn_prediction_label']} "
                   f"(confidence: {customer_result['prediction_confidence']:.2%})")
        
        return jsonify({
            'success': True,
            'result': customer_result
        }), 200
        
    except ValueError as ve:
        logger.error(f"Validation error in prediction: {str(ve)}")
        return jsonify({
            'error': 'Validation error',
            'message': str(ve)
        }), 400
        
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'error': 'Prediction failed',
            'message': 'Internal server error occurred'
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist'
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({
        'error': 'Method not allowed',
        'message': 'The requested method is not allowed for this endpoint'
    }), 405

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500

# Initialize pipeline when the module is imported
if __name__ == '__main__':
    # Initialize the prediction pipeline
    logger.info("Starting Credit Card Churn Prediction API...")
    
    if initialize_pipeline():
        logger.info("API startup successful - ready to serve predictions")
        
        # Run the Flask app
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            use_reloader=False  # Disable reloader to prevent pipeline reinitialization
        )
    else:
        logger.error("API startup failed - pipeline initialization failed")
        sys.exit(1)
else:
    # Initialize pipeline when imported as module
    initialize_pipeline()
