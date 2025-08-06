# Credit Card Churn Prediction

A machine learning web application that predicts credit card customer churn using Random Forest algorithm with Flask backend and Streamlit frontend.

## Overview

This project implements an end-to-end ML pipeline for predicting customer churn in credit card services. The system processes customer data, applies feature engineering, and provides real-time predictions through a web interface.

## Architecture

- **ML Pipeline**: Modular preprocessing, feature engineering, and model training
- **Backend API**: Flask REST API with health check and prediction endpoints
- **Frontend UI**: Streamlit web interface for user interaction
- **Logging**: Comprehensive logging across all components

## Quick Start

### Prerequisites

- Python 3.9+
- uv package manager

### Installation

```bash
# Clone the repository
git clone <https://github.com/skfrost19/Credit_Card_Churn_Prediction>
cd Credit_Card_Churn_Prediction

# Install dependencies using uv
uv sync

# If uv is not installed, visit https://docs.astral.sh/uv/getting-started/installation/
```

### Running the Application

**Step 1: Start the Backend**
```bash
cd backend
uv run main.py
```

**Step 2: Start the Frontend**
```bash
# In a new terminal
uv run streamlit run streamlit_ui.py
```

### Access Points

- Frontend UI: http://localhost:8501
- Backend API: http://localhost:5000
- API Health: http://localhost:5000/

## API Usage

### Health Check
```bash
GET /
```

### Prediction
```bash
POST /predict
Content-Type: application/json

{
  "CustomerID": "CUST_001",
  "Gender": "Male",
  "Age": 35,
  "Tenure": 5,
  "Balance": 120000.0,
  "NumOfProducts": 2,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 80000.0
}
```

### Response
```json
{
  "status": "success",
  "result": {
    "churn_prediction": 0,
    "prediction_confidence": 0.85,
    "confidence": {
      "churn_probability": 0.15,
      "no_churn_probability": 0.85
    }
  }
}
```

## Project Structure

```
Final_Project/
├── src/                      # ML pipeline modules
│   ├── data_preprocessing.py # Data cleaning and validation
│   ├── eda.py               # Exploratory data analysis
│   ├── feature_engineering.py # Feature transformation
│   ├── model_trainer.py     # Model training and evaluation
│   └── pipeline.py          # Prediction pipeline
├── backend/                  # Flask API
│   ├── main.py              # API server
│   └── test_api.py          # API tests
├── models/                   # Trained model artifacts
├── data/                     # Dataset files
├── logs/                     # Application logs
├── streamlit_ui.py          # Frontend interface
├── run_app.py              # Application launcher
├── test_pipeline.py        # Pipeline tests
├── pyproject.toml          # Project configuration
└── uv.lock                 # Dependency lock file
```

## Features

### Machine Learning
- Random Forest classifier
- Feature engineering and scaling
- Cross-validation and hyperparameter tuning
- Model persistence and loading

### Backend API
- RESTful endpoints
- Input validation
- Error handling
- Request logging

### Frontend Interface
- Interactive form inputs
- Real-time predictions
- Probability visualizations
- API status monitoring

## Model Performance

The Random Forest model achieves:
- High accuracy on validation data
- Balanced precision and recall
- Feature importance analysis
- Confidence scoring for predictions

## Testing

```bash
# Test ML pipeline
uv run test_pipeline.py

# Test API endpoints
uv run backend/test_api.py
```

## Configuration

### Environment Variables
- `API_PORT`: Backend port (default: 5000)
- `UI_PORT`: Frontend port (default: 8501)

### Model Files
Required model artifacts in `models/` directory:
- `random_forest_churn_model.pkl`
- `encoding_info.pkl`
- `all_scalers.pkl`

## Dependencies

This project uses `uv` for fast and reliable dependency management.

Key packages:
- pandas, numpy: Data manipulation
- scikit-learn: Machine learning
- flask: Backend API
- streamlit: Frontend UI
- plotly: Visualizations
- joblib: Model persistence

See `pyproject.toml` and `uv.lock` for complete dependency specifications.

## Troubleshooting

### Common Issues

1. **API Connection Error**: Ensure backend is running on port 5000
2. **Model Loading Error**: Verify model files exist in `models/` directory
3. **Port Conflicts**: Check if ports 5000/8501 are available

### Logs

Check application logs in `logs/` directory for detailed error information.

## Credits

- **Logger Module**: The `core/logger.py` implementation is adapted from [readme-ai](https://github.com/eli64s/readme-ai) project.

## License

This project is part of the EXL Training Capstone Project.