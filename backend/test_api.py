"""
Test script for the Flask backend API
Demonstrates how to interact with the churn prediction API
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:5000"

def test_health_check():
    """Test the health check endpoint."""
    print("Testing health check endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error testing health check: {e}")
        return False

def test_prediction():
    """Test the prediction endpoint."""
    print("\nTesting prediction endpoint...")
    
    # Sample customer data
    customer_data = {
        "CustomerID": "TEST_CUSTOMER_001",
        "Gender": "Male",
        "Age": 35,
        "Tenure": 5,
        "Balance": 120000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 80000.0
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=customer_data,
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
        
    except Exception as e:
        print(f"Error testing prediction: {e}")
        return False

def test_invalid_request():
    """Test the prediction endpoint with invalid data."""
    print("\nTesting prediction endpoint with invalid data...")
    
    # Missing required fields
    invalid_data = {
        "CustomerID": "INVALID_CUSTOMER",
        "Gender": "Male",
        "Age": 35
        # Missing other required fields
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=invalid_data,
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 400
        
    except Exception as e:
        print(f"Error testing invalid request: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING CREDIT CARD CHURN PREDICTION API")
    print("=" * 60)
    print("Make sure the Flask server is running on http://localhost:5000")
    print("Run: python backend/main.py")
    print("-" * 60)
    
    # Run tests
    health_ok = test_health_check()
    prediction_ok = test_prediction()
    invalid_ok = test_invalid_request()
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Health Check: {'✓ PASS' if health_ok else '✗ FAIL'}")
    print(f"Valid Prediction: {'✓ PASS' if prediction_ok else '✗ FAIL'}")
    print(f"Invalid Request Handling: {'✓ PASS' if invalid_ok else '✗ FAIL'}")
    
    if all([health_ok, prediction_ok, invalid_ok]):
        print("\nAll tests passed! API is working correctly.")
    else:
        print("\nSome tests failed. Check the server logs.")
