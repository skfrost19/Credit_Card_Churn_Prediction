"""
Streamlit UI for Credit Card Churn Prediction

This module provides a user-friendly web interface for the credit card churn prediction API.
Users can input customer data and get real-time churn predictions with confidence scores.
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Credit Card Churn Prediction",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:5000"

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .success-card {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
    }
    .warning-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffc107;
    }
    .error-card {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

def check_api_health():
    """Check if the API is healthy and available."""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, {"error": "API not responding correctly"}
    except Exception as e:
        return False, {"error": str(e)}

def make_prediction(customer_data):
    """Make a prediction using the API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=customer_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json()
            
    except Exception as e:
        return False, {"error": str(e)}

def create_gauge_chart(probability, title):
    """Create a gauge chart for probability visualization."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "lightgray"},
                {'range': [25, 50], 'color': "gray"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">💳 Credit Card Churn Prediction</h1>', unsafe_allow_html=True)
    
    # Sidebar for API status and controls
    with st.sidebar:
        st.header("🔧 System Status")
        
        # API Health Check
        if st.button("🔄 Check API Status", type="primary"):
            with st.spinner("Checking API status..."):
                is_healthy, health_data = check_api_health()
                
            if is_healthy:
                st.success("✅ API is healthy!")
                st.json(health_data)
            else:
                st.error("❌ API is not available!")
                st.json(health_data)
        
        st.divider()
        
        st.header("ℹ️ About")
        st.info(
            "This application predicts whether a credit card customer "
            "is likely to churn based on their profile and behavior data."
        )
        
        st.header("📊 Model Info")
        st.markdown("""
        - **Algorithm**: Random Forest
        - **Features**: 9 customer attributes
        - **Output**: Churn probability & prediction
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Customer Information")
        
        # Customer input form
        with st.form("customer_form"):
            # Customer ID
            customer_id = st.text_input(
                "Customer ID",
                value="CUST_" + str(int(time.time())),
                help="Unique identifier for the customer"
            )
            
            # Demographics
            st.subheader("👤 Demographics")
            col_demo1, col_demo2 = st.columns(2)
            
            with col_demo1:
                gender = st.selectbox(
                    "Gender",
                    options=["Male", "Female"],
                    help="Customer's gender"
                )
                
                age = st.slider(
                    "Age",
                    min_value=18,
                    max_value=100,
                    value=35,
                    help="Customer's age in years"
                )
            
            with col_demo2:
                tenure = st.slider(
                    "Tenure (years)",
                    min_value=0,
                    max_value=20,
                    value=5,
                    help="Number of years with the bank"
                )
                
                estimated_salary = st.number_input(
                    "Estimated Salary",
                    min_value=0.0,
                    max_value=500000.0,
                    value=80000.0,
                    step=1000.0,
                    help="Annual estimated salary"
                )
            
            # Financial Information
            st.subheader("💰 Financial Information")
            col_fin1, col_fin2 = st.columns(2)
            
            with col_fin1:
                balance = st.number_input(
                    "Account Balance",
                    min_value=0.0,
                    max_value=1000000.0,
                    value=120000.0,
                    step=1000.0,
                    help="Current account balance"
                )
                
                num_products = st.selectbox(
                    "Number of Products",
                    options=list(range(1, 11)),
                    index=1,
                    help="Number of bank products used"
                )
            
            with col_fin2:
                has_cr_card = st.selectbox(
                    "Has Credit Card",
                    options=[("No", 0), ("Yes", 1)],
                    format_func=lambda x: x[0],
                    index=1,
                    help="Whether customer has a credit card"
                )[1]
                
                is_active_member = st.selectbox(
                    "Active Member",
                    options=[("No", 0), ("Yes", 1)],
                    format_func=lambda x: x[0],
                    index=1,
                    help="Whether customer is an active member"
                )[1]
            
            # Submit button
            submitted = st.form_submit_button(
                "🔮 Predict Churn",
                type="primary",
                use_container_width=True
            )
    
    with col2:
        st.header("📊 Prediction Results")
        
        if submitted:
            # Prepare customer data
            customer_data = {
                "CustomerID": customer_id,
                "Gender": gender,
                "Age": age,
                "Tenure": tenure,
                "Balance": balance,
                "NumOfProducts": num_products,
                "HasCrCard": has_cr_card,
                "IsActiveMember": is_active_member,
                "EstimatedSalary": estimated_salary
            }
            
            # Make prediction
            with st.spinner("Making prediction..."):
                success, result = make_prediction(customer_data)
            
            if success and 'result' in result:
                pred_result = result['result']
                
                # Main prediction result
                if pred_result['churn_prediction'] == 1:
                    st.markdown(
                        f'<div class="warning-card">'
                        f'<h3 style="color: black;">⚠️ CHURN RISK DETECTED</h3>'
                        f'<p style="color: black;">This customer is likely to churn.</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="success-card">'
                        f'<h3 style="color: black;">✅ LOW CHURN RISK</h3>'
                        f'<p style="color: black;">This customer is likely to stay.</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                # Metrics
                col_met1, col_met2, col_met3 = st.columns(3)
                
                with col_met1:
                    st.metric(
                        "Churn Probability",
                        f"{pred_result['confidence']['churn_probability']:.1%}",
                        delta=f"{pred_result['confidence']['churn_probability'] - 0.5:.1%}"
                    )
                
                with col_met2:
                    st.metric(
                        "Retention Probability",
                        f"{pred_result['confidence']['no_churn_probability']:.1%}",
                        delta=f"{pred_result['confidence']['no_churn_probability'] - 0.5:.1%}"
                    )
                
                with col_met3:
                    st.metric(
                        "Confidence",
                        f"{pred_result['prediction_confidence']:.1%}",
                        delta=f"{pred_result['prediction_confidence'] - 0.5:.1%}"
                    )
                
                # Gauge charts
                col_gauge1, col_gauge2 = st.columns(2)
                
                with col_gauge1:
                    churn_gauge = create_gauge_chart(
                        pred_result['confidence']['churn_probability'],
                        "Churn Risk"
                    )
                    st.plotly_chart(churn_gauge, use_container_width=True)
                
                with col_gauge2:
                    retention_gauge = create_gauge_chart(
                        pred_result['confidence']['no_churn_probability'],
                        "Retention Probability"
                    )
                    st.plotly_chart(retention_gauge, use_container_width=True)
                
                # Detailed results
                with st.expander("🔍 Detailed Results"):
                    st.json(pred_result)
                
                # Customer data summary
                with st.expander("📋 Customer Data Summary"):
                    # Create a properly formatted DataFrame with string representations
                    display_data = {k: [str(v)] for k, v in customer_data.items()}
                    df = pd.DataFrame(display_data)
                    df_transposed = df.T
                    df_transposed.columns = ['Value']
                    df_transposed.index.name = 'Field'
                    st.dataframe(df_transposed, use_container_width=True)
                
            else:
                st.markdown(
                    f'<div class="error-card">'
                    f'<h3>❌ Prediction Failed</h3>'
                    f'<p>Error: {result.get("message", "Unknown error")}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                with st.expander("🔍 Error Details"):
                    st.json(result)
        
        else:
            st.info("👈 Fill in the customer information and click 'Predict Churn' to get started!")
            
            # Sample data display
            st.subheader("📋 Sample Customer Data")
            sample_data = {
                "CustomerID": "SAMPLE_001",
                "Gender": "Male",
                "Age": 35,
                "Tenure": 5,
                "Balance": 120000.0,
                "NumOfProducts": 2,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 80000.0
            }
            
            # Create a properly formatted DataFrame with string representations
            display_sample_data = {k: [str(v)] for k, v in sample_data.items()}
            sample_df = pd.DataFrame(display_sample_data)
            sample_df_transposed = sample_df.T
            sample_df_transposed.columns = ['Value']
            sample_df_transposed.index.name = 'Field'
            st.dataframe(sample_df_transposed, use_container_width=True)
    
    # Footer
    st.divider()
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Credit Card Churn Prediction System | "
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
