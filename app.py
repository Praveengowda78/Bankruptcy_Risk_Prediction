import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open('lgbm_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to make predictions using the model
def predict(input_data):
    # Convert input data into DataFrame
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    return prediction[0]

# Streamlit App Title and Description
st.title("💼 Bankruptcy Risk Prediction")
st.markdown("""
    **Predict the likelihood of bankruptcy risk** based on financial metrics.
    Fill in the values for the following key financial indicators:
""")

# List of selected important features
important_features = [
    'Interest-bearing debt interest rate', 
    'Revenue per person', 
    'Inventory Turnover Rate (times)', 
    'Cash Turnover Rate', 
    'Allocation rate per person', 
    'Non-industry income and expenditure/revenue', 
    'Fixed Assets Turnover Frequency', 
    'Accounts Receivable Turnover', 
    'Research and development expense rate', 
    'Quick Ratio'
]

# Input fields for each important feature
input_data = {}
for feature in important_features:
    input_data[feature] = st.number_input(f"Enter {feature} value", min_value=0.0, step=0.1)

# Add some styling and user engagement
st.markdown("""
    <style>
    .css-1cpxqw2, .css-15tx938 {
        font-family: 'Courier New', Courier, monospace;
        font-size: 18px;
        background-color: #f4f4f4;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Prediction button
if st.button('Predict'):
    prediction = predict(input_data)
    
    # Display result with formatting
    if prediction == 1:
        st.markdown("<h2 style='color: red;'>⚠️ The model predicts a high risk of bankruptcy!</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='color: green;'>✅ The model predicts no risk of bankruptcy.</h2>", unsafe_allow_html=True)

'''# Footer for better engagement
st.markdown("""
    ---
    *Created by Praveen - Machine Learning Enthusiast*  
    For inquiries or more information, contact: [Praveen's Contact](mailto:praveen@example.com)
""")'''
