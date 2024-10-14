import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('bankruptcy_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to make predictions
def predict(input_data):
    input_df = pd.DataFrame(input_data, index=[0])
    prediction = model.predict(input_df)
    return prediction[0]

# Streamlit app layout
st.title("Bankruptcy Prediction App")
st.markdown("This app predicts the likelihood of bankruptcy based on input features.")

# User input fields
interest_bearing_debt_interest_rate = st.number_input("Interest-bearing debt interest rate", min_value=0.0, step=0.01)
revenue_per_person = st.number_input("Revenue per person", min_value=0.0, step=0.01)
inventory_turnover_rate = st.number_input("Inventory Turnover Rate (times)", min_value=0.0, step=0.01)
cash_turnover_rate = st.number_input("Cash Turnover Rate", min_value=0.0, step=0.01)
allocation_rate_per_person = st.number_input("Allocation rate per person", min_value=0.0, step=0.01)
non_industry_income_expenditure = st.number_input("Non-industry income and expenditure/revenue", min_value=0.0, step=0.01)
fixed_assets_turnover_frequency = st.number_input("Fixed Assets Turnover Frequency", min_value=0.0, step=0.01)
accounts_receivable_turnover = st.number_input("Accounts Receivable Turnover", min_value=0.0, step=0.01)
research_development_expense_rate = st.number_input("Research and development expense rate", min_value=0.0, step=0.01)
quick_ratio = st.number_input("Quick Ratio", min_value=0.0, step=0.01)

# Prepare the input data
input_data = {
    'Interest-bearing debt interest rate': interest_bearing_debt_interest_rate,
    'Revenue per person': revenue_per_person,
    'Inventory Turnover Rate (times)': inventory_turnover_rate,
    'Cash Turnover Rate': cash_turnover_rate,
    'Allocation rate per person': allocation_rate_per_person,
    'Non-industry income and expenditure/revenue': non_industry_income_expenditure,
    'Fixed Assets Turnover Frequency': fixed_assets_turnover_frequency,
    'Accounts Receivable Turnover': accounts_receivable_turnover,
    'Research and development expense rate': research_development_expense_rate,
    'Quick Ratio': quick_ratio
}

# Predict and display the result
if st.button("Predict"):
    prediction = predict(input_data)
    st.success(f"The predicted bankruptcy status is: {'Bankrupt' if prediction == 1 else 'Not Bankrupt'}")
