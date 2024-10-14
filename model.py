import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
df = pd.read_csv('Bankruptcy Prediction.csv')

# Inspect the column names
print("Columns in DataFrame:", df.columns.tolist())

# Strip spaces from column names
df.columns = df.columns.str.strip()

# Select the top 10 important features
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

# Check if all features are in the DataFrame
missing_features = [feature for feature in important_features if feature not in df.columns]
if missing_features:
    print("Missing features:", missing_features)
    raise KeyError("Some features are missing from the DataFrame.")

# Prepare the features and target variable
X = df[important_features]
y = df['Bankrupt?']  # Assuming 'Bankrupt?' is the target column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the LightGBM model
model = lgb.LGBMClassifier()
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model accuracy:", accuracy)

# Save the trained model using pickle
with open('bankruptcy_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved successfully.")
