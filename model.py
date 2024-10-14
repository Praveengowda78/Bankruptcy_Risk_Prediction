import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('Bankruptcy Prediction.csv')

# Display first few rows to understand the data structure (optional)
print(data.head())

# Assuming 'Bankrupt?' is the target column
X = data.drop('Bankrupt?', axis=1)
y = data['Bankrupt?']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the LightGBM model
model = lgb.LGBMClassifier()
model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Feature importance analysis
feature_importances = model.feature_importances_
feature_names = X.columns

# Create a DataFrame to display feature importance
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Display the top features
print("Top features based on importance:")
print(importance_df.head(10))  # Adjust number of top features if needed

# Plot the feature importances for better visualization
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.gca().invert_yaxis()
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.show()

# Save the model using pickle
with open('lgbm_model.pkl', 'wb') as file:
    pickle.dump(model, file)
print("Model saved as 'lgbm_model.pkl'")
