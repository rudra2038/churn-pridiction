import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
# Replace with your actual data path or method
df = pd.read_csv('customer_data.csv')
# Preview the data
print(df.head())
# Example: Convert categorical columns to dummy variables
df = pd.get_dummies(df, drop_first=True)
# Optional: Check for nulls and fill or drop as needed
df = df.dropna()  # Or use df.fillna(method='ffill'), etc.

# Define features and target
X = df.drop('Churn', axis=1)  # 'Churn' is assumed to be the target column
y = df['Churn']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Export results for Power BI
df['Predicted_Churn'] = model.predict(X_scaled)
df.to_csv('churn_predictions_output.csv', index=False)
