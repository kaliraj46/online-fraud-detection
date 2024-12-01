# Importing the necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
# For demonstration purposes, we assume you have a CSV file 'creditcard.csv' containing the data
# dataset = pd.read_csv('creditcard.csv')
# For this example, we'll load the dataset from sklearn.datasets (for simplicity)

from sklearn.datasets import fetch_openml
dataset = fetch_openml(name='creditcard', version=1)

# Check the dataset information
df = dataset.frame
print(df.head())

# Step 2: Preprocessing
# Normalize the data (scaling features for better performance)
scaler = StandardScaler()
X = df.drop('class', axis=1)  # Features (excluding the 'class' column)
y = df['class']  # Target variable (fraudulent or not)

# Scaling the features
X_scaled = scaler.fit_transform(X)

# Step 3: Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 4: Train a Random Forest Classifier (you can choose other models like XGBoost or Logistic Regression)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
# Confusion Matrix
print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Classification Report (Precision, Recall, F1-score)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Precision and Recall
print(f"Precision: {precision_score(y_test, y_pred)}")
print(f"Recall: {recall_score(y_test, y_pred)}")

# Step 7: Visualize the Confusion Matrix using a heatmap
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Legitimate", "Fraudulent"], yticklabels=["Legitimate", "Fraudulent"])
plt.title("Confusion Matrix")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

