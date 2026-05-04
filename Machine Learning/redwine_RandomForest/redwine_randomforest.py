import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the dataset
df = pd.read_csv('winequality-red.csv')

# Inspect the 'quality' column distribution to determine the breakpoint
print("Value Counts for 'quality' column:")
print(df['quality'].value_counts().sort_index())

# Display general info and head of the data
print("\nDataFrame Info:")
print(df.info())
print("\nDataFrame Head:")
print(df.head())

# Inspect the 'quality' column to determine the breakpoint
print("Quality value counts (original data):")
print(df['quality'].value_counts().sort_index())

# Check for non-numeric columns that might need scaling or encoding
print("\nDataFrame Info:")
print(df.info())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Breakpoint chosen: Quality >= 7 is High Quality (1), otherwise Low Quality (0)
df['target'] = np.where(df['quality'] >= 7, 1, 0)
print(f"Number of High Quality wines (1): {df['target'].sum()}")
print(f"Number of Low Quality wines (0): {len(df) - df['target'].sum()}")

# Define features (X) and target (y)
X = df.drop(['quality', 'target'], axis=1)
y = df['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Scale features 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# Initialize and Train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy:.4f}")

print("\nConfusion Matrix:")
print(conf_matrix)

# Print a formatted version for the user
print("\nConfusion Matrix Interpretation:")
print(pd.DataFrame(conf_matrix, 
                   index=['Actual Low Quality (0)', 'Actual High Quality (1)'], 
                   columns=['Predicted Low Quality (0)', 'Predicted High Quality (1)']))