import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Define the mappings for each column
mappings = {
    'Gender': {'Male': 0, 'Female': 1},
    'Physical Activity Level': {'Low': 0, 'Medium': 1, 'High': 2},
    'Smoking Status': {'Never': 0, 'Former': 1, 'Current': 2},
    'Alcohol Consumption': {'Never': 0, 'Occasionally': 1, 'Regularly': 2},
    'Diabetes': {'Yes': 1, 'No': 0},
    'Hypertension': {'Yes': 1, 'No': 0},
    'Cholesterol Level': {'Normal': 0, 'High': 1},
    "Family History of Alzheimer's": {'Yes': 1, 'No': 0},
    'Depression Level': {'Low': 0, 'Medium': 1, 'High': 2},
    'Sleep Quality': {'Poor': 0, 'Average': 1, 'Good': 2},
    'Dietary Habits': {'Unhealthy': 0, 'Average': 1, 'Healthy': 2},
    'Air Pollution Exposure': {'Low': 0, 'Medium': 1, 'High': 2},
    'Employment Status': {'Retired': 0, 'Unemployed': 1, 'Employed': 2},
    'Marital Status': {'Single': 0, 'Married': 1, 'Widowed': 2},
    'Genetic Risk Factor': {'Yes': 1, 'No': 0},
    'Social Engagement Level': {'Low': 0, 'Medium': 1, 'High': 2},
    'Income Level': {'Low': 0, 'Medium': 1, 'High': 2},
    'Stress Levels': {'Low': 0, 'Medium': 1, 'High': 2},
    'Urban vs Rural Living': {'Urban': 0, 'Rural': 1},
    "Alzheimer's Diagnosis": {'Yes': 1, 'No': 0}
}

# Read the Alzheimer's prediction dataset
alzheimers_df = pd.read_csv('http://172.22.50.3/datasets/alzheimers_prediction_dataset.csv')

# Print the column names to check for discrepancies
print("Columns in the dataset:", alzheimers_df.columns)

# Apply the mappings to the dataset
for column, mapping in mappings.items():
    if column in alzheimers_df.columns:
        alzheimers_df[column] = alzheimers_df[column].map(mapping)
    else:
        print(f"Column '{column}' not found in the Alzheimer's dataset")

# Write the processed data to a new file
alzheimers_df.to_csv('processed_alzheimers_prediction_dataset.csv', index=False)

print("The data has been processed and saved to processed_alzheimers_prediction_dataset.csv")

# Read the country codes from GitHub without headers
country_codes_url = 'https://raw.githubusercontent.com/adgan94/CWI_AICC_ADG/refs/heads/main/CountryCodes.csv'
country_codes_df = pd.read_csv(country_codes_url, header=None)

# Manually assign column names
country_codes_df.columns = ['Country', 'Code']

# Print the column names to check for discrepancies
print("Columns in the CountryCodes dataset:", country_codes_df.columns)

# Ensure the Country column is of the same type in both DataFrames
alzheimers_df['Country'] = alzheimers_df['Country'].astype(str)
country_codes_df['Country'] = country_codes_df['Country'].astype(str)

# Merge the datasets on the Country column
merged_df = pd.merge(alzheimers_df, country_codes_df[['Country', 'Code']], on='Country', how='left')

# Replace the Country column with the associated codes
merged_df['Country'] = merged_df['Code']
merged_df = merged_df.drop(columns=['Code'])

# Write the merged data to a new file
merged_df.to_csv('processed_alzheimers_prediction_dataset_with_codes.csv', index=False)

print("The data has been processed and saved to processed_alzheimers_prediction_dataset_with_codes.csv")

# Read the processed Alzheimer's prediction dataset
df = pd.read_csv('processed_alzheimers_prediction_dataset_with_codes.csv')

# Drop the Country column
df = df.drop(columns=['Country'])

# Define the features (X) and target (Y)
X = df.drop(['Alzheimer’s Diagnosis'], axis=1)
Y = df['Alzheimer’s Diagnosis']

# Check for any remaining non-numeric columns
print("Non-numeric columns:", X.select_dtypes(include=['object']).columns)

# Ensure all columns are numeric
X = X.apply(pd.to_numeric, errors='coerce')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Initialize the kNN classifier with k=5 (you can change this value)
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the model
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Plotting the confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()