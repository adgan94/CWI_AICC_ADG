import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics

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

# Apply the mappings to the dataset
for column, mapping in mappings.items():
    alzheimers_df[column] = alzheimers_df[column].map(mapping)

# Print the column names to check for discrepancies
print(alzheimers_df.columns)

# Drop the Country column
df_sans_cc = alzheimers_df.drop(columns=['Country'])

'''Read the country codes
country_codes_df = pd.read_csv('CountryCodes.csv')

# Ensure the Country column is of the same type in both DataFrames
alzheimers_df['Country'] = alzheimers_df['Country'].astype(str)
country_codes_df['Country'] = country_codes_df['Country'].astype(str)

# Merge the datasets on the Country column
merged_df = pd.merge(alzheimers_df, country_codes_df[['Country', 'Code']], on='Country', how='left')

# Replace the Country column with the associated codes
merged_df['Country'] = merged_df['Code']
merged_df = merged_df.drop(columns=['Code'])'''

# Write the processed data to a new file
df_sans_cc.to_csv('processed_alzheimers_prediction_dataset.csv', index=False)

print("The data has been processed and saved to processed_alzheimers_prediction_dataset.csv")

# Read the processed Alzheimer's prediction dataset
df = pd.read_csv('processed_alzheimers_prediction_dataset.csv')

# Define the features (X) and target (Y)
X = df.drop(['Alzheimers Diagnosis'], axis=1)
Y = df['Alzheimers Diagnosis']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

# Normalize the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train the Linear Regression model
lin_reg = LinearRegression()
regressor = lin_reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = regressor.predict(X_test)

# Evaluate the model
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', rmse)