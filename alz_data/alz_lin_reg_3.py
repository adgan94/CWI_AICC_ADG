import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Read the Alzheimer's prediction dataset
alzheimers_df = pd.read_csv('/home/aganley/CWI_AICC_ADG/processed_alzheimers_prediction_dataset.csv')

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
