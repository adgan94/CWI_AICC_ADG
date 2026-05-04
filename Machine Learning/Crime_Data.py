import pandas as pd
import os
import numpy as np
'''import folium
from PIL import Image
import io
import time'''
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
'''df = pd.read_parquet('Crime_Data_from_2020_to_Present.parquet')

print(df.head())
print(df.info())



def convert_parquet_to_csv(parquet_filepath, csv_filepath):
    try:
        print(f"Reading Parquet file: {parquet_filepath}...")
        df = pd.read_parquet(parquet_filepath)
        
        print(f"Writing to CSV file: {csv_filepath}...")
        df.to_csv(csv_filepath, index=False)
        
        print(f"Conversion successful! File saved at {csv_filepath}")
        
    except FileNotFoundError:
        print(f"Error: The file {parquet_filepath} was not found.")
    except Exception as e:
        print(f"An error occurred during conversion: {e}")

input_file = 'Crime_Data_from_2020_to_Present.parquet'
output_file = 'Crime_Data_from_2020_to_Present.csv'

convert_parquet_to_csv(input_file, output_file)'''

df = pd.read_csv('Crime_Data_from_2020_to_Present.csv')
def time_to_minutes(time_obj):
    if isinstance(time_obj, str):
        try:
            # Handle string format if it exists, assuming HH:MM:SS or HH:MM
            parts = time_obj.split(':')
            hours = int(parts[0])
            minutes = int(parts[1])
            return hours * 60 + minutes
        except ValueError:
            return np.nan # Return NaN if string format is invalid
    elif hasattr(time_obj, 'hour') and hasattr(time_obj, 'minute'):
        return time_obj.hour * 60 + time_obj.minute
    return np.nan # Handle any other non-time format or None values

df['TIME OCC'] = df['TIME OCC'].apply(time_to_minutes)

# Fill any remaining NaN values (from conversion errors or original NaNs) with 0
df['TIME OCC'] = df['TIME OCC'].fillna(0).astype(int)

categorical_cols = [
    'Vict Sex',
    'Vict Descent',
    'Crm Cd Desc',
    'Premis Desc',
    'Weapon Desc',
    'Status Desc',
    'LOCATION',
    'Mocodes'
]

for col in categorical_cols:
    # Strip whitespace and convert to uppercase for standardization
    df[col] = df[col].astype(str).str.strip().str.upper()
    if col == 'Vict Descent':
        df[col] = df[col].replace(['NOT SPECIFIED'], 'UNKNOWN')
    df[col] = df[col].replace({'NAN': 'UNKNOWN', 'NOT SPECIFIED': 'UNKNOWN'}).fillna('UNKNOWN')


'''# Calculate the center of the map based on the mean latitude and longitude
center_lat = df['LAT'].mean()
center_lon = df['LON'].mean()

m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

for idx, row in df.head(500).iterrows():
    if pd.notnull(row['LAT']) and pd.notnull(row['LON']):
        folium.Marker(
            location=[row['LAT'], row['LON']],
            popup=f"Crime: {row['Crm Cd Desc']}",
            tooltip=row['Crm Cd Desc']
        ).add_to(m)

map_html_path = 'crime_geomap.html'
m.save(map_html_path)'''
df = pd.read_csv('Crime_Data_from_2020_to_Present.csv')
categorical_cols_initial = [
    'Vict Sex',
    'Vict Descent',
    'Crm Cd Desc',
    'Premis Desc',
    'Weapon Desc',
    'Status Desc'
]

for col in categorical_cols_initial:
    df[col] = df[col].astype(str).str.strip().str.upper()
    if col == 'Vict Descent':
        df[col] = df[col].replace(['NOT SPECIFIED'], 'UNKNOWN')
    df[col] = df[col].replace({'NAN': 'UNKNOWN', 'NOT SPECIFIED': 'UNKNOWN'}).fillna('UNKNOWN')

# Clean LOCATION and Mocodes
df['LOCATION'] = df['LOCATION'].astype(str).str.strip().str.upper()
df['LOCATION'] = df['LOCATION'].replace({'NAN': 'UNKNOWN'}).fillna('UNKNOWN')

df['Mocodes'] = df['Mocodes'].astype(str).str.strip().str.upper()
df['Mocodes'] = df['Mocodes'].replace({'NAN': 'UNKNOWN'}).fillna('UNKNOWN')

y = df['Crm Cd Desc']
X = df.drop('Crm Cd Desc', axis=1)

# Columns to be eliminated from the feature set X
columns_to_eliminate = [
    'DR_NO', 'Date Rptd', 'DATE OCC', 'Mocodes', 'LOCATION', 'Rpt Dist No',
    'Part 1-2', 'Crm Cd', 'Premis Cd', 'Weapon Used Cd', 'Status'
]

X = X.drop(columns=columns_to_eliminate, errors='ignore')

# Identify numerical and categorical features
numerical_features = [
    'TIME OCC',
    'AREA',
    'Vict Age',
    'LAT',
    'LON',
    'occ_year',
    'occ_date'
]
categorical_features = [
    'Vict Sex',
    'Vict Descent',
    'Premis Desc',
    'Weapon Desc',
    'Status Desc',
    'occ_month',
    'occ_day',
    'AREA NAME'
]

# Handle potential missing values in numerical features before scaling
for col in numerical_features:
    if X[col].isnull().any():
        X[col] = X[col].fillna(X[col].median()) # Filling with median for robustness

# One-hot encode categorical features
one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_categorical_encoded = one_hot_encoder.fit_transform(X[categorical_features])

# Create a DataFrame from the one-hot encoded features with proper column names
categorical_column_names = one_hot_encoder.get_feature_names_out(categorical_features)
X_categorical_df = pd.DataFrame(X_categorical_encoded, columns=categorical_column_names, index=X.index)

# Scale numerical features
scaler = StandardScaler()
X_numerical_scaled = scaler.fit_transform(X[numerical_features])

# Create a DataFrame from the scaled numerical features with original column names
X_numerical_df = pd.DataFrame(X_numerical_scaled, columns=numerical_features, index=X.index)

# Concatenate processed features
X_processed = pd.concat([X_numerical_df, X_categorical_df], axis=1)

# One-hot encode the target variable
y_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
y_encoded = y_encoder.fit_transform(y.to_frame())

# Create a DataFrame for the encoded target with proper column names
y_column_names = y_encoder.get_feature_names_out(['Crm Cd Desc'])
y_encoded_df = pd.DataFrame(y_encoded, columns=y_column_names, index=y.index)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encoded_df, test_size=0.2, random_state=42)


# --- Neural Network Model Construction, Training, and Evaluation ---
# 1. Define the model
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)), # Input layer
    Dropout(0.3),
    Dense(128, activation='relu'), # Hidden layer 1
    Dropout(0.3),
    Dense(64, activation='relu'),  # Hidden layer 2
    Dropout(0.3),
    Dense(y_train.shape[1], activation='softmax') # Output layer
])

# 2. Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display model summary
model.summary()

# 3. Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=10, # Number of training epochs
    batch_size=256, # Number of samples per gradient update
    validation_split=0.2, # Fraction of the training data to be used as validation data
    verbose=1
)

# 4. Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")