import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_model(model_path):
    return load(model_path)

def encode_data(dataframe):
    if dataframe.dtype == "object":
        dataframe = LabelEncoder().fit_transform(dataframe)
    return dataframe

def preprocess_data(data):
    # Dropping unnecessary columns
    data = data.drop(['customerID'], axis=1)
    
    # Converting 'TotalCharges' to numeric and handling missing values
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data.fillna(data['TotalCharges'].mean(), inplace=True)
    
    # Encoding categorical variables
    data['SeniorCitizen'] = data['SeniorCitizen'].map({0: "No", 1: "Yes"})
    data = data.apply(lambda x: encode_data(x))
    
    # Splitting data into features and target
    X = data.drop(columns='Churn')
    y = data['Churn'].values
    
    # Splitting data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4, stratify=y)
    
    # Standard scaling of features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def preprocess_and_predict(model_path, testing_data):
    X_test_scaled, _ = preprocess_data(testing_data)
    loaded_model = load_model(model_path)
    predictions = loaded_model.predict(X_test_scaled)

    return predictions

def evaluate_model(model_path, testing_data):
    X_test, y_test = preprocess_data(testing_data)

    predictions = preprocess_and_predict(model_path, testing_data)

    error = mean_squared_error(y_test, predictions)

    print(f'Mean Squared Error: {error}')

if __name__ == "__main__":
    data = pd.read_csv('testing.csv')
    model_path = 'final_model.pkl'
    evaluate_model(model_path, data)