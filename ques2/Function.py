import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score

# Function to load the custom dataset
def load_custom_dataset(dataset_path):
    custom_df = pd.read_csv(dataset_path)  # Assuming the dataset is in CSV format
    # Preprocess your custom dataset if needed (e.g., normalization)
    # Make sure the features in the custom dataset match the ones used during training
    return custom_df

# Function to perform prediction using the saved model
def predict_with_saved_model(model_path, custom_dataset):
    # Load the saved model
    model = tf.keras.models.load_model(model_path)
    
    # Assuming the custom dataset contains the same features as used during training
    custom_data = custom_dataset.drop(['engine', 'cycle'], axis=1)
    custom_data = np.array(custom_data)  # Convert to numpy array
    
    # Perform predictions
    predictions = model.predict(custom_data)
    
    return predictions

# Function to calculate RMSE and R^2 score
def evaluate_predictions(true_values, predicted_values):
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    r2 = r2_score(true_values, predicted_values)
    return rmse, r2

# Path to the saved model
model_path = 'path/to/saved_model.h5'

# Path to the custom dataset
custom_dataset_path = 'path/to/custom_dataset.csv'

# Load the custom dataset
custom_dataset = load_custom_dataset(custom_dataset_path)

# Perform prediction using the saved model
predictions = predict_with_saved_model(model_path, custom_dataset)

# Assuming the true RUL values are available in the custom dataset
true_values = custom_dataset['RUL']  # Make sure 'RUL' is the column name containing true RUL values

# Evaluate the predictions
rmse, r2 = evaluate_predictions(true_values, predictions)

print("RMSE:", rmse)
print("R^2 Score (Accuracy):", r2)
