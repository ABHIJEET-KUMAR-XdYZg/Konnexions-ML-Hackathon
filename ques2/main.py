import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn.metrics

def load_model(model_path, example_input):
    # Load the saved model
    model = tf.keras.models.load_model(model_path)
    
    # Provide an example input to infer the input shape for the normalization layer
    model.layers[0].adapt(example_input)

    return model

def preprocess_data(data):
    # Split each line in the text data and convert to DataFrame
    data = pd.DataFrame([line.split() for line in data.split('\n') if line.strip()])
    
    # Assign appropriate column names
    column_names = ['engine', 'cycle', 'setting_1', 'setting_2', 'setting_3',
                    'Sensor_1', 'Sensor_2', 'Sensor_3', 'Sensor_4', 'Sensor_5',
                    'Sensor_6', 'Sensor_7', 'Sensor_8', 'Sensor_9', 'Sensor_10',
                    'Sensor_11', 'Sensor_12', 'Sensor_13', 'Sensor_14', 'Sensor_15',
                    'Sensor_16', 'Sensor_17', 'Sensor_18', 'Sensor_19', 'Sensor_20',
                    'Sensor_21']
    data.columns = column_names

    # Drop unnecessary columns
    data.drop(columns=['Sensor_18','Sensor_19', "setting_3", 'Sensor_5','Sensor_1','setting_3','Sensor_16','Sensor_10'], inplace=True)
    data.drop(columns=['setting_1','setting_2','Sensor_14'], inplace=True)  # Additional drop based on your code
    
    return data

def preprocess_and_predict(model_path, testing_data):
    # Preprocess the testing data
    testing_data = preprocess_data(testing_data)
    
    # Prepare the testing dataset for prediction
    test_x = prepare_test_data(testing_data)
    
    # Load the pre-trained model
    model = load_model(model_path, test_x[0])  # Provide an example input
    
    # Make predictions
    predictions = model.predict(test_x)
    
    return predictions

def evaluate_model(model_path, testing_data, test_rul_path):
    # Preprocess the testing data
    testing_data = preprocess_data(testing_data)
    
    # Prepare the testing dataset for evaluation
    test_x = prepare_test_data(testing_data)
    
    # Load the pre-trained model
    model = load_model(model_path, test_x[0])  # Provide an example input
    
    # Make predictions
    predictions = model.predict(test_x)
    
    # Load the ground truth RUL values
    test_rul = pd.read_csv(test_rul_path, sep=" ", header=None)
    test_rul = np.array(test_rul.drop([1], axis=1)).squeeze()
    
    # Evaluate the model
    r2_score = sklearn.metrics.r2_score(test_rul, predictions)
    rmse = float(tf.keras.metrics.RootMeanSquaredError()(test_rul, predictions))
    
    return r2_score, rmse


def prepare_test_data(df):
    # Assuming the data is already preprocessed and only features are present
    return df

# Example usage
if __name__ == "__main__":
    model_path = 'JET_RUL.h5'
    test_data_path = 'test.txt'
    test_rul_path = 'RUL_FD001.txt'
    
    # Load the testing dataset
    with open(test_data_path, 'r') as file:
        test_data = file.read()
    
    # Evaluate the model
    r2_score, rmse = evaluate_model(model_path, test_data, test_rul_path)
    
    print(f"R^2 Score: {r2_score}")
    print(f"RMSE: {rmse}")
