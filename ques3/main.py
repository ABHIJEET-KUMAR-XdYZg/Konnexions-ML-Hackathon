import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
import joblib 
def load_model(model_path):
    loaded_model = joblib.load(model_path)
    return loaded_model

def preprocess_data(data):
    count = 0
    for sentance in tqdm(data['tweet'].values):
        sentance = re.sub(r"http\S+", "", sentance)
        sentance = decontracted(sentance)
        sentance = re.sub("\S*\d\S*", "", sentance).strip()
        sentance = re.sub('[^A-Za-z]+', ' ', sentance)
        sentance = ' '.join(ps.stem(e.lower()) for e in sentance.split() if e.lower() not in stopwords)
        sentance.replace("#", "").replace("_", " ")
        data["tweet"][count] = sentance.strip()
        count += 1

    X = vectorizer.transform(data["tweet"].values).toarray()
    return X

def preprocess_and_predict(model_path, testing_data):
    loaded_model = load_model(model_path)

    X_test = preprocess_data(testing_data)

    predictions = loaded_model.predict(X_test)

    return predictions

def evaluate_model(model_path, testing_data):
    X_test = preprocess_data(testing_data) 

    loaded_model = load_model(model_path)

    predictions = loaded_model.predict(X_test)

    y_true = testing_data['disaster']

    f1 = f1_score(y_true, predictions)
    acc = accuracy_score(y_true, predictions)
    print(f'F1 Score: {f1}')
    print(f'Accuracy: {acc}')

    return f1, acc

if __name__ == "__main__":
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')

    model_path = 'disaster.pkl'

    print("Evaluation on training data:")
    evaluate_model(model_path, train_data)

    print("Making predictions on test data:")
    test_data['predicted'] = preprocess_and_predict(model_path, test_data)
    
    test_data.to_csv('predicted.csv', index=False)