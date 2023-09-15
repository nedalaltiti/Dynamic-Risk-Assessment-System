"""
Evaluating the Logistic Regression Model.

Name: Needal Altiti
Date: 14 / 09 / 2023
"""
import pickle
import os
import json
import logging 
import sys 
from sklearn import metrics
from sklearn.metrics import f1_score
from training import segregate_dataset


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


# Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = os.path.join(config['test_data_path'], 'testdata.csv') 
trained_model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl')
metrics_path = os.path.join(config['output_model_path'], 'latestscore.txt')


def load_model(model_path):
    """
    Load the trained model.
    """
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def load_test_data(test_data_path):
    """
    Load the test data.
    """
    X,y = segregate_dataset(test_data_path)
    return X, y

# Function for model scoring
def score_model():
    """
    this function should take a trained model, load test data, 
    and calculate an F1 score for the model relative to the test data
    it should write the result to the latestscore.txt file
    """
    logging.info("Evaluating the Logistic Regression Model.")
    X,y = load_test_data(test_data_path)
    model = load_model(trained_model_path)
    pred = model.predict(X)
    f1score = metrics.f1_score(pred, y)  
    logging.info(f'F1 Score: {f1score}')   
    with open(metrics_path, 'w') as f:
        f.write(str(f1score))
    logging.info(f"Saving F1 score in {metrics_path}")  
    return f1score  
    


if __name__ == '__main__':
    score_model()
