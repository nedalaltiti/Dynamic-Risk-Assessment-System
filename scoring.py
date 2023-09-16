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
from training import segregate_dataset


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


with open('config.json', 'r') as f:
    config = json.load(f)

metrics_path = os.path.join(config['output_model_path'], 'latestscore.txt')    

def load_model(model_path):
    """
    Load the trained model.
    """
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model


def load_test_data(data_path):
    """
    Load the test data.
    """

    X, y = segregate_dataset(data_path)
    return X, y


def score_model(data_path, model_path):
    """
    Calculate the F1 score for the model relative to the test data
    and write the result to the latestscore.txt file.
    """
    logging.info("Evaluating the Logistic Regression Model.")

    X, y = load_test_data(data_path)
    model = load_model(model_path)
    pred = model.predict(X)
    f1score = metrics.f1_score(pred, y)
    logging.info(f'F1 Score: {f1score}')
    with open(metrics_path, 'w') as f:
        f.write(str(f1score))
    logging.info(f"Saving F1 score in {metrics_path}")
    return f1score


if __name__ == '__main__':
    test_data_path = os.path.join(config['test_data_path'], 'testdata.csv')
    trained_model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl')
    score_model(test_data_path, trained_model_path)