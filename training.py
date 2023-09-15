"""
Training the Logistic Regression Model in training data.

Name: Needal Altiti
Date: 14 / 09 / 2023
"""
import os
import sys
import json
import logging
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

# dataset_csv_path = os.path.join(config['output_folder_path']) 
dataset_csv_path = os.path.join(config['output_folder_path'], 'finaldata.csv')
model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl')

logging.info('Segregating the data.')
def segregate_dataset(dataset):
    """
    Eliminate features not used
    segregate the dataset into X and y
    input: dataset to segregate
    output: X and y
    """
    X = pd.read_csv(dataset).iloc[:, 1:-1].values.reshape(-1, 3)
    y = pd.read_csv(dataset)['exited'].values.reshape(-1, 1).ravel()

    return X,y

# Function for training the model
def train_model():
    logging.info('training the Logistic Regression Model started.')
    X, y = segregate_dataset(dataset_csv_path)
    #use this logistic regression for training
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    #write the trained model to your workspace in a file called trainedmodel.pkl
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    # Check if the model file already exists
    if os.path.exists(model_path):
        logging.info("Model already exists. Skipping saving the model.")
    else:
        # Fit the logistic regression to your data
        model.fit(X, y)

        # Write the trained model to your workspace in a file called trainedmodel.pkl
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        logging.info("Saving trained model")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

if __name__ == '__main__':
    train_model()        