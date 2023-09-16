import os
import sys
import json
import logging
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    dataset_csv_path = os.path.join(config['output_folder_path'], 'finaldata.csv')
    model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl')
    return dataset_csv_path, model_path

def segregate_dataset(dataset):
    X = pd.read_csv(dataset).iloc[:, 1:-1].values.reshape(-1, 3)
    y = pd.read_csv(dataset)['exited'].values.reshape(-1, 1).ravel()
    return X, y


def train_model(dataset_csv_path, model_path):
    logging.info('Training the Logistic Regression Model started.')
    X, y = segregate_dataset(dataset_csv_path)

    if os.path.exists(model_path):
        logging.info("Model already exists. Skipping saving the model.")
    else:
        model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                                    multi_class='auto', n_jobs=None, penalty='l2',
                                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                                    warm_start=False)
        model.fit(X, y)

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        logging.info("Saving trained model")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

if __name__ == '__main__':
    config_path = 'config.json'
    dataset_csv_path, model_path = load_config(config_path)
    train_model(dataset_csv_path, model_path)

