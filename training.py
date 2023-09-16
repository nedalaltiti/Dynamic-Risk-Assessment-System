import os
import sys
import json
import logging
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

class ModelTrainer:
    def __init__(self, config_path):
        self.config_path = config_path
        self.dataset_csv_path = None
        self.model_path = None
        self.model = None

    def load_config(self):
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        self.dataset_csv_path = os.path.join(config['output_folder_path'], 'finaldata.csv')
        self.model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl')

    def segregate_dataset(self, dataset):
        X = pd.read_csv(dataset).iloc[:, 1:-1].values.reshape(-1, 3)
        y = pd.read_csv(dataset)['exited'].values.reshape(-1, 1).ravel()
        return X, y

    def train_model(self):
        logging.info('Training the Logistic Regression Model started.')
        X, y = self.segregate_dataset(self.dataset_csv_path)

        if os.path.exists(self.model_path):
            logging.info("Model already exists. Skipping saving the model.")
        else:
            self.model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                                            intercept_scaling=1, l1_ratio=None, max_iter=100,
                                            multi_class='auto', n_jobs=None, penalty='l2',
                                            random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                                            warm_start=False)
            self.model.fit(X, y)

            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            logging.info("Saving trained model")
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)

if __name__ == '__main__':
    trainer = ModelTrainer('config.json')
    trainer.load_config()
    trainer.train_model()