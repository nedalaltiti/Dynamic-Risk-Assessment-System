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
from training import ModelTrainer


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class ModelEvaluation:
    def __init__(self, config_path):
        self.config_path = config_path

    def load_config(self):
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        return config

    def load_model(self, model_path):
        """
        Load the trained model.
        """
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model

    def load_test_data(self, test_data_path):
        """
        Load the test data.
        """
        trainer = ModelTrainer(self.config_path)
        trainer.load_config()
        X, y = trainer.segregate_dataset(test_data_path)
        return X, y

    def score_model(self):
        """
        Calculate the F1 score for the model relative to the test data
        and write the result to the latestscore.txt file.
        """
        logging.info("Evaluating the Logistic Regression Model.")
        config = self.load_config()
        test_data_path = os.path.join(config['test_data_path'], 'testdata.csv')
        trained_model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl')
        metrics_path = os.path.join(config['output_model_path'], 'latestscore.txt')

        X, y = self.load_test_data(test_data_path)
        model = self.load_model(trained_model_path)
        pred = model.predict(X)
        f1score = metrics.f1_score(pred, y)
        logging.info(f'F1 Score: {f1score}')
        with open(metrics_path, 'w') as f:
            f.write(str(f1score))
        logging.info(f"Saving F1 score in {metrics_path}")
        return f1score


if __name__ == '__main__':
    config_path = 'config.json'
    evaluation = ModelEvaluation(config_path)
    evaluation.score_model()