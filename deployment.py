"""
deploying the Logistic Regression Model in training data.

Name: Needal Altiti
Date: 14 / 09 / 2023
"""
import shutil
import os
import json
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class ModelDeployment:
    def __init__(self, config):
        self.config = config
        self.dataset_csv_path = os.path.join(config['output_folder_path'], 'ingestedfiles.txt') 
        self.trained_model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl')
        self.metrics_path = os.path.join(config['output_model_path'], 'latestscore.txt')
        self.prod_deployment_path = os.path.join(config['prod_deployment_path']) 

    def store_model_into_pickle(self):
        """
        Copy the latest pickle file, the latestscore.txt value, 
        and the ingestfiles.txt file into the deployment directory
        """
        logging.info("Deploying trained model to production")
        os.makedirs(self.prod_deployment_path, exist_ok=True)

        # Copy trained model
        shutil.copy(self.trained_model_path, self.prod_deployment_path)

        # Copy latestscore.txt
        shutil.copy(self.metrics_path, self.prod_deployment_path)

        # Copy ingestedfiles.txt
        shutil.copy(self.dataset_csv_path, self.prod_deployment_path)

        logging.info("Model, score, and ingested files copied to the deployment directory.")  


if __name__ == '__main__':
    # Load config.json and correct path variable
    with open('config.json','r') as f:
        config = json.load(f) 

    deployment = ModelDeployment(config)
    deployment.store_model_into_pickle()


        

