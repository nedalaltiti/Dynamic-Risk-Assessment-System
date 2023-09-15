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



# Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'], 'ingestedfiles.txt') 
trained_model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl')
metrics_path = os.path.join(config['output_model_path'], 'latestscore.txt')
prod_deployment_path = os.path.join(config['prod_deployment_path']) 

# unction for deployment
def store_model_into_pickle():
    """
    copy the latest pickle file, the latestscore.txt value, 
    and the ingestfiles.txt file into the deployment directory
    """
    logging.info("Deploying trained model to production")
    os.makedirs(prod_deployment_path, exist_ok=True)

    # Copy trained model
    shutil.copy(trained_model_path, prod_deployment_path)

    # Copy latestscore.txt
    shutil.copy(metrics_path, prod_deployment_path)

    # Copy ingestedfiles.txt
    shutil.copy(dataset_csv_path, prod_deployment_path)

    logging.info("Model, score, and ingested files copied to the deployment directory.")  


if __name__ == '__main__':
    store_model_into_pickle()
        
        
        

