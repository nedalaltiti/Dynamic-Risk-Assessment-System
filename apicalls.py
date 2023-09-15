"""
This script calls each of the API endpoints, combine the outputs, and write the combined outputs to a file called
apireturns.txt.

Name: Nedal Altiti
Date: 16 / 09 /2023
"""
import requests
import json
import os
import logging

logger = logging.getLogger(__name__)


# #Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

# # Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path']) 

response_pred = requests.post(
    f'{URL}/prediction',
    json={
        'filepath': os.path.join(test_data_path, 'testdata.csv')}).text

logger.info("Get request /scoring")
response_score = requests.get(f'{URL}/scoring').text

logger.info("Get request /summarystats")
response_stats = requests.get(f'{URL}/summarystats').text

logger.info("Get request /diagnostics")
response_diag = requests.get(f'{URL}/diagnostics').text

logger.info("Generating report text file")
with open(os.path.join(model_path, 'apireturns.txt'), 'w') as file:
    file.write('Ingested Data\n\n')
    file.write('Statistics Summary\n')
    file.write(response_stats)
    file.write('\nDiagnostics Summary\n')
    file.write(response_diag)
    file.write('\n\nTest Data\n\n')
    file.write('Model Predictions\n')
    file.write(response_pred)
    file.write('\nModel Score\n')
    file.write(response_score)

