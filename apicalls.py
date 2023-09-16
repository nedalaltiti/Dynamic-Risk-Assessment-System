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

class APICaller:
    def __init__(self, url, config_path):
        self.url = url
        self.config_path = config_path

    def load_config(self):
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        return config

    def make_post_request(self, endpoint, data):
        response = requests.post(f'{self.url}/{endpoint}', json=data).text
        logger.info(f'POST request /{endpoint}')
        return response

    def make_get_request(self, endpoint):
        response = requests.get(f'{self.url}/{endpoint}').text
        logger.info(f'GET request /{endpoint}')
        return response

    def generate_report(self, model_path, test_data_path):
        config = self.load_config()
        test_data_path = os.path.join(config['test_data_path'])
        model_path = os.path.join(config['output_model_path']) 

        response_pred = self.make_post_request('prediction', {
            'filepath': os.path.join(test_data_path, 'testdata.csv')
        })

        response_score = self.make_get_request('scoring')
        response_stats = self.make_get_request('summarystats')
        response_diag = self.make_get_request('diagnostics')

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

# Specify a URL that resolves my workspace
URL = "http://127.0.0.1:8000"

# Create an instance of APICaller
api_caller = APICaller(URL, 'config.json')

# Generate the report
api_caller.generate_report('output_model_path', 'test_data_path')


