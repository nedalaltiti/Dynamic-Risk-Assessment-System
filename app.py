"""
API set-up for model reporting: scripts that create reports related to the ML model, its performance, and related
diagnostics.

Name: Nedal Altiti
Date: 15 / 09 / 2023
"""
from flask import Flask, session, jsonify, request
import json
import os
import logging 
import ingestion, scoring, diagnostics

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json', 'r') as f:
    config = json.load(f)

output_folder_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])

class ModelAPI:
    def __init__(self, config_path):
        self.config_path = config_path

    def load_config(self):
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        return config

    @app.route("/")
    def greetings():
        return 'Welcome to my model API'

    @app.route("/prediction", methods=['POST', 'GET'])
    def predict():
        logger.info('running predict')
        if request.method == 'POST':
            filepath = request.get_json()['filepath']
            dataset = ingestion.read_csv_files(filepath)
            return diagnostics.model_predictions(dataset)
        if request.method == 'GET':
            file = request.args.get('filepath')
            dataset = ingestion.read_csv_files(file)
            return jsonify({'predictions': str(diagnostics.model_predictions(dataset))})

    @app.route("/scoring", methods=['GET'])
    def get_score():
        logger.info('running get_score')
        return jsonify({'F1 score': scoring.score_model(data_path, model_path)})

    @app.route("/summarystats", methods=['GET'])
    def get_stats():
        logger.info('running get_stats')
        return jsonify(diagnostics.dataframe_summary())

    @app.route("/diagnostics", methods=['GET'])
    def get_diagnostics():
        logger.info('running get_diagnostics')
        missing = diagnostics.missing_data()
        runtimes = diagnostics.execution_time()
        outdated_packages_ = diagnostics.outdated_packages_list()
        output = {
            'missing values (%)': missing,
            'Runtimes': runtimes,
            'Outdated packages': outdated_packages_
        }
        return jsonify(output)

if __name__ == "__main__":
    config_path = 'config.json'
    model_path = os.path.join(prod_deployment_path, 'trainedmodel.pkl')
    data_path = os.path.join(output_folder_path, 'finaldata.csv')
    model_api = ModelAPI(config_path)
    config = model_api.load_config()
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)