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

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


@app.route("/")
def greetings():        
    return 'Welcome to my model API'

# Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    """
    Calls the prediction function
    Returns:
            string with list of predictions            
        """
    logger.info('running predict')
    if request.method == 'POST':
        filepath = request.get_json()['filepath']
        dataset = ingestion.read_csv_files(filepath)
        return diagnostics.model_predictions(dataset)
    if request.method == 'GET':
        file = request.args.get('filepath')
        dataset = ingestion.read_csv_files(file)
        return {'predictions': str(diagnostics.model_predictions(dataset))}
    
# Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def get_score():        
    """
    check the score of the deployed model
    Returns:
           f1 score (str)
    """
    logger.info('running get_score')
    return {'F1 score': scoring.score_model()}


# Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def get_stats():        
    """
    check means, medians, and modes for each numerical column
    Returns:
            json dictionary of all calculated summary statistics
    """
    logger.info('running get_stats')
    return diagnostics.dataframe_summary()

# Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def get_diagnostics():        
    """
    check timing and percent NA values
    Returns:
            dict: Returns a value for all diagnostics
    """
    logger.info('running get_diagnostics')
    missing = diagnostics.missing_data()
    runtimes = diagnostics.execution_time()
    outdated_packages_ = diagnostics.outdated_packages_list()
    output = {
        'missing values (%)': missing,
        'Runtimes': runtimes,
        'Outdated packages': outdated_packages_
    }
    return str(output)

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
