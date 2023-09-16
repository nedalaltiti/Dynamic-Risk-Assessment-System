"""
Diagnosting the Logistic Regression Model and the Data.

Name: Needal Altiti
Date: 15 / 09 / 2023
"""
import os
import json
import logging
import pickle
import subprocess
import timeit
import numpy as np
import pandas as pd
from typing import Dict, List, Union

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'], 'finaldata.csv') 
test_data_path = os.path.join(config['test_data_path'], 'testdata.csv') 
prod_deployment_path = os.path.join(config['prod_deployment_path'])

def segregate_dataset(dataset):
    """
    Read the dataset.
    Returns:
            X, y
    """
    X = pd.read_csv(dataset).iloc[:, 1:-1].values.reshape(-1, 3)
    y = pd.read_csv(dataset)['exited'].values.reshape(-1, 1).ravel()
    return X, y

def load_model(model_path: str):
    """
    Load the trained model.
    Returns: 
            the model
    """
    model_path = os.path.join(prod_deployment_path, 'trainedmodel.pkl')
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def model_predictions(data=None) -> List[Union[int, float]]:
    """
    read the deployed model and a test dataset, calculate predictions
    Returns:
            list: Returns a list containing all predictions
    """
    X,y = segregate_dataset(test_data_path)
    logger.info('Loading the model')
    model = load_model(prod_deployment_path)
    logger.info('calculate model predictions')
    pred = model.predict(X)
    return pred.tolist()

def dataframe_summary() -> List[Dict[str, Dict[str, float]]]:
    """
    calculate summary statistics here
    Returns:
            list: Returns a list containing all summary statistics
    """
    logger.info('calculate statistics on the data')

    # collect dataset
    data = pd.read_csv(dataset_csv_path)
    data = data.drop('exited', axis=1)

    # Select numeric columns
    numeric_col_index = np.where(data.dtypes != object)[0]
    numeric_col = data.iloc[:, numeric_col_index]
    stats_dict = {}
    stats_dict['col_means'] = dict(numeric_col.mean(axis=0))
    stats_dict['col_medians'] = dict(numeric_col.median(axis=0))
    stats_dict['col_std'] = dict(numeric_col.std(axis=0))

    return [stats_dict]

def missing_data() -> Dict[str, float]:
    """
    Check the percentage of missing data for each column.
    Returns:
            Dictionary with keys corresponding to the columns of the dataset.
            Each element of the dictionary gives the percent of NA values in a particular column of the data.
    """
    logger.info('calculate missing values percentage for each column')

    data = pd.read_csv(dataset_csv_path)
    missing = data.isna().sum()
    n_data = data.shape[0]
    missing = missing / n_data
    return missing.to_dict()

def execution_time() -> Dict[str, float]:
    """
    Calculate timing of ingestion.py and training.py
    Returns:
        list: Returns a list of 2 timing values in seconds
    """
    logger.info('calculate timing of training.py and ingestion.py')

    times = []
    scripts = ['ingestion.py', 'training.py']
    for script in scripts:
        starttime = timeit.default_timer()
        subprocess.run(['python', script])
        timing = timeit.default_timer() - starttime
        times.append(timing)

    formatted_times = ["{:.2f}".format(time) for time in times]
    output = [f"{script}: {timing}" for script, timing in zip(scripts, formatted_times)]

    return output

def outdated_packages_list() -> List[Dict[str, str]]:
    """
    check dependencies
    Returns:
            list: Returns the list of outdated dependencies
    """
    logger.info('check the dependencies')
    
    outdated = subprocess.run(
            ['pip', 'list', '--outdated', '--format', 'json'], capture_output=True).stdout
    outdated = outdated.decode('utf8').replace("'", '"')
    outdated_list = json.loads(outdated)
    return outdated_list

def save_diagnostics() -> Dict[str, Union[List[Union[int, float]], List[Dict[str, Dict[str, float]]], Dict[str, float], List[str], List[Dict[str, str]]]]:
        """
        Save all diagnostics in json file
        """
        diagnostics = {
            "TestDataPrediction": model_predictions(),
            "DataFrameSummary": dataframe_summary(),
            "MissingData": missing_data(),
            "ExecutionTimes": execution_time(),
            "PackagesOutdated": outdated_packages_list(),
        }

        logger.info(f"Saving Diagnostics in {prod_deployment_path}")

        with open(os.path.join(prod_deployment_path, 'diagnostics.json'), 'w') as file:
            file.write(json.dumps(diagnostics, indent=2))

if __name__ == '__main__':
    print(model_predictions())
    print(dataframe_summary())
    print(missing_data())
    print(execution_time())
    print(outdated_packages_list())
    save_diagnostics()