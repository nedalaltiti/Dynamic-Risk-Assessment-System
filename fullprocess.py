"""
File with scripts that automate the ML model scoring and monitoring process.

This step includes checking for the criteria that will require model re-deployment, and re-deploying models as
necessary.

Name: Nedal Altiti
Date: 16 / 09 / 2023
"""
import json
import os
import logging
import subprocess
from datetime import datetime

from ingestion import merge_multiple_dataframe
from scoring import score_model
from training import train_model
from deployment import ModelDeployment

# Initialize logging
logging.basicConfig(filename='journal.log',
                    level=logging.INFO,
                    filemode='a',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y/%m/%d %I:%M:%S %p')
logger = logging.getLogger(__name__)

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = os.path.join(config['input_folder_path'])
output_folder_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
output_model_path = os.path.join(config['output_model_path'])


def go(testing_mode=False):
    """
    Full process of ML model scoring and monitoring
    :param testing_mode: used to specify if we are testing or running in production.
    :return:
    """
    exec_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    msg = '*' * 10 + ' ' * 4 + f'{exec_time}: Running full process' + ' ' * 4 + '*' * 10
    logger.info(msg)
    with open('cron_jobs_executed.txt', 'w') as f:
        f.write(msg + '\n')

    if testing_mode:
        logger.info('testing mode activated')

    # checking if directory for model exists
    if not os.path.isdir(output_model_path):
        os.mkdir(output_model_path)
        logger.info(f'Creation of directory {output_model_path} where the model will be saved')

    # Check and read new data
    # read ingestedfiles.txt
    logger.info('Checking if new data are available')
    try:
        with open(os.path.join(prod_deployment_path, 'ingestedfiles.txt')) as f:
            ingested_files = f.read().split('\n')
        ingested_files = [f for f in ingested_files if f]  # remove any empty string
        first_implementation = False
    except FileNotFoundError:
        logger.info('No file has been ingested yet in production')
        logger.info('This is the first time the production model is deployed')
        ingested_files = []
        first_implementation = True
    # determine whether the source data folder has files that aren't listed in ingestedfiles.txt
    filenames = next(os.walk(input_folder_path), (None, None, []))[2]  # [] if no file
    new_data = [file for file in filenames if file not in ingested_files]

    # Deciding whether to proceed, part 1
    # if we found new data, we should proceed, otherwise, we end the process here
    if new_data:
        logger.info('There are new data, we need to ingest them.')
        merge_multiple_dataframe()
    elif not testing_mode:
        logger.info('No new file found, stop the process here.')
        exit()
    else:
        logger.info('No new file found.')
        logger.info('as we are in testing mode process continue. Should stop in production')

    # Checking for model drift
    if first_implementation:
        model_drift = False
    else:
        # check whether the score from the deployed model is different from the score from the model that uses the
        # newest ingested data
        logger.info('checking for model drift using newly ingested data')
        with open(os.path.join(prod_deployment_path, 'latestscore.txt'), 'r') as f:
            latest_score = float(f.read())

        new_score = score_model(data_path, model_path)
        model_drift = True if new_score < latest_score else False

    # Deciding whether to proceed, part 2
    # # if we found model drift, we proceed. otherwise, we do end the process here
    if not model_drift and not testing_mode and not first_implementation:
        logger.info('No drift found, process stop here.')
        exit()
    elif first_implementation and not testing_mode:
        logger.info("First deployment in production of the model.")
    elif not testing_mode:
        logger.info('Model drift found we need to train and deploy a new model')
    else:
        logger.info('No drift found')
        logger.info('as we are in testing mode process continue. Should stop in production')

    # Re-training
    train_model(data_path, model_path)

    # Re-deployment
    deployment = ModelDeployment(config)
    deployment.store_model_into_pickle()

    # Diagnostics and reporting
    # run diagnostics.py and reporting.py for the re-deployed model
    logger.info('Running diagnostics and reporting: execute apicalls.py')
    subprocess.run(['python', 'apicalls.py'])


if __name__ == '__main__':
    print('starting')
    model_path = os.path.join(prod_deployment_path, 'trainedmodel.pkl')
    data_path = os.path.join(output_folder_path, 'finaldata.csv')
    go()