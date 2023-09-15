"""
This script used to generate a confusion matrix
and generate PDF report.

Name: Nedal Altiti
Date: 09 / 15 2023
"""
import json
import os
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from diagnostics import model_predictions
from ingestion import read_csv_files

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = os.path.join(config['test_data_path'], 'testdata.csv') 
output_model_path = os.path.join(config['output_model_path'])

# Function for reporting
def score_model():
    """
    calculate a confusion matrix using the test data and the deployed model
    """
    data = read_csv_files(test_data_path)
    y = data['exited']
    pred = model_predictions(data)
    cm = confusion_matrix(y, pred)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # disp.plot()
    # logger.info(f'Savind confusion matrix in {output_model_path}')
    # plt.savefig(os.path.join(output_model_path, 'confusionmatrix.png'))

    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(cm, annot=True, cmap='Blues')
    ax.set_title('Confusion matrix')
    ax.set_xlabel('\nPredicted Label')
    ax.set_ylabel('True Label')
    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])
    logger.info(f"confusion matrix saved in {config['output_model_path']}")
    plt.savefig(os.path.join(output_model_path, 'confusionmatrix.png'))

if __name__ == '__main__':
    score_model()
