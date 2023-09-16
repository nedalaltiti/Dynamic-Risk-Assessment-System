"""
Data ingestion process.

Name: Nedal Altiti
Date: 09 / 14 /2023
"""
import pandas as pd
import os
import json
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


with open('config.json', 'r') as f:
    config = json.load(f)

def read_csv_files(filename):
    return pd.read_csv(filename)

def merge_multiple_dataframe():
    logging.info('starting data ingestion process')
    input_folder_path = config['input_folder_path']
    output_folder_path = config['output_folder_path']

    csv_files = [file for file in os.listdir(input_folder_path) if file.endswith('.csv')]

    dataframes = []
    for file in csv_files:
        filepath = os.path.join(input_folder_path, file)
        df = read_csv_files(filepath)
        dataframes.append(df)
    merged_df = pd.concat(dataframes, ignore_index=True).drop_duplicates()

    output_file = os.path.join(output_folder_path, 'finaldata.csv')
    if os.path.exists(output_file):
        logging.info('Output file already exists. Skipping saving process.')
    else:
        merged_df.to_csv(output_file, index=False)
        logging.info('Data ingestion completed successfully.')

    ingested_files = '\n'.join(csv_files)
    ingested_files_path = os.path.join(output_folder_path, 'ingestedfiles.txt')
    if not os.path.exists(ingested_files_path):
        with open(ingested_files_path, 'w') as f:
            f.write(ingested_files)
            logging.info('Ingested files record created.')

if __name__ == '__main__':
    merge_multiple_dataframe()
