# Import libraries
import argparse
from azureml.core import Run
import joblib
import json
import os
import pandas as pd
import shutil
import lightgbm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from azureml.core import Workspace, Datastore, Dataset

# Import functions from train.py
from train import split_data, train_model, get_model_metrics

# Get the output folder for the model from the '--output_folder' parameter
parser = argparse.ArgumentParser()
parser.add_argument('--output_folder', type=str, dest='output_folder', default="outputs")
args = parser.parse_args()
output_folder = args.output_folder

# Get the experiment run context
run = Run.get_context()

ws = run.experiment.workspace
datastore = ws.get_default_datastore()
datastore_paths = [(datastore, 'safe_driver/porto_seguro_safe_driver_prediction_input.csv')]
traindata = Dataset.Tabular.from_delimited_files(path=datastore_paths)
data_df = traindata.to_pandas_dataframe()
print(data_df)

# load the safe driver prediction dataset
#data_df = pd.read_csv('../data/porto_seguro_safe_driver_prediction_input.csv')

# Load the parameters for training the model from the file
with open("parameters.json") as f:
    pars = json.load(f)
    parameters = pars["training"]

# Log each of the parameters to the run
# for param_name, param_value in parameters.items():
#     run.parent.log(param_name, param_value)

# Use the functions imported from train_safe_driver.py to prepare data, train the model, and calculate the metrics

data = split_data(data_df)
model = train_model(data, parameters)
model_metrics = get_model_metrics(model, data)
#run.parent.log('auc', model_metrics['auc'])

print("Exporting the model as pickle file...")
outputs_folder = './model'
os.makedirs(outputs_folder, exist_ok=True)
model_filename = "porto_seguro_safe_driver_model.pkl"
model_path = os.path.join(outputs_folder, model_filename)

# Save the trained model to the output folder
# os.makedirs(output_folder, exist_ok=True)
# model_filename = "porto_seguro_safe_driver_model.pkl"
# model_path = output_folder + "/" + model_filename

joblib.dump(value=model, filename=model_path)


# upload the model file explicitly into artifacts
print("Uploading the model into run artifacts...")
run.upload_file(name="./outputs/models/" + model_filename, path_or_stream=model_path)
print("Uploaded the model {} to experiment {}".format(model_filename, run.experiment.name))
dirpath = os.getcwd()
print(dirpath)
print("Following files are uploaded ")
print(run.get_file_names())

run.complete()
