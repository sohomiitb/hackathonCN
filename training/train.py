import os
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import lightgbm
from azureml.core import Workspace, Datastore, Dataset
from azureml.core import Run
import joblib

def split_data(data_df):
    """Split a dataframe into training and validation datasets"""
    features = data_df.drop(['target', 'id'], axis=1)
    labels = np.array(data_df['target'])
    features_train, features_valid, labels_train, labels_valid = train_test_split(features, labels, test_size=0.2,
                                                                                  random_state=0)

    train_data = lightgbm.Dataset(features_train, label=labels_train)
    valid_data = lightgbm.Dataset(features_valid, label=labels_valid, free_raw_data=False)

    return (train_data, valid_data)


def train_model(data, parameters):
    """Train a model with the given datasets and parameters"""
    # The object returned by split_data is a tuple.
    # Access train_data with data[0] and valid_data with data[1]

    model = lightgbm.train(parameters,
                           data[0],
                           valid_sets=data[1],
                           num_boost_round=500,
                           early_stopping_rounds=20)
    return model


def get_model_metrics(model, data):
    """Construct a dictionary of metrics for the model"""
    predictions = model.predict(data[1].data)
    fpr, tpr, thresholds = metrics.roc_curve(data[1].label, predictions)
    model_metrics = {"auc": (metrics.auc(fpr, tpr))}

    return model_metrics


def main():
    """This method invokes the training functions for development purposes"""
    run = Run.get_context()
    ws = run.experiment.workspace
    datastore = ws.get_default_datastore()
    datastore_paths = [(datastore, 'safe_driver/porto_seguro_safe_driver_prediction_input.csv')]
    traindata = Dataset.Tabular.from_delimited_files(path=datastore_paths)
    data_df = traindata.to_pandas_dataframe()
    print(data_df)
    # Read data from a file
    #data_df = pd.read_csv('porto_seguro_safe_driver_prediction_input.csv')

    # Hard code the parameters for training the model
    parameters = {
        'learning_rate': 0.02,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'sub_feature': 0.7,
        'num_leaves': 60,
        'min_data': 100,
        'min_hessian': 1,
        'verbose': 2
    }

    # Call the functions defined in this file
    data = split_data(data_df)
    model = train_model(data, parameters)
    model_metrics = get_model_metrics(model, data)

    # Print the resulting metrics for the model
    print(model_metrics)

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


if __name__ == '__main__':
    main()

