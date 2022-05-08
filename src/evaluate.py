"""Evaluation script for measuring model accuracy."""
import json
import logging
import pathlib
import tarfile
import numpy as np
import pandas as pd
import joblib
import os
import pandas as pd
from sklearn import model_selection

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def load_model(model_path):
    """
    Accepts path of the model in tar.gz format and returns model in joblib format
    """
    with tarfile.open(model_path) as tar:
        tar.extractall(path="..")
    clf = joblib.load("model.joblib")
    return clf

def prepare_test_data():
    '''
    This prepares test dataset from url https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv. 
    This should be changed to get test data from a more streamlined approach 
    '''
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    pima_df = pd.read_csv(url, names=names)
    pima_array = pima_df.values
    X = pima_array[:,0:8]
    Y = pima_array[:,8]
    test_size = 0.33
    seed = 7
    _, X_test, _, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
    return X_test, Y_test

if __name__ == "__main__":
    model_path = "/opt/ml/processing/model/model.tar.gz"

    logger.debug("Loading scikitlearn model.")
    model = load_model(model_path)

    logger.debug("Loading test input data.")
    X_test, Y_test = prepare_test_data()

    logger.info("Performing predictions against test data.")
    prediction_probabilities = model.predict(X_test)
    predictions = np.round(prediction_probabilities)

    precision = precision_score(Y_test, predictions, zero_division=1)
    recall = recall_score(Y_test, predictions)
    accuracy = accuracy_score(Y_test, predictions)
    conf_matrix = confusion_matrix(Y_test, predictions)
    fpr, tpr, _ = roc_curve(Y_test, prediction_probabilities)

    logger.debug("Accuracy: {}".format(accuracy))
    logger.debug("Precision: {}".format(precision))
    logger.debug("Recall: {}".format(recall))
    logger.debug("Confusion matrix: {}".format(conf_matrix))

    # Add below metrics to model in the model registry
    report_dict = {
        "binary_classification_metrics": {
            "accuracy": {"value": accuracy, "standard_deviation": "NaN"},
            "precision": {"value": precision, "standard_deviation": "NaN"},
            "recall": {"value": recall, "standard_deviation": "NaN"},
            "confusion_matrix": {
                "0": {"0": int(conf_matrix[0][0]), "1": int(conf_matrix[0][1])},
                "1": {"0": int(conf_matrix[1][0]), "1": int(conf_matrix[1][1])},
            },
            "receiver_operating_characteristic_curve": {
                "false_positive_rates": list(fpr),
                "true_positive_rates": list(tpr),
            },
        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
