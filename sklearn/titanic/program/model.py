import pyarrow as pa
import pyarrow.parquet as pq
import os
import numpy as np
import requests
import argparse
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from dkube.sdk import *
import mlflow

inp_path = "/opt/dkube/input/"
out_path = "/opt/dkube/output/"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", dest="url", default=None, type=str, help="setup URL")
    parser.add_argument("--fs", dest="fs", required=True, type=str, help="featureset")

    global FLAGS
    FLAGS, unparsed = parser.parse_known_args()
    dkubeURL = FLAGS.url
    fs = FLAGS.fs

    ########--- Read features from input FeatureSet ---########

    # Featureset API
    authToken = os.getenv("DKUBE_USER_ACCESS_TOKEN")
    # Get client handle
    api = DkubeApi(URL=dkubeURL, token=authToken)

    # Read features
    feature_df = api.read_featureset(name = fs)  # output: data

    ########--- Train ---########

    # preparing input output pairs
    y = feature_df["Survived"].values
    x = feature_df.drop("Survived", 1).values

    # Training random forest classifier
    model_RFC = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    model_RFC.fit(x, y)
    predictions = model_RFC.predict(x)

    ########--- Log metrics to DKube ---########

    # Calculating accuracy
    accuracy = accuracy_score(y, predictions)
    # logging acuracy to DKube using mlflow
    mlflow.log_metric("accuracy", accuracy)

    ########--- Write model to DKube ---########

    # Exporting model
    filename = os.path.join(out_path, "model.joblib")
    joblib.dump(model_RFC, filename)
