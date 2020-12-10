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

import sys
sys.path.insert(0, os.path.abspath("/usr/local/lib/python3.6/dist-packages"))
from dkube.sdk import *

parser = argparse.ArgumentParser()
parser.add_argument("--url", dest = 'url', default=None, type = str, help="setup URL")

def log_metrics(key, value):
    url = "http://dkube-exporter.dkube:9401/mlflow-exporter"
    train_metrics = {}
    train_metrics['mode']="train"
    train_metrics['key'] = key
    train_metrics['value'] = value
    train_metrics['epoch'] = 1
    train_metrics['step'] = 1
    train_metrics['jobid']=os.getenv('DKUBE_JOB_ID')
    train_metrics['run_id']=os.getenv('DKUBE_JOB_UUID')
    train_metrics['username']=os.getenv('DKUBE_USER_LOGIN_NAME')
    requests.post(url, json = train_metrics)
    
inp_path = '/opt/dkube/input/'
out_path = '/opt/dkube/output/'
filename = 'featureset.parquet'
global FLAGS
FLAGS,unparsed=parser.parse_known_args()
dkubeURL = FLAGS.url
authToken = os.getenv('DKUBE_USER_ACCESS_TOKEN') # Dkube user access token for API authentication

# Dkube API calling
api = DkubeApi(URL=dkubeURL, token=authToken)

# Featureset API
featureset = DkubeFeatureSet()
# featureset path update
featureset.update_features_path(path=inp_path)

# Reading featureset
data  = featureset.read() # output: response json with data

feature_df = data["data"]

# preparing input output pairs
y = feature_df['Survived'].values
x = feature_df.drop('Survived', 1).values

# Training random forest classifier
model_RFC = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model_RFC.fit(x, y)
predictions = model_RFC.predict(x)

# Calculating accuracy
accuracy = accuracy_score(y, predictions)
# logging acuracy to DKube
log_metrics('accuracy', accuracy)

# Exporting model
filename = os.path.join(out_path, 'model.joblib')
joblib.dump(model_RFC, filename)
