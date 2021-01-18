import zipfile
import os
import numpy as np
import pandas as pd
import argparse
import yaml
from dkube.sdk import *

inp_path = "/opt/dkube/input/"
train_out_path = "/opt/dkube/output/"

if __name__ == "__main__":

    ########--- Parse for parameters ---########

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", dest="url", default=None, type=str, help="setup URL")
    parser.add_argument("--fs", dest="fs", required=True, type=str, help="featureset")

    global FLAGS
    FLAGS, unparsed = parser.parse_known_args()

    ########--- Get DKube client handle ---########

    dkubeURL = FLAGS.url
    # Dkube user access token for API authentication
    authToken = os.getenv("DKUBE_USER_ACCESS_TOKEN")
    # Get client handle
    api = DkubeApi(URL=dkubeURL, token=authToken)

    ########--- Extract and load data  ---########

    if not os.path.exists("titanic"):
        os.makedirs("titanic")
    with zipfile.ZipFile(os.path.join(inp_path, "titanic.zip"), "r") as zip_ref:
        zip_ref.extractall("titanic")

    train_data = pd.read_csv("titanic/train.csv")
    print(train_data.describe())

    ########--- Process raw data  ---########

    # Fill in null values with median
    train_data["Age"].fillna(value=train_data["Age"].median(), inplace=True)

    # Drop rows where fare is less than 100
    train_data = train_data[train_data["Fare"] < 100]

    # Fill in null values
    train_data["Embarked"].fillna(method="ffill", inplace=True)

    # test_data = pd.read_csv("titanic/test.csv")
    # test_data['Age'].fillna(value=test_data['Age'].median(), inplace=True)
    # test_data['Fare'].fillna(test_data['Fare'].median() , inplace = True)

    # Select features for training
    features = ["Pclass", "Sex", "SibSp", "Parch"]
    train_df = pd.get_dummies(train_data[features])
    train_df = pd.concat([train_data[["Age", "Fare", "Survived"]], train_df], axis=1)
    print(train_df.head())

    ########--- Upload Featureset metadata ---########

    # featureset to use
    fs = FLAGS.fs
    # Features
    df = train_df
    # Prepare featurespec - Name, Description, Schema for each feature
    
    keys = df.keys()
    schema = df.dtypes.to_list()
    featureset_metadata = []
    for i in range(len(keys)):
        metadata = {}
        metadata["name"] = str(keys[i])
        metadata["description"] = None
        metadata["schema"] = str(schema[i])
        featureset_metadata.append(metadata)

    # Commit featureset
    resp = api.commit_featureset(name=fs, df=df, metadata=featureset_metadata)
    print("featureset commit response:", resp)
