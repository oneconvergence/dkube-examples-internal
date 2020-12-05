import zipfile
import os
import numpy as np
import pandas as pd
import argparse
import sys
sys.path.insert(0, os.path.abspath("/usr/local/lib/python3.6/dist-packages"))
from dkube.sdk import *

inp_path = '/opt/dkube/input/'
train_out_path = "/opt/dkube/output/"
# test_out_path = "./"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", dest = 'url', default=None, type = str, help="setup URL")
    global FLAGS
    FLAGS, unparsed = parser.parse_known_args()
    dkubeURL = FLAGS.url
    authToken = os.getenv('DKUBE_USER_ACCESS_TOKEN')
    
    if not os.path.exists('titanic'):
        os.makedirs('titanic')
    with zipfile.ZipFile(os.path.join(inp_path,'titanic.zip'), 'r') as zip_ref:
        zip_ref.extractall('titanic')

    train_data = pd.read_csv("titanic/train.csv")

    print(train_data.describe())

    train_data['Age'].fillna(value=train_data['Age'].median(), inplace=True)

    train_data = train_data[train_data['Fare'] < 100]

    train_data['Embarked'].fillna(method = 'ffill' , inplace = True)

    # test_data = pd.read_csv("titanic/test.csv")
    # test_data['Age'].fillna(value=test_data['Age'].median(), inplace=True)
    # test_data['Fare'].fillna(test_data['Fare'].median() , inplace = True)

    features = ['Pclass', 'Sex', 'SibSp', 'Parch']
    train_df = pd.get_dummies(train_data[features])
    train_df = pd.concat([train_data[['Age', 'Fare', 'Survived']], train_df], axis=1)
    print(train_df.head())
    
    dataset = train_df
    featureset = DkubeFeatureSet()
    featureset.update_features_path(path=train_out_path)
    featureset.write(dataset)
    ####### Featureset metadata #########
    keys   = dataset.keys()
    schema = dataset.dtypes.to_list()
    featureset_metadata = []
    for i in range(len(keys)):
        metadata = {}
        metadata["name"] = str(keys[i])
        metadata["description"] = None
        metadata["schema"] = str(schema[i])
        featureset_metadata.append(metadata)
        
    api = DkubeApi(URL=dkubeURL, token=authToken)
    featureset_metadata = yaml.dump(featureset_metadata, default_flow_style=False)
    with open("fspec.yaml", 'w') as f:
         f.write(featureset_metadata)
    resp = api.upload_featurespec(featureset = 'mnist-fs',filepath = "./fspec.yaml")
    print("featurespec upload response:", resp)
    resp = api.commit_features()
    print("featureset commit response:", resp)
