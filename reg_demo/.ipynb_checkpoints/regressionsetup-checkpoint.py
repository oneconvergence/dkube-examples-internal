import sys, time
from dkube.sdk import *
from dkube.sdk.lib.api import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--url', dest = 'url', type=str)
parser.add_argument('--auth_token',dest = 'authtoken', type=str)
parser.add_argument('--user',dest = 'user' type=str)
args = parser.parse_args()

dkubeURL = args.url
authToken = args.authtoken
user = args.user

api = DkubeApi(dkubeURL=dkubeURL, authToken=authToken)

project = DkubeProject(user, name='regression')
project.update_project_source(source='github')
project.update_github_details('https://github.com/oneconvergence/dkube-examples/tree/reg-demo/reg_demo', branch='reg-demo', authmode = 'apikey')
try:
    api.create_project(project)
except:
    print("Datum already exists")
    
dataset = DkubeDataset(user, name='clinical')
dataset.update_dataset_source(source='github')
dataset.update_github_details('https://github.com/oneconvergence/dkube-examples/tree/reg-demo-data/reg_demo_data/clinical', branch='reg-demo-data')
try:
    api.create_dataset(dataset)
except:
    print("Datum already exists")
    
dataset = DkubeDataset(user, name='images')
dataset.update_dataset_source(source='github')
dataset.update_github_details('https://github.com/oneconvergence/dkube-examples/tree/reg-demo-data/reg_demo_data/image_data', branch='reg-demo-data')
try:
    api.create_dataset(dataset)
except:
    print("Datum already exists")
    
dataset = DkubeDataset(user, name='rna')
dataset.update_dataset_source(source='github')
dataset.update_github_details('https://github.com/oneconvergence/dkube-examples/tree/reg-demo-data/reg_demo_data/rna', branch='reg-demo-data')
try:
    api.create_dataset(dataset)
except:
    print("Datum already exists")
    
dvs_datasets = ['clinical-preprocessed', 'clinical-train', 'clinical-test', 'clinical-val',
                'images-preprocessed', 'images-train', 'images-test', 'images-val',
                'rna-train', 'rna-test', 'rna-val']

for each_dataset in dvs_datasets:
    dataset = DkubeDataset(user, name=each_dataset)
    dataset.update_dataset_source(source='dvs')
    try:
        api.create_dataset(dataset)
    except:
        print("Datum already exists")
        
model = DkubeModel(user, name='regression-model')
model.update_model_source(source='dvs')
try:
    api.create_model(model)
except:
    print("Datum already exists")

print("Finishing Dataset creation")
time.sleep(60)