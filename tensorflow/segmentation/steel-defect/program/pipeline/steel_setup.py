import sys, time
from dkube.sdk import *
from dkube.sdk.lib.api import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--url', dest = 'url', type=str)
parser.add_argument('--auth_token',dest = 'authtoken', type=str)
parser.add_argument('--git_token',dest = 'gittoken', type=str)
parser.add_argument('--user',dest = 'user', type=str)
args = parser.parse_args()

dkubeURL = args.url
authToken = args.authtoken
user = args.user
gittoken = args.gittoken

api = DkubeApi(dkubeURL=dkubeURL, authToken=authToken)

project = DkubeProject(user, name='steel')
project.update_project_source(source='github')
project.update_github_details('https://github.com/oneconvergence/dkube-apps/tree/steel', branch='steel', authmode = 'apikey', authkey = gittoken)
try:
    api.create_project(project)
except:
    print("Datum already exists")
    
dataset = DkubeDataset(user, name='steel-data')
dataset.update_dataset_source(source='github')
dataset.update_github_details('https://github.com/oneconvergence/dkube-apps/tree/severstal-data', branch='severstal-data', authmode = 'apikey', authkey = gittoken)
try:
    api.create_dataset(dataset)
except:
    print("Datum already exists")
    
dvs_datasets = ['steel-preprocessed', 'steel-train', 'steel-test']

for each_dataset in dvs_datasets:
    dataset = DkubeDataset(user, name=each_dataset)
    dataset.update_dataset_source(source='dvs')
    try:
        api.create_dataset(dataset)
    except:
        print("Datum already exists")
        
model = DkubeModel(user, name='resUnet')
model.update_model_source(source='dvs')
try:
    api.create_model(model)
except:
    print("Datum already exists")

print("Finishing Dataset creation")
time.sleep(60)