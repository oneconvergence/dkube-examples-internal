import sys, time
from dkube.sdk import *
from dkube.sdk.lib.api import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--url', dest = 'url', type=str)
parser.add_argument('--auth_token',dest = 'authtoken', type=str)
parser.add_argument('--user',dest = 'user', type=str)
args = parser.parse_args()

dkubeURL = args.url
authToken = args.authtoken
user = args.user

api = DkubeApi(dkubeURL=dkubeURL, authToken=authToken)
try:
    res = api.get_project(user, 'regression')
    print("Datum already exists")
except:
    project = DkubeProject(user, name='regression')
    project.update_project_source(source='github')
    project.update_github_details('https://github.com/oneconvergence/dkube-examples/tree/reg-demo-2-1-3-1/reg_demo', branch='reg-demo-2-1-3-1')
    api.create_project(project)

    
try:
    res = api.get_dataset(user, 'clinical')
    print("Datum already exists")
except:
    dataset = DkubeDataset(user, name='clinical')
    dataset.update_dataset_source(source='github')
    dataset.update_github_details('https://github.com/oneconvergence/dkube-examples/tree/reg-demo-data/reg_demo_data/clinical', branch='reg-demo-data')
    api.create_dataset(dataset)

try:
    res = api.get_dataset(user, 'images')
    print("Datum already exists")
except:
    dataset = DkubeDataset(user, name='images')
    dataset.update_dataset_source(source='github')
    dataset.update_github_details('https://github.com/oneconvergence/dkube-examples/tree/reg-demo-data/reg_demo_data/image_data', branch='reg-demo-data')
    api.create_dataset(dataset)
    
try:
    res = api.get_dataset(user, 'rna')
    print("Datum already exists")
except:
    dataset = DkubeDataset(user, name='rna')
    dataset.update_dataset_source(source='github')
    dataset.update_github_details('https://github.com/oneconvergence/dkube-examples/tree/reg-demo-data/reg_demo_data/rna', branch='reg-demo-data')
    api.create_dataset(dataset)

    
dvs_datasets = ['clinical-preprocessed', 'clinical-train', 'clinical-test', 'clinical-val',
                'images-preprocessed', 'images-train', 'images-test', 'images-val',
                'rna-train', 'rna-test', 'rna-val']

for each_dataset in dvs_datasets:
    try:
        res = api.get_dataset(user, name=each_dataset)
        print("Datum already exists")
    except:
        dataset = DkubeDataset(user, name=each_dataset)
        dataset.update_dataset_source(source='dvs')
        api.create_dataset(dataset)
 
try:
    res = api.get_model(user, 'regression-model')
    print("Datum already exists")
except:
    model = DkubeModel(user, name='regression-model')
    model.update_model_source(source='dvs')
    api.create_model(model)
print("Finishing Dataset creation")
time.sleep(60)
