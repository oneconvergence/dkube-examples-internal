import kfp.dsl as dsl
from kfp import components
import json
import time
import os
import sys
sys.path.insert(0,'./create_resource.py')
from create_resource import create_resource_job
# from download_NIH_dataset import download
from launch_download_job import download_job
WS_SOURCE_LINK = "https://github.com/oneconvergence/dkube-examples/tree/dell-model-1.4.1-pipeline/tensorflow/dell-xray-classification"
DATASET_URL = "https://github.com/oneconvergence/dkube-examples/tree/dell-model-1.4.1-pipeline/tensorflow/dell-xray-classification/dataset"
# WS_NAME = "chexnet-ws"
# DS_NAME = "chexnet"
######
'''
def create_ds(url, user, token):
    import os
    import argparse
    import json
    import requests
    import datetime
    import time
    from string import Template
    from requests.packages import urllib3
    create_url = Template('$url/dkube/v2/users/$user/datums')
    header = {"content-type": "application/keyauth.api.v1+json",
              'Authorization': 'Bearer {}'.format(token)}
    if url[-1] == '/':
        url = url[:-1]
    try:
        url = create_url.substitute({'url': url,
                                     'user': user})
        create_header = header.copy()
        session = requests.Session()
        data = {"class": "dataset",
                "name": DS_NAME,
                "remote": False,
                "source": "git",
                "tags": [],
                "url": DATASET_URL}
        data = json.dumps(data)
        resp = session.post(
            url, data=data, headers=create_header, verify=False)
        if resp.status_code != 200:
            print('Unable to create dataset %s, It may be already exist' % DS_NAME)
            return None
    except Exception as e:
        return None

def create_ws(url, user, token):
    import os
    import argparse
    import json
    import requests
    import datetime
    import time
    from string import Template
    from requests.packages import urllib3
    create_url = Template('$url/dkube/v2/users/$user/datums')
    header = {"content-type": "application/keyauth.api.v1+json",
              'Authorization': 'Bearer {}'.format(token)}
    if url[-1] == '/':
        url = url[:-1]
    try:
        url = create_url.substitute({'url': url,
                                     'user': user})
        create_header = header.copy()
        session = requests.Session()
        data = {"class": "program",
                "name": WS_NAME,
                "remote": False,
                "source": "git",
                "tags": [],
                "url": WS_SOURCE_LINK}
        data = json.dumps(data)
        resp = session.post(
            url, data=data, headers=create_header, verify=False)
        if resp.status_code != 200:
            print('Unable to create workspace %s, It may be already exist' % WS_NAME)
            return None
    except Exception as e:
        return None
'''
###########


#Changes here
# dkube_create_resource_op = components.load_component_from_file(
#   "components/create_resource/component.yaml")

dkube_download_dataset_op = components.load_component_from_file(
  "components/download_dataset/component.yaml")
dkube_preprocess_op = components.load_component_from_file(
    "components/preprocess/component.yaml")
dkube_training_op = components.load_component_from_file(
    "components/training/component.yaml")
dkube_serving_op = components.load_component_from_file(
    "components/serving/component.yaml")
dkube_viewer_op = components.load_component_from_file(
    "components/viewer/component.yaml")

# constants
WORKSPACE = "chexnet-ws"
# input dataset for preprocessing
PREPROCESS_DATASET = "chexnet"
TARGET_DATASET = "chexnet-preprocessed"
STEPS = 20000  # max no of steps
EPOCHS = 1
BATCHSIZE = 32
SERVING_EXAMPLE = "chestnet"


@dsl.pipeline(name='Dkube-ChexNet-pl',
              description=('Dell ChexNet pipeline'
                           'with dkube components'))
def d3pipeline(
    access_url,
    auth_token,
    user,
    preprocess_container=json.dumps(
        {'image': 'docker.io/ocdr/dkube-datascience-tf-cpu:v1.14'}),
    preprocess_script="python preprocess.py",
    preprocess_program=WORKSPACE,
    preprocess_target_name=TARGET_DATASET,  # dataset
    # RAW dataset containing zip files of Chest X-Rays from NIH
    preprocess_datasets=json.dumps([PREPROCESS_DATASET]),
    training_container=json.dumps(
        {'image': 'docker.io/ocdr/dkube-datascience-tf-gpu:v1.14'}),
    training_program=WORKSPACE,
    training_script="python model.py --ngpus=1",
    training_gpus=1,
    training_envs=json.dumps([{"steps": STEPS,
                               "epochs": EPOCHS,
                               "batchsize": BATCHSIZE}])):

    # create dkube resource stage
    # create_resource = dkube_create_resource_op(access_url,
    #                                            auth_token,
    #                                            user)

    # dkube_create_workspace_op = components.func_to_container_op(create_ws, base_image='ocdr/dkube-datascience-tf-cpu:v1.13')
    # create_workspace = dkube_create_workspace_op(access_url, user, auth_token)

    # dkube_create_dataset_op = components.func_to_container_op(create_ds, base_image='ocdr/dkube-datascience-tf-cpu:v1.13')
    # create_dataset = dkube_create_dataset_op(access_url, user, auth_token).after(create_workspace)

    # create resource
    create_resource_op = components.func_to_container_op(create_resource_job, base_image='docker.io/ocdr/dkube-datascience-tf-cpu:v1.14')
    create_res = create_resource_op(user=user,
                                         url=access_url,
                                         token=auth_token,
                                         ws_name=WORKSPACE,
                                         ws_link=WS_SOURCE_LINK,
                                         ds_name=PREPROCESS_DATASET,
                                         ds_link=DATASET_URL)

    # download dataset
    download_dataset_op = components.func_to_container_op(download_job, base_image='docker.io/ocdr/dkube-datascience-tf-cpu:v1.14') #.after(dkube_create_dataset_op)  #.after(create_resource)
    download_dataset = download_dataset_op(url=access_url,
                                           user=user,
                                           token=auth_token,
                                           ws_name=WORKSPACE,
                                           ds_name=PREPROCESS_DATASET).after(create_res)
    # preprocessing stage
    preprocess = dkube_preprocess_op(auth_token, preprocess_target_name,
                                     preprocess_container,
                                     program=preprocess_program,
                                     datasets=preprocess_datasets,
                                     run_script=preprocess_script).after(download_dataset)

    # training stage
    preprocess_dataset_name = json.dumps([str(preprocess_target_name)])
    train = dkube_training_op(auth_token, training_container,
                              program=training_program,
                              run_script=training_script,
                              datasets=preprocess_dataset_name,
                              ngpus=training_gpus,
                              envs=training_envs).after(preprocess)
    # serving stage
    serving = dkube_serving_op(
        auth_token, train.outputs['artifact']).after(train)
    # inference stage
    inference = dkube_viewer_op(
        auth_token, serving.outputs['servingurl'],
        SERVING_EXAMPLE, viewtype='inference').after(serving)


if __name__ == '__main__':
    import kfp.compiler as compiler
    compiler.Compiler().compile(d3pipeline, __file__ + '.tar.gz')
