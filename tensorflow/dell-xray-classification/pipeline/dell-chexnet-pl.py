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

#input for create_resource
WS_SOURCE_LINK = "https://github.com/oneconvergence/dkube-examples/tree/dell-model/tensorflow/dell-xray-classification"
DATASET_URL = "https://github.com/oneconvergence/dkube-examples/tree/dell-model/tensorflow/dell-xray-classification/dataset"

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

    # create resource stage
    create_resource_op = components.func_to_container_op(create_resource_job, base_image='docker.io/ocdr/dkube-datascience-tf-cpu:v1.14')
    create_res = create_resource_op(user=user,
                                         url=access_url,
                                         token=auth_token,
                                         ws_name=WORKSPACE,
                                         ws_link=WS_SOURCE_LINK,
                                         ds_name=PREPROCESS_DATASET,
                                         ds_link=DATASET_URL)

    # download dataset stage
    download_dataset_op = components.func_to_container_op(download_job, base_image='docker.io/ocdr/dkube-datascience-tf-cpu:v1.14') 
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
