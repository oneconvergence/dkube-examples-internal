import kfp.dsl as dsl
from kfp import components
from kubernetes import client as k8s_client
import json

from random import randint

dkube_training_op           = components.load_component_from_file("../components/training/component.yaml")
dkube_serving_op            = components.load_component_from_file("../components/serving/component.yaml")
dkube_viewer_op             = components.load_component_from_file('../components/viewer/component.yaml')

@dsl.pipeline(
    name='dkube-cats dogs-pl',
    description='sample cats dogs pipeline with dkube components'
)
def d3pipeline(
    #Dkube auth token
    auth_token,
    #By default tf v1.12 image is used here, v1.10, v1.11 or v1.13 can be used. 
    #Or any other custom image name can be supplied.
    #For custom private images, please input username/password 
    training_container=json.dumps({'image':'docker.io/ocdr/dkube-datascience-tf-gpu:v1.12', 'username':'', 'password': ''}),
    #Name of the workspace in dkube. Update accordingly if different name is used while creating a workspace in dkube.
    training_program="catsdogs",
    #Script to run inside the training container
    training_script="python model.py",
    #Input datasets for training. Update accordingly if different name is used while creating dataset in dkube.    
    training_datasets=json.dumps(["catsdogs"]),
    #Request gpus as needed. Val 0 means no gpu, then training_container=docker.io/ocdr/dkube-datascience-tf-cpu:v1.12.
    training_gpus=1,
    #Any envs to be passed to the training program
    training_envs=json.dumps([{"steps": 100}])):



    train       = dkube_training_op(auth_token, training_container,
                                    program=training_program, run_script=training_script,
                                    datasets=training_datasets, ngpus=training_gpus,
                                    envs=training_envs)
    serving     = dkube_serving_op(auth_token, train.outputs['artifact']).after(train)
    inference   = dkube_viewer_op(auth_token, serving.outputs['servingurl'], 
                                  'catsdogs', viewtype='inference').after(serving)


if __name__ == '__main__':
  import kfp.compiler as compiler
  compiler.Compiler().compile(d3pipeline, __file__ + '.tar.gz')
