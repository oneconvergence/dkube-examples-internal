import kfp.dsl as dsl
from kfp import components
from kubernetes import client as k8s_client
import json

from random import randint

dkube_training_op           = components.load_component_from_file("../components/training/component.yaml")
dkube_serving_op            = components.load_component_from_file("../components/serving/component.yaml")
dkube_viewer_op             = components.load_component_from_file('../components/viewer/component.yaml')

@dsl.pipeline(
    name='dkube-bolts-pl',
    description='sample bolts pipeline with dkube components'
)
def d3pipeline(
    access_url,
    auth_token,
    training_container=json.dumps({'image':'docker.io/ocdr/dkube-datascience-tf-gpu:v1.12', 'username':'', 'password': ''}),
    training_program="catsdogs",
    training_script="python model.py",
    training_datasets=json.dumps(["catsdogs"]),
    training_gpus=1,
    training_envs=json.dumps([{"steps": 100}])):



    train       = dkube_training_op(auth_token, training_container,
                                    program=training_program, run_script=training_script,
                                    datasets=training_datasets, ngpus=training_gpus,
                                    envs=training_envs, access_url=access_url)
    serving     = dkube_serving_op(auth_token, train.outputs['artifact'], access_url=access_url).after(train)
    inference   = dkube_viewer_op(auth_token, serving.outputs['servingurl'], 
                                  'catsdogs', viewtype='inference', access_url=access_url).after(serving)


if __name__ == '__main__':
  import kfp.compiler as compiler
  compiler.Compiler().compile(d3pipeline, __file__ + '.tar.gz')
