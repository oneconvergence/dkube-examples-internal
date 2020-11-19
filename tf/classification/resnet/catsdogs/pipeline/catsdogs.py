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
    #Dkube authentication token
    auth_token,
    #Name of the project in dkube
    training_program,
    #Dataset to train on
    training_dataset,
    #Output model 
    training_output_model,
    #By default 'default' is used as the job group for runs
    job_group = 'default',
    #Framework. One of tensorflow, pytorch, sklearn
    framework = "tensorflow",
    #Framework version
    version = '1.14',
    #By default tf v1.14 image is used here, v1.13 or v1.14 can be used. 
    #Or any other custom image name can be supplied.
    #For custom private images, please input username/password
    training_container=json.dumps({'image':'docker.io/ocdr/d3-datascience-tf-cpu:v1.14', 'username':'', 'password': ''}),
    #Script to run inside the training container
    training_script="python model.py",
    #Input dataset mount paths
    training_input_dataset_mount="/opt/dkube/input",
    #Output models for training.
    training_output_mount="/opt/dkube/output",
    #Request gpus as needed. Val 0 means no gpu, then training_container=docker.io/ocdr/dkube-datascience-tf-cpu:v1.12.
    training_gpus=0,
    #Any envs to be passed to the training program
    training_envs=json.dumps([{"steps": 100}]),
    #Serving image
    serving_image=json.dumps({'image':'ocdr/tensorflowserver:1.14', 'username':'', 'password': ''}),
    #Transformer image
    transformer_image=json.dumps({'image':'docker.io/ocdr/catdogs-example-preprocess:2.0.7', 'username':'', 'password': ''}),
    #Script to execute the transformer
    transformer_code="tensorflow/classification/resnet/catsdogs/transformer/transformer.py"
):

    train       = dkube_training_op(auth_token, training_container,
                                    program=training_program, run_script=training_script,
                                    datasets=json.dumps([str(training_dataset)]), outputs=json.dumps([str(training_output_model)]),
                                    input_dataset_mounts=json.dumps([str(training_input_dataset_mount)]),
                                    output_mounts=json.dumps([str(training_output_mount)]),
                                    ngpus=training_gpus,
                                    envs=training_envs, job_group=job_group,
                                    framework=framework, version=version)
    serving     = dkube_serving_op(auth_token, train.outputs['artifact'],
                                serving_image=serving_image,
                                transformer_image=transformer_image,
                                transformer_project=training_program,
                                transformer_code=transformer_code).after(train)
if __name__ == '__main__':
  import kfp.compiler as compiler
  compiler.Compiler().compile(d3pipeline, __file__ + '.tar.gz')
