import kfp.dsl as dsl
from kfp import components
from kubernetes import client as k8s_client

import os
import json
from random import randint

dkube_preprocess_op         = components.load_component_from_file("../components/preprocess/component.yaml")
dkube_training_op           = components.load_component_from_file("../components/training/component.yaml")
dkube_serving_op            = components.load_component_from_file("../components/serving/component.yaml")
dkube_viewer_op             = components.load_component_from_file('../components/viewer/component.yaml')

@dsl.pipeline(
    name='dkube-regression-pl',
    description='sample regression pipeline with dkube components'
)

def d3pipeline(
    #Clinical preprocess
    clinical_preprocess_script="python cli-pre-processing.py",
    clinical_preprocess_datasets=json.dumps(["clinical"]),
    clinical_preprocess_input_mounts=json.dumps(["/opt/dkube/input"]),
    clinical_preprocess_outputs=json.dumps(["clinical-preprocessed"]),
    clinical_preprocess_output_mounts=json.dumps(["/opt/dkube/output"]),
    
    #Image preprocess
    image_preprocess_script="python img-pre-processing.py",
    image_preprocess_datasets=json.dumps(["images"]),
    image_preprocess_input_mounts=json.dumps(["/opt/dkube/input"]),
    image_preprocess_outputs=json.dumps(["images-preprocessed"]),
    image_preprocess_output_mounts=json.dumps(["/opt/dkube/output"]),
    
    #Clinical split
    clinical_split_script="python split.py --datatype clinical",
    clinical_split_datasets=json.dumps(["clinical-preprocessed"]),
    clinical_split_input_mounts=json.dumps(["/opt/dkube/input"]),
    clinical_split_outputs=json.dumps(["clinical-train", "clinical-test", "clinical-val"]),
    clinical_split_output_mounts=json.dumps(["/opt/dkube/outputs/train", "/opt/dkube/outputs/test", "/opt/dkube/outputs/val"]),
    
    #Image split
    image_split_script="python split.py --datatype image",
    image_split_datasets=json.dumps(["images-preprocessed"]),
    image_split_input_mounts=json.dumps(["/opt/dkube/input"]),
    image_split_outputs=json.dumps(["images-train", "images-test", "images-val"]),
    image_split_output_mounts=json.dumps(["/opt/dkube/outputs/train", "/opt/dkube/outputs/test", "/opt/dkube/outputs/val"])	,
    
    #RNA split
    rna_split_script="python split.py --datatype rna",
    rna_split_datasets=json.dumps(["rna"]),
    rna_split_input_mounts=json.dumps(["/opt/dkube/input"]),
    rna_split_outputs=json.dumps(["rna-train", "rna-test", "rna-val"]),
    rna_split_output_mounts=json.dumps(["/opt/dkube/outputs/train", "/opt/dkube/outputs/test", "/opt/dkube/outputs/val"]),
    
    #Training
    #In notebook DKUBE_USER_ACCESS_TOKEN is automatically picked up from env variable
    auth_token  = os.getenv("DKUBE_USER_ACCESS_TOKEN"),
    #By default tf v1.14 image is used here, v1.13 or v1.14 can be used. 
    #Or any other custom image name can be supplied.
    #For custom private images, please input username/password
    training_container=json.dumps({'image':'docker.io/ocdr/d3-datascience-tf-cpu:v1.14', 'username':'', 'password': ''}),
    #Name of the workspace in dkube. Update accordingly if different name is used while creating a workspace in dkube.
    training_program="regression",
    #Script to run inside the training container    
    training_script="python train_nn.py --epochs 5",
    #Input datasets for training. Update accordingly if different name is used while creating dataset in dkube.    
    training_datasets=json.dumps(["clinical-train", "clinical-val", "images-train",
                                  "images-val", "rna-train", "rna-val"]),
    training_input_dataset_mounts=json.dumps(["/opt/dkube/inputs/train/clinical", "/opt/dkube/inputs/val/clinical",
                                      "/opt/dkube/inputs/train/images", "/opt/dkube/inputs/val/images",
                                      "/opt/dkube/inputs/train/rna", "/opt/dkube/inputs/val/rna"]),
    training_outputs=json.dumps(["regression-model"]),
    training_output_mounts=json.dumps(["/opt/dkube/output"]),
    #Request gpus as needed. Val 0 means no gpu, then training_container=docker.io/ocdr/dkube-datascience-tf-cpu:v1.12    
    training_gpus=0,
    #Any envs to be passed to the training program    
    training_envs=json.dumps([{"steps": 100}]),
    
    #Evaluation
    evaluation_script="python evaluate.py",
    evaluation_datasets=json.dumps(["clinical-test", "images-test", "rna-test"]),
    evaluation_input_dataset_mounts=json.dumps(["/opt/dkube/inputs/test/clinical", "/opt/dkube/inputs/test/images",
                                      "/opt/dkube/inputs/test/rna"]),
    evaluation_models=json.dumps(["regression-model"]),
    evaluation_input_model_mounts=json.dumps(["/opt/dkube/inputs/model"]),
    
    #Serving
    #Device to be used for serving - dkube mnist example trained on gpu needs gpu for serving else set this param to 'cpu'
    serving_device='cpu',
    serving_container=json.dumps({'image':'docker.io/ocdr/new-preprocess:satish', 'username':'', 'password': ''})):
    
    clinical_preprocess  = dkube_preprocess_op(auth_token, training_container,
                                      program=training_program, run_script=clinical_preprocess_script,
                                      datasets=clinical_preprocess_datasets, outputs=clinical_preprocess_outputs,
                                      input_dataset_mounts=clinical_preprocess_input_mounts, output_mounts=clinical_preprocess_output_mounts)
    image_preprocess  = dkube_preprocess_op(auth_token, training_container,
                                      program=training_program, run_script=image_preprocess_script,
                                      datasets=image_preprocess_datasets, outputs=image_preprocess_outputs,
                                      input_dataset_mounts=image_preprocess_input_mounts, output_mounts=image_preprocess_output_mounts)
                                      
    clinical_split  = dkube_preprocess_op(auth_token, training_container,
                                      program=training_program, run_script=clinical_split_script,
                                      datasets=clinical_split_datasets, outputs=clinical_split_outputs,
                                      input_dataset_mounts=clinical_split_input_mounts,
                                      output_mounts=clinical_split_output_mounts).after(clinical_preprocess)
                                      
    image_split  = dkube_preprocess_op(auth_token, training_container,
                                      program=training_program, run_script=image_split_script,
                                      datasets=image_split_datasets, outputs=image_split_outputs,
                                      input_dataset_mounts=image_split_input_mounts,
                                      output_mounts=image_split_output_mounts).after(image_preprocess)
                                      
    rna_split  = dkube_preprocess_op(auth_token, training_container,
                                      program=training_program, run_script=rna_split_script,
                                      datasets=rna_split_datasets, outputs=rna_split_outputs,
                                      input_dataset_mounts=rna_split_input_mounts, output_mounts=rna_split_output_mounts)
                                      
    train       = dkube_training_op(auth_token, training_container,
                                    program=training_program, run_script=training_script,
                                    datasets=training_datasets, outputs=training_outputs,
                                    input_dataset_mounts=training_input_dataset_mounts,
                                    output_mounts=training_output_mounts,
                                    ngpus=training_gpus,
                                    envs=training_envs).after(clinical_split).after(image_split).after(rna_split)
    evaluate    = dkube_training_op(auth_token, training_container,
                                    program=training_program, run_script=evaluation_script,
                                    datasets=evaluation_datasets,
                                    input_dataset_mounts=evaluation_input_dataset_mounts,
                                    models=evaluation_models,
                                    input_model_mounts=evaluation_input_model_mounts,
                                    ngpus=training_gpus,
                                    envs=training_envs).after(train)
    serving     = dkube_serving_op(auth_token, train.outputs['artifact'], device=serving_device, serving_container=serving_container).after(evaluate)
    #inference   = dkube_viewer_op(auth_token, serving.outputs['servingurl'],
    #                              'digits', viewtype='inference').after(serving)
