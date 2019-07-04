import kfp.dsl as dsl
from kfp import components
import json
import time

dkube_preprocess_op = components.load_component_from_file(
    "components/preprocess/component.yaml")
dkube_training_op = components.load_component_from_file(
    "components/training/component.yaml")
dkube_serving_op = components.load_component_from_file(
    "components/serving/component.yaml")
dkube_viewer_op = components.load_component_from_file(
    "components/viewer/component.yaml")

# constants
INT_MAX = 1 << 31
WORKSPACE = "Anjani-tiles-videos"
PREPROCESS_DATASET = "Anjani-tiles-s3-bucket"
TARGET_DATASET = "Anjani-tiles"
STEPS = 20000  # max no of steps
EPOCHS = 100
BATCHSIZE = 32
SERVING_EXAMPLE = "catsdogs"


@dsl.pipeline(name='dkube-Anjani-tiles-pl',
              description=('Anjani-tiles pipeline with videos as input '
                           'with dkube components'))
def d3pipeline(
               access_url,
               auth_token,
               preprocess_container=json.dumps(
                   {'image': 'docker.io/ocdr/dkube-datascience-tf-cpu:v1.10'}),
               preprocess_script="python preprocessing.py",
               preprocess_program=WORKSPACE,
               preprocess_target_name=TARGET_DATASET,  # dataset
               # AWS S3 dataset - containing tiles videos
               preprocess_datasets=json.dumps([PREPROCESS_DATASET]),
               training_container=json.dumps(
                   {'image': 'docker.io/ocdr/dkube-datascience-tf-gpu:v1.12'}),
               training_program=WORKSPACE,
               training_script="python model.py",
               training_gpus=1,
               training_envs=json.dumps([{"steps": STEPS,
                                          "epochs": EPOCHS,
                                          "batchsize": BATCHSIZE}])):

    # generate random preprocess_target_name
    preprocess_target_name = "{}-{}".format(preprocess_target_name,
                                            str(int(time.time())))

    # preprocessing stage
    preprocess = dkube_preprocess_op(auth_token, preprocess_target_name,
                                     preprocess_container,
                                     program=preprocess_program,
                                     datasets=preprocess_datasets,
                                     run_script=preprocess_script,
                                     access_url=access_url)

    # training stage
    preprocess_dataset_name = json.dumps([preprocess_target_name])
    train = dkube_training_op(auth_token, training_container,
                              program=training_program,
                              run_script=training_script,
                              datasets=preprocess_dataset_name,
                              ngpus=training_gpus,
                              envs=training_envs,
                              access_url=access_url).after(preprocess)
    # serving stage
    serving = dkube_serving_op(
        auth_token, train.outputs['artifact'],
        access_url=access_url).after(train)
    # inference stage
    inference = dkube_viewer_op(
        auth_token, serving.outputs['servingurl'],
        SERVING_EXAMPLE, viewtype='inference',
        access_url=access_url).after(serving)


if __name__ == '__main__':
    import kfp.compiler as compiler
    compiler.Compiler().compile(d3pipeline, __file__ + '.tar.gz')
