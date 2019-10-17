# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Kubeflow Pipeline Example"""

import datetime
import logging
import os

from tfx.components.example_gen.csv_example_gen.component import CsvExampleGen

# pylint: disable=line-too-long
from tfx.components.statistics_gen.component import StatisticsGen
from tfx.components.schema_gen.component import SchemaGen
from tfx.components.example_validator.component import ExampleValidator

from tfx.components.transform.component import Transform

from tfx.proto import trainer_pb2  # Step 5
from tfx.components.trainer.component import Trainer  # Step 5

from tfx.proto import evaluator_pb2  # Step 6
from tfx.components.evaluator.component import Evaluator  # Step 6

from tfx.proto import pusher_pb2  # Step 7
from tfx.components.model_validator.component import ModelValidator  # Step 7
from tfx.components.pusher.component import Pusher  # Step 7
from tfx.utils.dsl_utils import csv_input

from tfx.orchestration import pipeline
from tfx.orchestration.kubeflow.runner import KubeflowRunner

# pylint: enable=line-too-long

# Directory and data locations (uses Google Cloud Storage).
_input_bucket = 'gs://kf-pipelines/data/'
_utils_bucket = 'gs://kf-pipelines/tfx/'
_output_bucket = 'gs://kf-pipelines/'
_pipeline_root = os.path.join(_output_bucket, 'tfx')

# Google Cloud Platform project id to use when deploying this pipeline.
_project_id = 'oreilly-book'

# Python module file to inject customized logic into the TFX components. The
# Transform and Trainer both require user-defined functions to run successfully.
# Copy this from the current directory to a GCS bucket and update the location
# below.
pipeline_module_file = os.path.join(_utils_bucket, 'utils.py')

# Path which can be listened to by the model server.  Pusher will output the
# trained model here.
# _serving_model_dir = os.path.join(
#     _output_bucket, 'serving_model/taxi_bigquery')

# Region to use for Dataflow jobs and AI Platform training jobs.
#   Dataflow: https://cloud.google.com/dataflow/docs/concepts/regional-endpoints
#   AI Platform: https://cloud.google.com/ml-engine/docs/tensorflow/regions
_gcp_region = 'us-central1'

# Path which can be listened to by the model server.  Pusher will output the
# trained model here.
_serving_model_dir = os.path.join(
    _output_bucket, 'serving_model/complaint_model')

# A dict which contains the training job parameters to be passed to Google
# Cloud AI Platform. For the full set of parameters supported by Google Cloud AI
# Platform, refer to
# https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#Job
_ai_platform_training_args = {
    'project': _project_id,
    'region': _gcp_region,
    'jobDir': os.path.join(_output_bucket, 'tmp'),
    # Starting from TFX 0.14, 'runtimeVersion' is not relevant anymore.
    # Instead, it will be populated by TFX as <major>.<minor> version of
    # the imported TensorFlow package;
    'runtimeVersion': '1.13',
    'runtime_version': '1.13',
    # Starting from TFX 0.14, 'pythonVersion' is not relevant anymore.
    # Instead, it will be populated by TFX as the <major>.<minor> version
    # of the running Python interpreter;
    'pythonVersion': '2.7',
    # 'pythonModule' will be populated by TFX;
    # 'args' will be populated by TFX;
    # If 'packageUris' is not empty, AI Platform trainer will assume this
    # will populate this field with an ephemeral TFX package built from current
    # installation.
}

# A dict which contains the serving job parameters to be passed to Google
# Cloud AI Platform. For the full set of parameters supported by Google Cloud AI
# Platform, refer to
# https://cloud.google.com/ml-engine/reference/rest/v1/projects.models
_ai_platform_serving_args = {
    'model_name': 'complaint_model',
    'project_id': _project_id,
    # 'runtimeVersion' will be populated by TFX as <major>.<minor> version of
    #   the imported TensorFlow package;
}


def _create_pipeline():
    """Implements the complaint model pipeline with TFX and Kubeflow Pipelines."""

    # Brings data into the pipeline or otherwise joins/converts training data.
    # example_gen = BigQueryExampleGen(query=_query)

    examples = csv_input(_input_bucket)

    # Brings data into the pipeline or otherwise joins/converts training data.
    example_gen = CsvExampleGen(input_base=examples)

    # Computes statistics over data for visualization and example validation.
    statistics_gen = StatisticsGen(input_data=example_gen.outputs.examples)

    # Generates schema based on statistics files.
    infer_schema = SchemaGen(stats=statistics_gen.outputs.output)

    # Performs anomaly detection based on statistics and data schema.
    validate_stats = ExampleValidator(
        stats=statistics_gen.outputs.output,
        schema=infer_schema.outputs.output)

    # Performs transformations and feature engineering in training and serving.
    transform = Transform(
        input_data=example_gen.outputs.examples,
        schema=infer_schema.outputs.output,
        module_file=pipeline_module_file)

    # Uses user-provided Python function that implements a model using TF-Learn
    # to train a model on Google Cloud AI Platform.
    try:
        from tfx.extensions.google_cloud_ai_platform.trainer import executor as ai_platform_trainer_executor  # pylint: disable=g-import-not-at-top
        # Train using a custom executor. This requires TFX >= 0.14.
        trainer = Trainer(
            executor_class=ai_platform_trainer_executor.Executor,
            module_file=pipeline_module_file,
            transformed_examples=transform.outputs.transformed_examples,
            schema=infer_schema.outputs.output,
            transform_output=transform.outputs.transform_output,
            train_args=trainer_pb2.TrainArgs(num_steps=10000),
            eval_args=trainer_pb2.EvalArgs(num_steps=5000),
            custom_config={'ai_platform_training_args': _ai_platform_training_args})
    except ImportError:
        # Train using a deprecated flag.
        trainer = Trainer(
            module_file=pipeline_module_file,
            transformed_examples=transform.outputs.transformed_examples,
            schema=infer_schema.outputs.output,
            transform_output=transform.outputs.transform_output,
            train_args=trainer_pb2.TrainArgs(num_steps=10000),
            eval_args=trainer_pb2.EvalArgs(num_steps=5000),
            custom_config={'cmle_training_args': _ai_platform_training_args})

    # Uses TFMA to compute a evaluation statistics over features of a model.
    model_analyzer = Evaluator(
        examples=example_gen.outputs.examples,
        model_exports=trainer.outputs.output,
        feature_slicing_spec=evaluator_pb2.FeatureSlicingSpec(specs=[
            evaluator_pb2.SingleSlicingSpec(
                column_for_slicing=['trip_start_hour'])
        ]))

    # Performs quality validation of a candidate model (compared to a baseline).
    model_validator = ModelValidator(
        examples=example_gen.outputs.examples,
        model=trainer.outputs.output)

    # Checks whether the model passed the validation steps and pushes the model
    # to a destination if check passed.
    try:
        from tfx.extensions.google_cloud_ai_platform.pusher import executor as ai_platform_pusher_executor  # pylint: disable=g-import-not-at-top
        # Deploy the model on Google Cloud AI Platform. This requires TFX >=0.14.
        pusher = Pusher(
            executor_class=ai_platform_pusher_executor.Executor,
            model_export=trainer.outputs.output,
            model_blessing=model_validator.outputs.blessing,
            custom_config={'ai_platform_serving_args': _ai_platform_serving_args})
    except ImportError:
        # Deploy the model on Google Cloud AI Platform, using a deprecated flag.
        pusher = Pusher(
            model_export=trainer.outputs.output,
            model_blessing=model_validator.outputs.blessing,
            custom_config={'cmle_serving_args': _ai_platform_serving_args},
            push_destination=pusher_pb2.PushDestination(
                filesystem=pusher_pb2.PushDestination.Filesystem(
                    base_directory=_serving_model_dir)))

    return pipeline.Pipeline(
        pipeline_name='complaint_model_pipeline_kubeflow',
        pipeline_root=_pipeline_root,
        components=[
            example_gen, statistics_gen, infer_schema, validate_stats,
            transform, trainer, model_analyzer, model_validator, pusher
        ],
        additional_pipeline_args={
            'beam_pipeline_args': [
                '--runner=DataflowRunner',
                '--experiments=shuffle_mode=auto',
                '--project=' + _project_id,
                '--temp_location=' + os.path.join(_output_bucket, 'tmp'),
                '--region=' + _gcp_region,
            ],
            # Optional args:
            # 'tfx_image': custom docker image to use for components.
            # This is needed if TFX package is not installed from an RC
            # or released version.
        },
        log_root='/var/tmp/tfx/logs',
    )


_ = KubeflowRunner().run(_create_pipeline())
