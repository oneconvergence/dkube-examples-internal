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

import os
from kfp import onprem

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
from tfx.orchestration.kubeflow.proto import kubeflow_pb2
from tfx.orchestration.kubeflow import kubeflow_dag_runner
from ml_metadata.proto import metadata_store_pb2
from ml_metadata.metadata_store import metadata_store
from typing import Text

# pylint: enable=line-too-long

# Directory and data locations (uses Google Cloud Storage).
_input_bucket = '/book-classification/data/'
_utils_bucket = '/book-classification/tfx/'
_output_bucket = '/book-classification'
_pipeline_root = os.path.join(_output_bucket, 'tfx')
pipeline_name = 'complaint_model_pipeline_kubeflow'

# Python module file to inject customized logic into the TFX components. The
# Transform and Trainer both require user-defined functions to run successfully.
# Copy this from the current directory to a GCS bucket and update the location
# below.
pipeline_module_file = os.path.join(_utils_bucket, 'utils.py')

# Path which can be listened to by the model server.  Pusher will output the
# trained model here.
# _serving_model_dir = os.path.join(
#     _output_bucket, 'serving_model/taxi_bigquery')

# Path which can be listened to by the model server.  Pusher will output the
# trained model here.
_serving_model_dir = os.path.join(
    _output_bucket, 'serving_model/complaint_model')


def _get_kubeflow_metadata_config(pipeline_name: Text
                                  ) -> kubeflow_pb2.KubeflowMetadataConfig:
    config = kubeflow_pb2.KubeflowMetadataConfig()
    config.mysql_db_service_host.value = '10.233.64.94'
    config.mysql_db_service_port.value = '3306'
    config.mysql_db_name.value = _get_mlmd_db_name(pipeline_name)
    config.mysql_db_user.value = 'root'
    config.mysql_db_password.value = ''
    return config

def _get_metadata_store(pipeline_name: Text
                        ) -> metadata_store_pb2.ConnectionConfig:
    config = metadata_store_pb2.ConnectionConfig()
    config.mysql.host = '10.233.64.94'
    config.mysql.port = 3306
    config.mysql.database = _get_mlmd_db_name(pipeline_name)
    config.mysql.user = 'root'
    config.mysql.password = ''
    store = metadata_store.MetadataStore(config)
    return store


def _get_mlmd_db_name(pipeline_name: Text):
    # MySQL DB names must not contain '-' while k8s names must not contain '_'.
    # So we replace the dashes here for the DB name.
    valid_mysql_name = pipeline_name.replace('-', '_')
    # MySQL database name cannot exceed 64 characters.
    return 'mlmd_{}'.format(valid_mysql_name[-59:])


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

    trainer = Trainer(
        module_file=pipeline_module_file,
        transformed_examples=transform.outputs.transformed_examples,
        schema=infer_schema.outputs.output,
        transform_output=transform.outputs.transform_output,
        train_args=trainer_pb2.TrainArgs(num_steps=10000),
        eval_args=trainer_pb2.EvalArgs(num_steps=5000))

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

    pusher = Pusher(
        model_export=trainer.outputs.output,
        model_blessing=model_validator.outputs.blessing,
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=_serving_model_dir)))

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=_pipeline_root,
        metadata_connection_config=_get_metadata_store(pipeline_name),
        components=[
            example_gen, statistics_gen, infer_schema, validate_stats,
            transform, trainer, model_analyzer, model_validator, pusher
        ],
        additional_pipeline_args={
            'beam_pipeline_args': [
                '--runner=DirectRunner',
                '--experiments=shuffle_mode=auto',
                '--temp_location=' + os.path.join(_output_bucket, 'tmp')
            ],
        },
        log_root='/var/tmp/tfx/logs'
    )


mount_volume_op = onprem.mount_pvc('book-classification-claim',
                                   'book-classification',
                                   '/book-classification')
config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
    pipeline_operator_funcs=[mount_volume_op],
    kubeflow_metadata_config=_get_kubeflow_metadata_config(pipeline_name)
)

kubeflow_dag_runner.KubeflowDagRunner(config=config).run(_create_pipeline())
