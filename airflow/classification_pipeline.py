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
"""Airflow Pipeline Example"""

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

from tfx.orchestration.airflow.airflow_runner import AirflowDAGRunner
from tfx.orchestration.pipeline import Pipeline
from tfx.utils.dsl_utils import csv_input
# pylint: enable=line-too-long

PIPELINE_NAME = "complaint_classifier"

project_root_path = os.path.join(
    os.environ['HOME'], 'airflow')  # TODO: change path
data_root_path = os.path.join(project_root_path, 'data/')

# Python module file to inject customized logic into the TFX components. The
# Transform and Trainer both require user-defined functions to run successfully.
pipeline_module_file = os.path.join(project_root_path, 'dags/utils.py')

# Path which can be listened to by the model server.  Pusher will output the
# trained model here.
serving_model_dir = os.path.join(
    project_root_path, 'saved_models/complaint_model')

# Directory and data locations.  This example assumes all of the complaint model
# example code and metadata library is relative to $HOME, but you can store
# these files anywhere on your local filesystem.
tfx_root = os.path.join(project_root_path, 'tfx')  # TODO: Update path
pipeline_root = os.path.join(tfx_root, 'pipelines')
metadata_db_root = os.path.join(tfx_root, 'metadata')
log_root = os.path.join(tfx_root, 'logs')

# Logging overrides
logger_overrides = {
    'log_root': log_root,
    'log_level': logging.INFO
}

# Airflow-specific configs; these will be passed directly to airflow
_airflow_config = {
    'pipeline_name': PIPELINE_NAME,
    'schedule_interval': None,
    'start_date': datetime.datetime(2019, 1, 1),
    'pipeline_root': pipeline_root,
    'metadata_db_root': metadata_db_root,
    'additional_pipeline_args': {'logger_args': logger_overrides},
    'enable_cache': True
}

examples = csv_input(data_root_path)

# Brings data into the pipeline or otherwise joins/converts training data.
example_gen = CsvExampleGen(input_base=examples)

# Computes statistics over data for visualization and example validation.
# pylint: disable=line-too-long
statistics_gen = StatisticsGen(input_data=example_gen.outputs.examples)
# pylint: enable=line-too-long

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

# Uses user-provided Python function that implements a model using TF-Learn.
trainer = Trainer(  # Step 5
    module_file=pipeline_module_file,  # Step 5
    transformed_examples=transform.outputs.transformed_examples,  # Step 5
    schema=infer_schema.outputs.output,  # Step 5
    transform_output=transform.outputs.transform_output,  # Step 5
    train_args=trainer_pb2.TrainArgs(num_steps=10000),  # Step 5
    eval_args=trainer_pb2.EvalArgs(num_steps=5000))  # Step 5

# Uses TFMA to compute a evaluation statistics over features of a model.
model_analyzer = Evaluator(  # Step 6
    examples=example_gen.outputs.examples,  # Step 6
    model_exports=trainer.outputs.output,  # Step 6
    feature_slicing_spec=evaluator_pb2.FeatureSlicingSpec(specs=[  # Step 6
        evaluator_pb2.SingleSlicingSpec(  # Step 6
            column_for_slicing=['Company'])  # Step 6
    ]))  # Step 6

# Performs quality validation of a candidate model (compared to a baseline).
model_validator = ModelValidator(  # Step 7
    examples=example_gen.outputs.examples,  # Step 7
    model=trainer.outputs.output)  # Step 7

# Checks whether the model passed the validation steps and pushes the model
# to a file destination if check passed.
pusher = Pusher(  # Step 7
    model_export=trainer.outputs.output,  # Step 7
    model_blessing=model_validator.outputs.blessing,  # Step 7
    push_destination=pusher_pb2.PushDestination(  # Step 7
        filesystem=pusher_pb2.PushDestination.Filesystem(  # Step 7
            base_directory=serving_model_dir)))  # Step 7


components = [
    example_gen,
    statistics_gen, infer_schema, validate_stats,
    transform,
    trainer,
    model_analyzer,  # Step 6
    model_validator, pusher  # Step 7
]
_pipeline = Pipeline(
    pipeline_name='complaint_classification',
    pipeline_root=pipeline_root,
    components=components,
)

pipeline = AirflowDAGRunner(_airflow_config).run(_pipeline)
