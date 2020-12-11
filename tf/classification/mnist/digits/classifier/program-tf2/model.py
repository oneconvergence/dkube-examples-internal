# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""Runs a simple model on the MNIST dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

## Importing libraries
import os
import requests
import json
import dataset_preprocessing
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import argparse


if os.getenv('DKUBE_JOB_CLASS',None) == 'notebook':
  if not os.path.exists('/opt/dkube/output'):
    os.makedirs('/opt/dkube/output')
  DATA_DIR = '/opt/dkube/input'
  MODEL_DIR = '/opt/dkube/output'

MODEL_DIR = "/opt/dkube/output"
DATA_DIR = "/opt/dkube/input"
BATCH_SIZE = int(os.getenv('BATCHSIZE',1024))
EPOCHS = int(os.getenv('EPOCHS', 4))
num_train_examples = 60000
num_eval_examples = 10000

def logging_metrics(key, value, step, epoch):
    url = "http://dkube-exporter.dkube:9401/mlflow-exporter"
    train_metrics = {}
    train_metrics['mode']="train"
    train_metrics['key'] = key
    train_metrics['value'] = value
    train_metrics['step'] = step
    train_metrics['epoch'] = epoch
    train_metrics['jobid']=os.getenv('DKUBE_JOB_ID')
    train_metrics['run_id']=os.getenv('DKUBE_JOB_UUID')
    train_metrics['username']=os.getenv('DKUBE_USER_LOGIN_NAME')
    requests.post(url, json = train_metrics)


def build_model():
  """Constructs the ML model used to predict handwritten digits."""

  image = tf.keras.layers.Input(shape=(28, 28, 1),name='input')

  y = tf.keras.layers.Conv2D(filters=32,
                             kernel_size=5,
                             padding='same',
                             activation='relu')(image)
  y = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                   strides=(2, 2),
                                   padding='same')(y)
  y = tf.keras.layers.Conv2D(filters=32,
                             kernel_size=5,
                             padding='same',
                             activation='relu')(y)
  y = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                   strides=(2, 2),
                                   padding='same')(y)
  y = tf.keras.layers.Flatten()(y)
  y = tf.keras.layers.Dense(1024, activation='relu')(y)
  y = tf.keras.layers.Dropout(0.4)(y)

  probs = tf.keras.layers.Dense(10, activation='softmax',name='output')(y)

  model = tf.keras.models.Model(inputs=image, outputs=probs)

  return model



def start_mnist(flags_obj):
  """Run MNIST model training and eval loop using native Keras APIs."""
  ### Getting the data ##
  train_input_dataset,eval_input_dataset=dataset_preprocessing.get_final_data(flags_obj.batch_size)
  model = build_model()
  model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])
  train_steps = num_train_examples // flags_obj.batch_size
  train_epochs = flags_obj.num_epochs
  ckpt_full_path = os.path.join(MODEL_DIR, 'model.ckpt-{epoch:04d}')
  callbacks = [
      tf.keras.callbacks.ModelCheckpoint(
          ckpt_full_path, save_weights_only=True),
      tf.keras.callbacks.TensorBoard(log_dir=MODEL_DIR),
  ]

  
  num_eval_steps = num_eval_examples // flags_obj.batch_size

  history = model.fit(
      train_input_dataset,
      epochs=train_epochs,
      steps_per_epoch=train_steps,
      callbacks=callbacks,
      validation_steps=num_eval_steps,
      validation_data=eval_input_dataset,
      validation_freq=True)

  export_path = os.path.join(MODEL_DIR,'1')
  model.save(export_path, include_optimizer=False)

  eval_output = model.evaluate(eval_input_dataset, steps=num_eval_steps, verbose=2)
  
  step=1
  for epoch in range(0,flags_obj.num_epochs):
    logging_metrics('loss',history.history["loss"][epoch].item(),step,epoch+1)
    logging_metrics('accuracy',history.history["sparse_categorical_accuracy"][epoch].item(),step,epoch+1)
    step=step+1


def main():
    try:
      fp = open(os.getenv('DKUBE_JOB_HP_TUNING_INFO_FILE', 'None'),'r')
      hyperparams = json.loads(fp.read())
      hyperparams['num_epochs'] = EPOCHS
    except:
      hyperparams = {"batch_size": BATCH_SIZE, "num_epochs": EPOCHS }
      pass
    parser = argparse.ArgumentParser(description='Tensorflow MNIST Example')
    parser.add_argument('--batch_size', type=int, default=int(hyperparams['batch_size']),
                        help='input batch size for training (default: 1024)')
    parser.add_argument('--num_epochs', type=int, default=int(hyperparams['num_epochs']),
                        help='number of epochs to train (default: 4)')
    
    flags_obj,unparsed=parser.parse_known_args()
    start_mnist(flags_obj)

if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  main()
