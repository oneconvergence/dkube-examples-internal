#  Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from dkube import dkubeLoggerHook as logger_hook
import argparse
import os
import sys
import json
import tensorflow as tf
import dataset
import json

tf.compat.v1.disable_eager_execution()

FLAGS = None
TF_TRAIN_STEPS = int(os.getenv('STEPS',1000))
MODEL_DIR = "/opt/dkube/output"
DATA_DIR = "/opt/dkube/input"
BATCH_SIZE = int(os.getenv('BATCHSIZE', 10))
EPOCHS = int(os.getenv('EPOCHS', 1))
TF_MODEL_DIR = MODEL_DIR
steps_epoch = 0
summary_interval = 100
print ("ENV, EXPORT_DIR:{}, DATA_DIR:{}".format(MODEL_DIR, DATA_DIR))
print ("TF_CONFIG: {}".format(os.getenv("TF_CONFIG", '{}')))

def count_epochs(iterator):
    sess = tf.compat.v1.Session()
    if os.getenv('TF_CONFIG') is not None:
        cluster_spec = json.loads(os.getenv('TF_CONFIG',None))
        role = cluster_spec['task']
        host = cluster_spec['cluster'][role['type']][role['index']]
        if len(cluster_spec['cluster'].keys()) > 1:
            sess = tf.compat.v1.Session('grpc://'+ host)
    global steps_epoch
    if not steps_epoch:
        while True:
            try:
                sess.run(iterator)
                steps_epoch += 1
            except Exception as OutOfRangeError:
                if steps_epoch == 0:
                   steps_epoch = TF_TRAIN_STEPS
                steps_epoch /= FLAGS.num_epochs
                break

class Model(object):
  """Class that defines a graph to recognize digits in the MNIST dataset."""

  def __init__(self, data_format):
    """Creates a model for classifying a hand-written digit.

    Args:
      data_format: Either 'channels_first' or 'channels_last'.
        'channels_first' is typically faster on GPUs while 'channels_last' is
        typically faster on CPUs. See
        https://www.tensorflow.org/performance/performance_guide#data_formats
    """
    if data_format == 'channels_first':
      self._input_shape = [-1, 1, 28, 28]
    else:
      assert data_format == 'channels_last'
      self._input_shape = [-1, 28, 28, 1]

    self.conv1 = tf.compat.v1.layers.Conv2D(
        32, 5, padding='same', data_format=data_format, activation=tf.nn.relu)
    self.conv2 = tf.compat.v1.layers.Conv2D(
        64, 5, padding='same', data_format=data_format, activation=tf.nn.relu)
    self.fc1 = tf.compat.v1.layers.Dense(1024, activation=tf.nn.relu)
    self.fc2 = tf.compat.v1.layers.Dense(10)
    self.dropout = tf.compat.v1.layers.Dropout(0.4)
    self.max_pool2d = tf.compat.v1.layers.MaxPooling2D(
        (2, 2), (2, 2), padding='same', data_format=data_format)

  def __call__(self, inputs, training):
    """Add operations to classify a batch of input images.

    Args:
      inputs: A Tensor representing a batch of input images.
      training: A boolean. Set to True to add operations required only when
        training the classifier.

    Returns:
      A logits Tensor with shape [<batch_size>, 10].
    """
    y = tf.reshape(inputs, self._input_shape)
    y = self.conv1(y)
    y = self.max_pool2d(y)
    y = self.conv2(y)
    y = self.max_pool2d(y)
    y = tf.compat.v1.layers.flatten(y)
    y = self.fc1(y)
    y = self.dropout(y, training=training)
    return self.fc2(y)


def model_fn(features, labels, mode, params):
  """The model_fn argument for creating an Estimator."""
  model = Model(params['data_format'])
  image = features
  if isinstance(image, dict):
    image = features['image']

  if mode == tf.estimator.ModeKeys.PREDICT:
    logits = model(image, training=False)
    predictions = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits),
    }
    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.PREDICT,
        predictions=predictions,
        export_outputs={
            'classify': tf.estimator.export.PredictOutput(predictions)
        })
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    logits = model(image, training=True)
    loss = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
    accuracy = tf.compat.v1.metrics.accuracy(
        labels=tf.argmax(input=labels, axis=1), predictions=tf.argmax(input=logits, axis=1))
    # Name the accuracy tensor 'train_accuracy' to demonstrate the
    # LoggingTensorHook.
    tf.identity(accuracy[1], name='train_accuracy')
    tf.compat.v1.summary.scalar('train_accuracy', accuracy[1])
    logging_hook = logger_hook({"loss": loss, "accuracy":accuracy[1] ,
            "step" : tf.compat.v1.train.get_or_create_global_step(), "steps_epoch": steps_epoch, "mode":"train"}, every_n_iter=summary_interval)
    return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=optimizer.minimize(loss, tf.compat.v1.train.get_or_create_global_step()),
            training_hooks = [logging_hook])
  if mode == tf.estimator.ModeKeys.EVAL:
    logits = model(image, training=False)
    loss = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
    accuracy = tf.compat.v1.metrics.accuracy(
        labels=tf.argmax(input=labels, axis=1), predictions=tf.argmax(input=logits, axis=1))
    logging_hook = logger_hook({"loss": loss, "accuracy":accuracy[1] ,
        "step" : tf.compat.v1.train.get_or_create_global_step(), "steps_epoch": steps_epoch, "mode":"eval"}, every_n_iter=summary_interval)
    return tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.EVAL,
                loss=loss,
                eval_metric_ops={
                    'accuracy':
                        tf.compat.v1.metrics.accuracy(
                        labels=tf.argmax(input=labels, axis=1),
                        predictions=tf.argmax(input=logits, axis=1)),
                },
                evaluation_hooks = [logging_hook])

def main(unused_argv):
  try:
      fp = open(os.getenv('DKUBE_JOB_HP_TUNING_INFO_FILE', 'None'),'r')
      hyperparams = json.loads(fp.read())
      hyperparams['num_epochs'] = EPOCHS
  except:
      hyperparams = { "learning_rate":1e-4, "batch_size":BATCH_SIZE, "num_epochs":EPOCHS }
      pass
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type=float, default=float(hyperparams['learning_rate']), help='Learning rate for training.')
  parser.add_argument('--batch_size', type=int, default=int(hyperparams['batch_size']), help='Batch size for training.')
  parser.add_argument('--num_epochs', type=int, default=int(hyperparams['num_epochs']), help='Number of epochs to train for.')
  global FLAGS
  FLAGS, unparsed = parser.parse_known_args()
  data_format = None
  if data_format is None:
    data_format = ('channels_first'
                   if tf.test.is_built_with_cuda() else 'channels_last')
  if DATA_DIR is None:
        print("No input dataset specified. Exiting...")
        return 1
  training_config = tf.estimator.RunConfig(model_dir=TF_MODEL_DIR, save_summary_steps=summary_interval, save_checkpoints_steps=summary_interval)
  mnist_classifier = tf.estimator.Estimator(
      model_fn=model_fn,
      model_dir=MODEL_DIR,
      params={
          'data_format': data_format
      }, config=training_config)

  # Export the model
  if MODEL_DIR is not None:
    image = tf.compat.v1.placeholder(tf.float32, [None, 28, 28])
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        'image': image,
    })
    export_fn = input_fn
    export_final = tf.estimator.FinalExporter(MODEL_DIR, serving_input_receiver_fn=input_fn)

  # Train the model
  def train_input_fn():
    # When choosing shuffle buffer sizes, larger sizes result in better
    # randomness, while smaller sizes use less memory. MNIST is a small
    # enough dataset that we can easily shuffle the full epoch.
    ds = dataset.train(DATA_DIR)
    ds = ds.cache().shuffle(buffer_size=50000).batch(FLAGS.batch_size).repeat(FLAGS.num_epochs)
    (images, labels) = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
    (cimages, clabels) = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
    count_epochs(cimages)
    return (images, labels)

  '''
  # Set up training hook that logs the training accuracy every 100 steps.
  tensors_to_log = {'train_accuracy': 'train_accuracy'}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=100)
  mnist_classifier.train(input_fn=train_input_fn, hooks=[logging_hook])
  '''

  train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn, max_steps=TF_TRAIN_STEPS)

  # Evaluate the model and print results
  def eval_input_fn():
    return tf.compat.v1.data.make_one_shot_iterator(dataset.test(DATA_DIR).batch(FLAGS.batch_size)).get_next()

  eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                      steps=1,
                                      #exporters=export_final,
                                      throttle_secs=1,
                                      start_delay_secs=1)
  tf.estimator.train_and_evaluate(mnist_classifier, train_spec, eval_spec)
  if os.getenv('TF_CONFIG', '') != '':
        config = json.loads(os.getenv('TF_CONFIG'))
        if config['task']['type'] == 'master':
            mnist_classifier.export_saved_model(MODEL_DIR, export_fn)
  else:
        mnist_classifier.export_saved_model(MODEL_DIR, export_fn)

  '''
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print()
  print('Evaluation results:\n\t%s' % eval_results)

  # Export the model
  if FLAGS.export_dir is not None:
    image = tf.placeholder(tf.float32, [None, 28, 28])
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        'image': image,
    })
    mnist_classifier.export_savedmodel(FLAGS.export_dir, input_fn)
 '''

def run():
  global summary_interval
  summary_interval = 100
  if TF_TRAIN_STEPS%100 < 10 and TF_TRAIN_STEPS < 1000:
    summary_interval = TF_TRAIN_STEPS/10
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  tf.compat.v1.app.run(main=main)

if __name__ == '__main__':
    if os.getenv("STEPS") is None:
        os.environ['STEPS'] = str(TF_TRAIN_STEPS)
    run()
