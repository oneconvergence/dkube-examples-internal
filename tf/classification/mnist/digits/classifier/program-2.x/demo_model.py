import numpy as np
import struct
import os
import pandas as pd
import sys
import argparse
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

inp_path = '/opt/dkube/input/'
out_path = '/opt/dkube/output/'
batch_size = 32

steps_per_epoch = int(60000/batch_size)
epochs = 5

path = os.path.join(inp_path, 'mnist.npz')

def train_dataset():
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
    x_train = x_train[..., tf.newaxis].astype("float32")
    return (
        tf.data.Dataset.from_tensor_slices(dict(x=x_train, y=y_train)).repeat().batch(batch_size)
    )


@tf.function
def train_step(net, example, optimizer):
    """Trains `net` on `example` using `optimizer`."""
    images, labels = example['x'], example['y']
    with tf.GradientTape() as tape:
        output = net(images, training=True)
        loss = loss_object(labels, output)
    variables = net.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    train_loss(loss)
    train_accuracy(labels, output)
    return loss


class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10)

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

# Create an instance of the model
model = MyModel()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

opt = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

dataset = train_dataset()
iterator = iter(dataset)
ckpt = tf.train.Checkpoint(
    step=tf.Variable(1), optimizer=opt, net=model, iterator=iterator
)
manager = tf.train.CheckpointManager(ckpt, os.path.join(out_path, 'run-1'), max_to_keep=3)
ckpt.restore(manager.latest_checkpoint)

steps = steps_per_epoch * epochs

for _ in range(steps):
    example = next(iterator)
    loss = train_step(model, example, opt)
    ckpt.step.assign_add(1)
    if int(ckpt.step) % 100 == 0:
        save_path = manager.save()
        print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
        print("loss {:1.2f}".format(loss.numpy()))
        
model.save(out_path)