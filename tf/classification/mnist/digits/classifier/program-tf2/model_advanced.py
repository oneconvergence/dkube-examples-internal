import os
import mlflow
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

DATA_DIR = '/opt/dkube/input'
MODEL_DIR = '/opt/dkube/output'
batch_size = 32
epochs = 5

### Preparing and loading the dataset from DKube ###
mnist = np.load(DATA_DIR+'/mnist.npz')

def get_dataset(flag):
    x_train=mnist['x_train']
    x_test=mnist['x_test']
    y_train=mnist['y_train']
    y_test=mnist['y_test']
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train[..., tf.newaxis].astype("float32")
    x_test = x_test[..., tf.newaxis].astype("float32")
    if flag=='training':
        return x_train,y_train
    if flag=='testing':
        return x_test,y_test

def train_dataset():
    x_train,y_train=get_dataset('training')
    return (tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size))

def test_dataset():
    x_test,y_test=get_dataset('testing')
    return(tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32))

### Building the Model ###

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10,activation='softmax')

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

### Create an instance of the model ###
model = MyModel()
### Choose an optimizer and loss function for training ###
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
### Select the metrics ###
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
### Use tf.GradientTape to train the model ###
@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)
    
### Test the model ###
@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)
  test_loss(t_loss)
  test_accuracy(labels, predictions)

model.compile(optimizer=optimizer,loss = loss_object,metrics = train_accuracy)
### calling the functions ###

train_ds=train_dataset()
test_ds=test_dataset()

for epoch in range(epochs):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

  for images, labels in train_ds:
    train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)
  
  mlflow.log_metric("train_accuracy",train_accuracy.result().numpy())
  mlflow.log_metric("train_loss",train_loss.result().numpy())
  mlflow.log_metric("test_accuracy",test_accuracy.result().numpy())
  mlflow.log_metric("test_loss",test_loss.result().numpy())

export_path = os.path.join(MODEL_DIR,'1')
model.save(export_path, include_optimizer=True)
