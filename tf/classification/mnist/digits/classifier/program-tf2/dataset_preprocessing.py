import numpy as np
import tensorflow as tf
from six.moves import urllib
import gzip

_MNIST_IMAGE_SIZE = 28
_MNIST_NUM_CLASSES = 10
_TRAIN_EXAMPLES = 60000
_TEST_EXAMPLES = 10000

DATA_DIR='/opt/dkube/input'

def _extract_mnist_images(image_filepath, num_images):
    f = gzip.open(image_filepath,'r')
    f.read(16)  # header
    buf = f.read(_MNIST_IMAGE_SIZE * _MNIST_IMAGE_SIZE * num_images)
    data = np.frombuffer(
        buf,
        dtype=np.uint8,
    ).reshape(num_images, _MNIST_IMAGE_SIZE, _MNIST_IMAGE_SIZE, 1)
    return data


def _extract_mnist_labels(labels_filepath, num_labels):
    f = gzip.open(labels_filepath,'r')
    f.read(8)  # header
    buf = f.read(num_labels)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels

def get_final_data(batch_size):  
  train_data=_extract_mnist_images(DATA_DIR+'/train-images-idx3-ubyte.gz',_TRAIN_EXAMPLES)
  test_data=_extract_mnist_images(DATA_DIR+'/t10k-images-idx3-ubyte.gz',_TEST_EXAMPLES)
  x_train , x_test = (np.array(train_data, np.float32),np.array(test_data, np.float32))
  y_train=_extract_mnist_labels(DATA_DIR+'/train-labels-idx1-ubyte.gz',_TRAIN_EXAMPLES)
  y_test=_extract_mnist_labels(DATA_DIR+'/t10k-labels-idx1-ubyte.gz',_TEST_EXAMPLES)
  train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  train_data = train_data.cache().repeat().shuffle(5000).batch(batch_size)
  test_data = tf.data.Dataset.from_tensor_slices((x_test,y_test))
  test_data = test_data.repeat().batch(batch_size)
  return train_data,test_data
