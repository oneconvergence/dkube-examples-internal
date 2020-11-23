import tensorflow_datasets as tfds
DATA_DIR = '/opt/dkube/input'
mnist = tfds.builder('mnist',data_dir=DATA_DIR)
mnist.download_and_prepare()
