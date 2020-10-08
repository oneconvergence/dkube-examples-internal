import os
import json
import tensorflow as tf
import tensorflow_hub as hub
import zipfile
import tarfile
import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
import requests

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", dest = 'epochs', type = int, default = 10, help="no. of epochs")
args = parser.parse_args()
epochs = args.epochs

DATA_DIR = "/opt/dkube/input/"
MODEL_DIR = "/opt/dkube/output/"
EXTRACT_PATH = "/tmp/data/"
ZIP_FILE = DATA_DIR + "data.zip"
img_shape = (298,298)

def log_metrics(key, value):
    url = "http://dkube-exporter.dkube:9401/mlflow-exporter"
    train_metrics = {}
    train_metrics['mode']="train"
    train_metrics['key'] = key
    train_metrics['value'] = value
    train_metrics['epoch'] = 1
    train_metrics['step'] = 1
    train_metrics['jobid']=os.getenv('DKUBE_JOB_ID')
    train_metrics['run_id']=os.getenv('DKUBE_JOB_UUID')
    train_metrics['username']=os.getenv('DKUBE_USER_LOGIN_NAME')
    requests.post(url, json = train_metrics)

if os.path.exists(ZIP_FILE):
    print("Extracting compressed training data...")
    archive = zipfile.ZipFile(ZIP_FILE)
    for file in archive.namelist():
        if file.startswith('data'):
            archive.extract(file, EXTRACT_PATH)
    print("Training data successfuly extracted")
    DATA_DIR = EXTRACT_PATH + "/data"
    
datagen = ImageDataGenerator(rescale=1.0/255.0)
train_it = datagen.flow_from_directory(DATA_DIR + '/train/', class_mode='binary', batch_size=64, target_size=img_shape)
test_it = datagen.flow_from_directory(DATA_DIR + '/valid/', class_mode='binary', batch_size=64, target_size=img_shape)

model = ResNet50(include_top=False, input_shape=(298, 298, 3))
for layer in model.layers:
    layer.trainable = False
flat1 = Flatten()(model.layers[-1].output)
class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
output = Dense(1, activation='sigmoid')(class1)
model = Model(inputs=model.inputs, outputs=output)

opt = SGD(lr=0.001, momentum=0.9)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit_generator(train_it, steps_per_epoch=len(train_it), epochs=epochs, verbose=1)

if 'acc' in history.history.keys():
    log_metrics('accuracy', float(history.history['acc'][-1]))
    log_metrics('loss', float(history.history['loss'][-1]))
else:
    log_metrics('accuracy', float(history.history['accuracy'][-1]))
    log_metrics('loss', float(history.history['loss'][-1]))

export_path = MODEL_DIR
version = 0
if not tf.io.gfile.exists(export_path):
    tf.io.gfile.makedirs(export_path)
model_contents = tf.io.gfile.listdir(export_path)

saved_models = []
for mdir in model_contents:
    if mdir != 'logs' and mdir != 'metrics':
        saved_models.append(int(mdir))
if len(saved_models) < 1:
    version = 1
else:
    version = max(saved_models) + 1
model.save(export_path + 'weights.h5')
tf.keras.backend.set_learning_phase(0)  # Ignore dropout at inference

if '1.1' in tf.__version__:
    with tf.keras.backend.get_session() as sess:
        tf.saved_model.simple_save(
            sess,
            export_path + str(version),
            inputs={'input': model.input},
            outputs={'output': model.output})
elif '2.' in tf.__version__:
    with tf.compat.v1.keras.backend.get_session() as sess:
        tf.compat.v1.saved_model.simple_save(
            sess,
            export_path + str(version),
            inputs={'input': model.input},
            outputs={'output': model.output})
print("Model saved, version = ", version)