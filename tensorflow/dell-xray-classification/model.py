from __future__ import print_function

import os
import sys
import argparse
import numpy as np
import PIL.Image as pil
import PIL.ImageOps
import keras
import zipfile
import requests
import json
import threading
import tensorflow as tf
import pandas as pd
from keras.applications import DenseNet121
from keras.models import Model
from keras.applications.densenet import preprocess_input
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.layers import GlobalAveragePooling2D, Dense
from keras import optimizers
from keras import backend as K
from keras.models import load_model
from keras.utils import multi_gpu_model
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model.signature_def_utils import predict_signature_def
from tensorflow.python.saved_model import tag_constants
# from auc_callback import AucRoc
# import horovod.keras as hvd
import time
from sklearn.metrics import roc_auc_score

DATUMS_PATH = os.getenv('DATUMS_PATH', None)
DATASET_NAME = os.getenv('DATASET_NAME', None)
MODEL_DIR = os.getenv('OUT_DIR', None)
BATCH_SIZE = int(os.getenv('TF_BATCH_SIZE', 32))
EPOCHS = int(os.getenv('TF_EPOCHS', 1))
TF_TRAIN_STEPS = int(os.getenv('TF_TRAIN_STEPS', 1000))
DATASET_DIR = "{}/{}".format(DATUMS_PATH, DATASET_NAME)
EXTRACT_PATH = "/tmp/dataset"
USE_COLUMNS = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
CLASS_NAMES = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
               "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
               "Edema", "Emphysema", "Fibrosis",
               "Pleural_Thickening", "Hernia"]


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


class AucRoc(keras.callbacks.Callback):
    def __init__(self, val_generator, steps):
        self.val_generator = val_generator
        self.steps = steps

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        labels = []
        probs = []
        for i in range(self.steps):
            batch_images, batch_labels = next(self.val_generator)
            probs.extend(self.model.predict_on_batch(batch_images))
            labels.extend(batch_labels)
        for i in range(14):
            roc_auc = roc_auc_score(np.asarray(labels)[:, i],
                                    np.asarray(probs)[:, i])
            print("ROC AUC for {} = {}".format(CLASS_NAMES[i], roc_auc))
            #tf.summary.scalar("roc_auc", roc_auc, family=CLASS_NAMES[i])

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


def dkubeLoggerHook(epoch, logs):
    train_metrics = {
        'mode': "train",
        'accuracy': logs.get('acc', 0),
        'loss': logs.get('loss', 0),
        'epoch': int(epoch),
        'jobid': os.getenv('JOBID'),
        'username': os.getenv('USERNAME')
    }
    eval_metrics = {
        'mode': "eval",
        'accuracy': logs.get('val_acc', 0),
        'loss': logs.get('val_loss', 0),
        'epoch': int(epoch),
        'jobid': os.getenv('JOBID'),
        'username': os.getenv('USERNAME')
    }
    try:
        url = "http://dkube-ext.dkube:9401/export-training-info"
        requests.post(url, data=json.dumps({'data': [train_metrics]}))
        requests.post(url, data=json.dumps({'data': [eval_metrics]}))
    except Exception as exc:
        print(exc)


# def load_train_valid_labels(train_label, validation_label):

#     with open(train_label+'/training_labels_new.pkl', 'rb') as f:
#         training_labels = pickle.load(f)
#     training_files = np.asarray(list(training_labels.keys()))

#     with open(FLAGS.validation_label+'/validation_labels_new.pkl', 'rb') as f:
#         validation_labels = pickle.load(f)
#     validation_files = np.asarray(list(validation_labels.keys()))
#     #labels = dict(training_labels.items() +  validation_labels.items())
#     labels = dict(list(training_labels.items()) +
#                   list(validation_labels.items()))
#     return labels, training_files, validation_files
def load_train_valid_labels(train_label, validation_label):
    training_labels = dict()
    validation_labels = dict()
    training_files = []
    validation_files = []
    train_df = pd.read_csv(train_label, usecols=USE_COLUMNS)
    val_df = pd.read_csv(validation_label, usecols=USE_COLUMNS)
    for index, row in train_df.iterrows():
        training_labels.update({row['Image Index']: (row.values)[1:]})
        training_files.append(row['Image Index'])
    for index, row in val_df.iterrows():
        validation_labels.update({row['Image Index']: (row.values)[1:]})
        validation_files.append(row['Image Index'])

    #labels = dict(training_labels.items() +  validation_labels.items())
    labels = dict(list(training_labels.items()) +
                  list(validation_labels.items()))
    return labels, training_files, validation_files


def load_batch(batch_of_files, labels, is_training=False):
    batch_images = []
    batch_labels = []
    for filename in batch_of_files:
        try:
            image_path = os.path.join(FLAGS.data_dir, filename)
            if os.path.exists(image_path):
                img = pil.open(image_path)
                img = img.convert('RGB')
                img = img.resize((FLAGS.image_size, FLAGS.image_size),
                                 pil.NEAREST)
                if is_training and np.random.randint(2):
                    img = PIL.ImageOps.mirror(img)
                batch_images.append(np.asarray(img))
                batch_labels.append(
                    np.asarray(labels[filename], dtype=np.float32))
        except Exception:
            pass
    return preprocess_input(np.float32(np.asarray(batch_images))), np.asarray(batch_labels)


@threadsafe_generator
def train_generator(num_of_steps, training_files, labels):
    while True:
        np.random.shuffle(training_files)
        batch_size = FLAGS.batch_size
        for i in range(num_of_steps):
            batch_of_files = training_files[i *
                                            batch_size: i*batch_size + batch_size]
            batch_images, batch_labels = load_batch(
                batch_of_files, labels, True)
            yield batch_images, batch_labels

@threadsafe_generator
def val_generator(num_of_steps, validation_files, labels):
    while True:
        np.random.shuffle(validation_files)
        batch_size = FLAGS.batch_size
        for i in range(num_of_steps):
            batch_of_files = validation_files[i *
                                              batch_size: i*batch_size + batch_size]
            batch_images, batch_labels = load_batch(
                batch_of_files, labels, True)
            yield batch_images, batch_labels


def export_h5_to_pb(model_file):
    export_path = FLAGS.model_dir + "/1"
    # Set the learning phase to Test since the model is already trained.
    K.set_learning_phase(0)

    # Load the Keras model
    keras_model = load_model(model_file)

    # Build the Protocol Buffer SavedModel at 'export_path'
    builder = saved_model_builder.SavedModelBuilder(export_path)

    # Create prediction signature to be used by TensorFlow Serving Predict API
    signature = predict_signature_def(
        inputs={"inputs": keras_model.input},
        outputs={"predictions": keras_model.output})

    with K.get_session() as sess:
        # Save the meta graph and the variables
        builder.add_meta_graph_and_variables(
            sess=sess,
            tags=[tag_constants.SERVING],
            signature_def_map={"serving_default": signature})

    builder.save()


def main():

    train_label = FLAGS.train_label
    validation_label = FLAGS.validation_label
    labels, training_files, validation_files = load_train_valid_labels(
        train_label, validation_label)

    # hvd.init()

    # np.random.seed(hvd.rank())

    # Horovod: print logs on the first worker.
    # verbose = 2 if hvd.rank() == 0 else 0
    verbose = 2

    print("Running with the following config:")
    for item in FLAGS.__dict__.items():
        print('%s = %s' % (item[0], str(item[1])))

    keras.backend.get_session().run(tf.global_variables_initializer())
    num_gpus = int(FLAGS.ngpus)

    if num_gpus > 1:
        with tf.device('/cpu:0'):
            base_model = DenseNet121(include_top=False,
                                     weights='imagenet',
                                     input_shape=(FLAGS.image_size, FLAGS.image_size, 3))
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            predictions = Dense(14, activation='sigmoid', bias_initializer='ones')(x)
            model = Model(inputs=base_model.input, outputs=predictions)
        parallel_model = multi_gpu_model(model, gpus=num_gpus,
                                         cpu_merge=True, cpu_relocation=False)
    else:
        base_model = DenseNet121(include_top=False,
                                 weights='imagenet',
                                 input_shape=(FLAGS.image_size, FLAGS.image_size, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(14, activation='sigmoid', bias_initializer='ones')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        parallel_model = model

    opt = optimizers.Adam(lr=FLAGS.lr)

    # hvd_opt = hvd.DistributedOptimizer(opt)

    parallel_model.compile(loss='binary_crossentropy',
                           optimizer=opt,
                           metrics=['accuracy'])
    # Path to weights file
    weights_file = "/tmp/weights.h5"

    # Callbacks
    # steps_per_epoch = 77871 // FLAGS.batch_size
    # val_steps = 8653 // FLAGS.batch_size
    steps_per_epoch = 104266 // FLAGS.batch_size
    val_steps = 6336 // FLAGS.batch_size
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                   epsilon=0.01, cooldown=0, patience=1,
                                   min_lr=1e-15, verbose=2)
    tf_summary_logs_path = FLAGS.model_dir + "/logs"
    file_writer = tf.contrib.summary.create_file_writer(tf_summary_logs_path)
    file_writer.set_as_default()
    auc = AucRoc(val_generator(val_steps, validation_files, labels), val_steps)
    model_checkpoint = ModelCheckpoint(weights_file, monitor="val_loss",
                                       save_best_only=True,
                                       save_weights_only=True, verbose=2)
    dkube_logger_callback = keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: dkubeLoggerHook(epoch, logs))

    callbacks = [
        # hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        # hvd.callbacks.MetricAverageCallback(),
        TensorBoard(log_dir=tf_summary_logs_path,
                    histogram_freq=0,
                    update_freq='epoch'),
        lr_reducer,
        auc,
        dkube_logger_callback
    ]

    # if hvd.rank() == 0:
    #     # callbacks.append(auc)
    #     callbacks.append(model_checkpoint)
    callbacks.append(model_checkpoint)

    start_time = time.time()

    workers = num_gpus
    print("Running batch generator with {} workers".format(str(workers)))
    # specify training params and start training
    parallel_model.fit_generator(
        train_generator(steps_per_epoch, training_files, labels),
        steps_per_epoch=steps_per_epoch,
        epochs=FLAGS.epochs,
        validation_data=val_generator(
            val_steps, validation_files, labels),
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=verbose,
        workers=workers,
        use_multiprocessing=True,
        max_queue_size=20)

    # Set the learning phase to Test since the model is already trained.
    K.set_learning_phase(0)

    export_path = FLAGS.model_dir + "/1"
    # Build the Protocol Buffer SavedModel at 'export_path'
    builder = saved_model_builder.SavedModelBuilder(export_path)

    # Create prediction signature to be used by TensorFlow Serving Predict API
    signature = predict_signature_def(
        inputs={"inputs": model.get_input_at(0)},
        outputs={"predictions": model.get_output_at(0)})

    with K.get_session() as sess:
        # Save the meta graph and the variables
        builder.add_meta_graph_and_variables(
            sess=sess,
            tags=[tag_constants.SERVING],
            clear_devices=True,
            signature_def_map={"serving_default": signature})

    builder.save()

    end_time = time.time()
    print("start time: {} , end time: {} , elapsed time: {}".format(
          start_time, end_time, end_time - start_time))


if __name__ == '__main__':
    ZIP_FILE = DATASET_DIR + "/data.zip"
    if os.path.exists(ZIP_FILE):
        print("Extracting compressed training data...")
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_PATH)
        print("Training data successfuly extracted")
        DATA_DIR = EXTRACT_PATH + "/data"

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_dir', type=str, default=DATA_DIR + "/images",
        help='The directory where the input data is stored.')

    parser.add_argument(
        '--model_dir', type=str, default=MODEL_DIR,
        help='The directory where the model will be stored.')

    parser.add_argument(
        '--batch_size', type=int, default=16,
        help='Batch size for SGD')

    parser.add_argument(
        '--image_size', type=int, default='256',
        help='Image size')

    parser.add_argument(
        '--opt', type=str, default='adam',
        help='Optimizer to use (adam, sgd, rmsprop, adagrad, adadelta, adamax, nadam)')

    parser.add_argument(
        '--momentum', type=float, default='0.0',
        help='Momentum rate for SGD optimizer')

    parser.add_argument(
        '--nesterov', type=bool, default=False,
        help='Use Nesterov momentum for SGD optimizer')

    parser.add_argument(
        '--lr', type=float, default='1e-3',
        help='Learning rate for optimizer')

    parser.add_argument(
        '--epochs', type=int, default=1,
        help='Number of epochs to train')

    parser.add_argument(
        '--train_label', type=str, default="./train.csv",
        help='Path to the training label file')

    parser.add_argument(
        '--validation_label', type=str, default="./validation.csv",
        help='Path to the validation label file')

    parser.add_argument(
        '--ngpus', type=int, default=1,
        help='Number of gpus')

    FLAGS, _ = parser.parse_known_args()
    FLAGS.epochs = EPOCHS
    FLAGS.batch_size = BATCH_SIZE

    main()
