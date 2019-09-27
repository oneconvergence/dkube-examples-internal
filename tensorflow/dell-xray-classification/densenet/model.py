from __future__ import print_function

import os
import argparse
import pickle
import numpy as np
import PIL.Image as pil
import PIL.ImageOps
import keras
import tensorflow as tf
from keras.applications import DenseNet121
from keras.utils import multi_gpu_model
from keras.models import Model
from keras.applications.densenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.layers import GlobalAveragePooling2D, Dense
from keras import optimizers
#from auc_callback import AucRoc
#import horovod.keras as hvd
import time
from sklearn.metrics import roc_auc_score


FLAGS = None
DATUMS_PATH = os.getenv('DATUMS_PATH', None)
DATASET_NAME = os.getenv('DATASET_NAME', None)
MODEL_DIR = os.getenv('OUT_DIR', None)
BATCH_SIZE = int(os.getenv('TF_BATCH_SIZE', 32))
EPOCHS = int(os.getenv('TF_EPOCHS', 1))
TF_TRAIN_STEPS = int(os.getenv('TF_TRAIN_STEPS', 1000))
DATASET_DIR = "{}/{}".format(DATUMS_PATH, DATASET_NAME)
EXTRACT_PATH = "/tmp/dataset"
DATA_DIR =  None


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
            print(roc_auc_score(np.asarray(labels)
                                [:, i], np.asarray(probs)[:, i]))

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


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
    with open(train_label, mode='r') as infile:
        reader = csv.reader(infile)
        reader.next()
        training_labels = {rows[0]:rows[3:] for rows in reader}
        training_files = np.asarray(list(training_labels.keys()))

    with open(validation_label, mode='r') as infile:
        reader = csv.reader(infile)
        reader.next()
        validation_labels = {rows[0]:rows[3:] for rows in reader}
        validation_files = np.asarray(list(validation_labels.keys()))

    #labels = dict(training_labels.items() +  validation_labels.items())
    labels = dict(list(training_labels.items()) +
                  list(validation_labels.items()))
    return labels, training_files, validation_files


def load_batch(batch_of_files, labels, is_training=False):
    batch_images = []
    batch_labels = []
    for filename in batch_of_files:
        img = pil.open(os.path.join(FLAGS.data_dir, filename))
        img = img.convert('RGB')
        img = img.resize((FLAGS.image_size, FLAGS.image_size), pil.NEAREST)
        if is_training and np.random.randint(2):
            img = PIL.ImageOps.mirror(img)
        batch_images.append(np.asarray(img))
        batch_labels.append(labels[filename])
    return preprocess_input(np.float32(np.asarray(batch_images))), np.asarray(batch_labels)


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


def main():

    train_label = FLAGS.train_label
    validation_label = FLAGS.validation_label
    labels, training_files, validation_files = load_train_valid_labels(
        train_label, validation_label)

    #hvd.init()

    #np.random.seed(hvd.rank())

    # Horovod: print logs on the first worker.
    #verbose = 2 if hvd.rank() == 0 else 0
    verbose = 2

    print("Running with the following config:")
    for item in FLAGS.__dict__.items():
        print('%s = %s' % (item[0], str(item[1])))

    base_model = DenseNet121(include_top=False,
                             weights='imagenet',
                             input_shape=(FLAGS.image_size, FLAGS.image_size, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(14, activation='sigmoid', bias_initializer='ones')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    opt = optimizers.Adam(lr=FLAGS.lr)

    #hvd_opt = hvd.DistributedOptimizer(opt)

    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    # Path to weights file
    weights_file = "/tmp/model.h5"

    # Callbacks
    steps_per_epoch = 77871 // FLAGS.batch_size
    val_steps = 8653 // FLAGS.batch_size
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.1, epsilon=0.01,
                                   cooldown=0, patience=1, min_lr=1e-15, verbose=2)
    auc = AucRoc(val_generator(val_steps, validation_files, labels), val_steps)
    model_checkpoint = ModelCheckpoint(weights_file, monitor="val_loss", save_best_only=True,
                                       save_weights_only=True, verbose=2)

    tf_summary_logs_path = FLAGS.model_dir + "/logs"
    callbacks = [
        #hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        #hvd.callbacks.MetricAverageCallback(),
        keras.callbacks.TensorBoard(
            log_dir=tf_summary_logs_path, histogram_freq=0, batch_size=64),
        lr_reducer
    ]

    # if hvd.rank() == 0:
    #     # callbacks.append(auc)
    #     callbacks.append(model_checkpoint)
    callbacks.append(model_checkpoint)

    start_time = time.time()
    # specify training params and start training
    model.fit_generator(
        train_generator(steps_per_epoch, training_files, labels),
        steps_per_epoch=steps_per_epoch,
        epochs=FLAGS.epochs,
        validation_data=val_generator(
            3 * val_steps, validation_files, labels),
        validation_steps=3 * val_steps,
        callbacks=callbacks,
        verbose=verbose)
    end_time = time.time()
    print("start time: {} , end time: {} , elapsed time: {}".format(
        start_time, end_time, end_time-start_time))

    # The export path contains the name and the version of the model
    tf.keras.backend.set_learning_phase(0) # Ignore dropout at inference
    model = tf.keras.models.load_model(weights_file)
    export_path = FLAGS.model_dir + "/1"
    tf.gfile.MkDir(export_path)

    # Fetch the Keras session and save the model
    # The signature definition is defined by the input and output tensors
    # And stored with the default serving key
    with tf.keras.backend.get_session() as sess:
        tf.saved_model.simple_save(
            sess,
            export_path,
            inputs={'input_image': model.input},
            outputs={t.name:t for t in model.outputs})

    #copy weight file
    #tf.gfile.Copy(weights_file, FLAGS.model_dir + "model.h5")


if __name__ == '__main__':
    global FLAGS
    ZIP_FILE = DATASET_DIR + "/data.zip"
    if os.path.exists(ZIP_FILE):
        print("Extracting compressed training data...")
        archive = zipfile.ZipFile(ZIP_FILE)
        for file in archive.namelist():
            if file.startswith('data'):
                archive.extract(file, EXTRACT_PATH)
        print("Training data successfuly extracted")
        global DATA_DIR
        DATA_DIR = EXTRACT_PATH + "/data"

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_dir', type=str, default=DATA_DIR + "/images",
        help='The directory where the input data is stored.')

    parser.add_argument(
        '--model_dir', type=str, default=MODEL_DIR,
        help='The directory where the model will be stored.')

    parser.add_argument(
        '--batch_size', type=int, default=BATCH_SIZE,
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
        '--epochs', type=int, default=15,
        help='Number of epochs to train')

    parser.add_argument(
        '--train_label', type=str, default="./train.csv",
        help='Path to the training label file')

    parser.add_argument(
        '--validation_label', type=str, default="./valiadtion.csv",
        help='Path to the validation label file')
    FLAGS, _ = parser.parse_known_args()

    main()
