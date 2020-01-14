import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, LambdaCallback
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model.signature_def_utils import predict_signature_def
from tensorflow.python.saved_model import tag_constants
import pandas as pd
import pydicom
import numpy as np
from tensorflow.keras import regularizers
from sklearn.metrics import mean_absolute_error
import argparse
import requests

DATA_DIR = 'data_splits/'
TRAIN_DATA_CLI = DATA_DIR + 'train/CLI/'
TRAIN_DATA_IMG = DATA_DIR + 'train/IMG/'
TRAIN_DATA_RNA = DATA_DIR + 'train/RNA/'

VAL_DATA_CLI = DATA_DIR + 'val/CLI/'
VAL_DATA_IMG = DATA_DIR + 'val/IMG/'
VAL_DATA_RNA = DATA_DIR + 'val/RNA/'

TEST_DATA_CLI = DATA_DIR + 'test/CLI/'
TEST_DATA_IMG = DATA_DIR + 'test/IMG/'
TEST_DATA_RNA = DATA_DIR + 'test/RNA/'

train_df = pd.read_csv(TRAIN_DATA_CLI + "cli_data_processed_train.csv")
val_df = pd.read_csv(VAL_DATA_CLI + "cli_data_processed_val.csv")
test_df = pd.read_csv(TEST_DATA_CLI + "cli_data_processed_test.csv")

Y_train = train_df['days_to_death']
Y_train = np.asarray(Y_train)
Y_train = Y_train.reshape(Y_train.shape[0],1)
train_imgs = list(train_df['bcr_patient_barcode'])

X1_train = train_df.drop(['days_to_death','bcr_patient_barcode'], axis = 1)
X1_train = np.asarray(X1_train)
X1_train = X1_train.reshape(X1_train.shape[0],X1_train.shape[1],1)


Y_val = val_df['days_to_death']
Y_val = np.asarray(Y_val)
Y_val = Y_val.reshape(Y_val.shape[0],1)

X1_val = val_df.drop(['days_to_death','bcr_patient_barcode'], axis = 1)
X1_val = np.asarray(X1_val)
X1_val = X1_val.reshape(X1_val.shape[0],X1_val.shape[1],1)


Y_test = test_df['days_to_death']
Y_test = np.asarray(Y_test)
Y_test = Y_test.reshape(Y_test.shape[0],1)
X1_test = test_df.drop(['days_to_death','bcr_patient_barcode'], axis = 1)
X1_test = np.asarray(X1_test)
X1_test = X1_test.reshape(X1_test.shape[0],X1_test.shape[1],1)


X2_train = []
X2_val = []
X2_test = []
train_imgs = tf.io.gfile.listdir(TRAIN_DATA_IMG)
for each_img in train_imgs:
    ds = pydicom.dcmread(TRAIN_DATA_IMG + each_img)
    X2_train.append(ds.pixel_array)
X2_train = np.asarray(X2_train)
X2_train = X2_train.reshape(X2_train.shape[0],X2_train.shape[1],X2_train.shape[2],1)

val_imgs = tf.io.gfile.listdir(VAL_DATA_IMG)
for each_img in val_imgs:
    ds = pydicom.dcmread(VAL_DATA_IMG + each_img)
    X2_val.append(ds.pixel_array)
X2_val = np.asarray(X2_val)
X2_val = X2_val.reshape(X2_val.shape[0],X2_val.shape[1],X2_val.shape[2],1)

test_imgs = tf.io.gfile.listdir(TEST_DATA_IMG)
for each_img in test_imgs:
    ds = pydicom.dcmread(TEST_DATA_IMG + each_img)
    X2_test.append(ds.pixel_array)
X2_test = np.asarray(X2_test)
X2_test = X2_test.reshape(X2_test.shape[0],X2_test.shape[1],X2_test.shape[2],1)

def dkubeLoggerHook(epoch, logs):
    epoch = epoch + 1
    train_metrics = {
        'mode': "train",
        # 'accuracy': float(logs.get('accuracy', 0)),
        'loss': logs.get('loss', 0),
        'epoch': epoch
        # 'jobid': os.getenv('DKUBE_JOB_ID'),
        # 'username': os.getenv('DKUBE_USER_LOGIN_NAME')
    }
    eval_metrics = {
        'mode': "eval",
        # 'accuracy': float(logs.get('val_accuracy', 0)),
        'loss': logs.get('val_loss', 0),
        'epoch': epoch
        # 'jobid': os.getenv('DKUBE_JOB_ID'),
        # 'username': os.getenv('DKUBE_USER_LOGIN_NAME')
    }
    try:
        url = "http://dkube-ext.dkube:9401/export-training-info"
        requests.post(url, data=json.dumps({'data': [train_metrics]}))
        requests.post(url, data=json.dumps({'data': [eval_metrics]}))
    except Exception as exc:
        # print(exc)
        pass



def build_cnn_block(img_input_shape, penalty):
    cnn_input = k.layers.Input(shape=img_input_shape, name='img_input')
    cnn_block = k.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(penalty),
                activity_regularizer=regularizers.l1(penalty))(cnn_input)
    cnn_block = k.layers.MaxPooling2D(pool_size=(2, 2))(cnn_block)
    cnn_block = k.layers.Dropout(0.25)(cnn_block)
    cnn_block = k.layers.Flatten()(cnn_block)
    cnn_block = k.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(penalty),
                activity_regularizer=regularizers.l1(penalty))(cnn_block)
    return cnn_block, cnn_input

def build_dense_block(csv_input_shape, penalty):
    csv_input = k.layers.Input(shape=csv_input_shape, name='csv_input')
    dense_block = k.layers.Dense(32,activation='tanh',kernel_regularizer=regularizers.l2(penalty),
                activity_regularizer=regularizers.l1(penalty))(csv_input)
    dense_block = k.layers.Dropout(0.25)(dense_block)
    dense_block = k.layers.Flatten()(dense_block)
    dense_block = k.layers.Dense(32, activation='tanh', kernel_regularizer=regularizers.l2(penalty),
                activity_regularizer=regularizers.l1(penalty))(dense_block)
    return dense_block, csv_input

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", dest = 'epochs', type = int, required=True, help="no. of epochs")
    parser.add_argument("--learningrate", type = float, default=0.01, dest = 'lr', help="learning rate")
    parser.add_argument("--penalty", type = float, default=0.01, dest = 'penalty', help="regularizatio penalty range 0.001 to 0.01")
    parser.add_argument("--modeldir", default='model/', dest = 'modeldir', help="path to save model")
    args = parser.parse_args()

    lr = args.lr
    epochs = args.epochs
    penalty = args.penalty
    modeldir = args.modeldir
    img_input_shape = (512,512,1)
    csv_input_shape = (15,1)

    weights_file = "model/weights.h5"    # "/tmp/weights.h5"
    tf_summary_logs_path = "model/logs"
    export_path = 'model/infer_model/'
    model_checkpoint = ModelCheckpoint(weights_file, monitor="val_loss",
                                       save_best_only=True,
                                       save_weights_only=True, verbose=2)
    dkube_logger_callback = LambdaCallback(
        on_epoch_end=lambda epoch, logs: dkubeLoggerHook(epoch, logs))

    callbacks = [
        TensorBoard(log_dir=tf_summary_logs_path,
                    histogram_freq=0,
                    update_freq='epoch'),
        dkube_logger_callback,
        model_checkpoint
    ]

    cnn_block, cnn_input = build_cnn_block(img_input_shape, penalty)
    dense_block, csv_input = build_dense_block(csv_input_shape, penalty)
    merged = k.layers.Concatenate()([cnn_block,dense_block])
    merged = k.layers.Dense(16, activation='tanh')(merged)
    merged = k.layers.Dense(8, activation='tanh')(merged)
    merged = k.layers.Dense(1, activation='sigmoid')(merged)
    model = k.models.Model(inputs=[cnn_input, csv_input], outputs=[merged])

    ada_grad = k.optimizers.Adagrad(lr=lr, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=ada_grad, loss='mse')
    history = model.fit(x=[X2_train,X1_train], y= Y_train, validation_data =([X2_val,X1_val],Y_val), 
                        epochs=epochs, verbose=1, callbacks = callbacks)
    train_preds = model.predict([X2_train,X1_train])
    val_preds = model.predict([X2_val,X1_val])
    test_preds = model.predict([X2_test,X1_test])
    train_error = mean_absolute_error(Y_train, train_preds)
    val_error = mean_absolute_error(Y_val, val_preds)
    test_error = mean_absolute_error(Y_test, test_preds)
    print("Training error: ", train_error)
    print("Validation error: ", val_error)
    print("Test error: ", test_error)
    ########### Saving Model #################
    tf.keras.backend.set_learning_phase(0)  # Ignore dropout at inference
    with tf.keras.backend.get_session() as sess:
        tf.saved_model.simple_save(
            sess,
            export_path,
            inputs={t.name: t for t in model.inputs},
            outputs={'output': model.output})
    # if not os.path.exists(modeldir):
    #     os.makedirs(modeldir)
    # model.save(weights_file)
    # export_h5_to_pb(weights_file, export_path)
    print("Model saved")