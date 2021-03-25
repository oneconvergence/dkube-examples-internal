import tensorflow as tf
from tensorflow import reduce_sum
from tensorflow.keras.backend import pow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D, Concatenate, Add, Flatten
from tensorflow.keras.losses import binary_crossentropy
from sklearn.model_selection import train_test_split
import pandas as pd
import cv2, os
import numpy as np

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model.signature_def_utils import predict_signature_def
from tensorflow.python.saved_model import tag_constants

import pickle, json
import argparse

def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

def rle_to_mask(rle_string,height,width):
    rows, cols = height, width
    if rle_string == -1:
        return np.zeros((height, width))
    else:
        rleNumbers = [int(numstring) for numstring in rle_string.split(' ')]
        rlePairs = np.array(rleNumbers).reshape(-1,2)
        img = np.zeros(rows*cols,dtype=np.uint8)
        for index,length in rlePairs:
            index -= 1
            img[index:index+length] = 255
        img = img.reshape(cols,rows)
        img = img.T
        return img
    
def bn_act(x, act=True):
    'batch normalization layer with an optinal activation layer'
    x = tf.keras.layers.BatchNormalization()(x)
    if act == True:
        x = tf.keras.layers.Activation('relu')(x)
    return x

def conv_block(x, filters, kernel_size=3, padding='same', strides=1):
    'convolutional layer which always uses the batch normalization layer'
    conv = bn_act(x)
    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
    return conv

def stem(x, filters, kernel_size=3, padding='same', strides=1):
    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    conv = conv_block(conv, filters, kernel_size, padding, strides)
    shortcut = Conv2D(filters, kernel_size=1, padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    output = Add()([conv, shortcut])
    return output

def residual_block(x, filters, kernel_size=3, padding='same', strides=1):
    res = conv_block(x, filters, k_size, padding, strides)
    res = conv_block(res, filters, k_size, padding, 1)
    shortcut = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    output = Add()([shortcut, res])
    return output

def upsample_concat_block(x, xskip):
    u = UpSampling2D((2,2))(x)
    c = Concatenate()([u, xskip])
    return c

def ResUNet(img_h, img_w):
    f = [16, 32, 64, 128, 256]
    inputs = Input((img_h, img_w, 1))
    
    ## Encoder
    e0 = inputs
    e1 = stem(e0, f[0])
    e2 = residual_block(e1, f[1], strides=2)
    e3 = residual_block(e2, f[2], strides=2)
    e4 = residual_block(e3, f[3], strides=2)
    e5 = residual_block(e4, f[4], strides=2)
    
    ## Bridge
    b0 = conv_block(e5, f[4], strides=1)
    b1 = conv_block(b0, f[4], strides=1)
    
    ## Decoder
    u1 = upsample_concat_block(b1, e4)
    d1 = residual_block(u1, f[4])
    
    u2 = upsample_concat_block(d1, e3)
    d2 = residual_block(u2, f[3])
    
    u3 = upsample_concat_block(d2, e2)
    d3 = residual_block(u3, f[2])
    
    u4 = upsample_concat_block(d3, e1)
    d4 = residual_block(u4, f[1])
    
    outputs = tf.keras.layers.Conv2D(4, (1, 1), padding="same", activation="sigmoid")(d4)
    model = tf.keras.models.Model(inputs, outputs)
    return model

def dsc(y_true, y_pred):
    smooth = 1.
    y_true_f = Flatten()(y_true)
    y_pred_f = Flatten()(y_pred)
    intersection = reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (reduce_sum(y_true_f) + reduce_sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dsc(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

def tversky(y_true, y_pred, smooth=1e-6):
    y_true_pos = tf.keras.layers.Flatten()(y_true)
    y_pred_pos = tf.keras.layers.Flatten()(y_pred)
    true_pos = tf.reduce_sum(y_true_pos * y_pred_pos)
    false_neg = tf.reduce_sum(y_true_pos * (1-y_pred_pos))
    false_pos = tf.reduce_sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

def focal_tversky_loss(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return tf.keras.backend.pow((1-pt_1), gamma)

def get_batch_data(batch_df, inp_dir, img_h, img_w):
    b_size = len(batch_df)
    X = np.empty((b_size, img_h, img_w, 1))
    Y = np.empty((b_size, img_h, img_w, 4))
    for i in range(b_size):
        filename = batch_df.iloc[i]['ImageId']
        x = cv2.imread(inp_dir + 'images/' + filename, 0)
        x = np.array(x, dtype=np.float64)
        x -= x.mean()
        x /= x.std()
        
        mask = np.empty((img_h, img_w, 4))
        rle = batch_df.iloc[i]['EncodedPixels']
        for idm, image_class in enumerate([1,2,3,4]):
            if batch_df.iloc[i]['ClassId'] == image_class:
                class_mask = rle_to_mask(rle, width=img_w, height=img_h)
            else:
                class_mask = np.zeros((img_w, img_h))

            class_mask_resized = cv2.resize(class_mask, (img_w,img_h))
            mask[...,idm] = class_mask_resized
        y = mask
        y = (y > 0).astype(np.float32)
        
        
        X[i,] = x.reshape(img_h, img_w, 1)
        Y[i,] = y
    return X, Y

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", dest = 'epochs', type = int, default = 1, help="no. of epochs")
args = parser.parse_args()

DATA_PATH = '/opt/dkube/input/'
OUT_DIR = '/opt/dkube/output/'

# DATA_PATH = '../../../splits/train/'
# OUT_DIR = '../../../'
MODEL_DIR = OUT_DIR + 'model/'
LOG_DIR = MODEL_DIR + 'logs/'
METRIC_PATH = OUT_DIR + 'metrics/'
INF_EXPORT_PATH = MODEL_DIR + 'inf_model/'

img_w = 800 # resized weidth
img_h = 256 # resized height
batch_size = 12
epochs = args.epochs
k_size = 3 # kernel size 3x3

if tf.test.is_gpu_available():
    print("****************Training on GPU****************")
else:
    print("****************Training on CPU****************")

print("Creating model")
model = ResUNet(img_h=img_h, img_w=img_w)
adam = tf.keras.optimizers.Adam(lr = 0.01, epsilon = 0.1)

model.compile(optimizer=adam, loss=focal_tversky_loss, metrics=[tversky])

print("Loading data")
train_df = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
no_of_samples = len(train_df)
no_of_pass = int(no_of_samples/batch_size)

callback = TensorBoard(LOG_DIR)
callback.set_model(model)
train_names = ['train_loss', 'train_tversky']
print("Starting training, no of batches per epoch are ", no_of_pass)
for each_epoch in range(epochs):
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    idx = 0
    pass_count = 1
    train_metrics = []
    val_metrics = []
    for each_pass in range(1, no_of_pass+1):
        x, y = get_batch_data(train_df[idx:each_pass*batch_size], DATA_PATH, img_h, img_w)
        logs = model.train_on_batch(x=x, y= y)
        write_log(callback, train_names, logs, each_epoch)
        idx = each_pass*batch_size
        train_metrics.append(logs)
    train_metrics = np.asarray(train_metrics)
    train_metrics = np.average(train_metrics, axis=0)
    print('Epoch = ', each_epoch+1, ', loss = ',train_metrics[0], 'tversky_dist = ', train_metrics[1])

############### Writing Metrics ##########################
metrics = []
metric_names = ['train_loss', 'train_tversky']
if not tf.io.gfile.exists(METRIC_PATH):
    tf.io.gfile.makedirs(METRIC_PATH)
for i in range(2):
    temp = {}
    temp['class'] = 'scalar'
    temp['name'] = metric_names[i]
    temp['value'] = str(train_metrics[i])
    metrics.append(temp)
metrics = {'metrics':metrics}
with open(METRIC_PATH + 'metrics.json', 'w') as outfile:
    json.dump(metrics, outfile, indent=4)
############### Saving Model ###############################
version = 0
if not tf.io.gfile.exists(INF_EXPORT_PATH):
    tf.io.gfile.makedirs(INF_EXPORT_PATH)
saved_models = tf.io.gfile.listdir(INF_EXPORT_PATH)
saved_models = [int(mdir) for mdir in saved_models if '.' not in mdir]
if len(saved_models) < 1:
    version = 1
else:
    version = max(saved_models) + 1
model.save(MODEL_DIR + 'weights.h5')
tf.keras.backend.set_learning_phase(0)  # Ignore dropout at inference
with tf.keras.backend.get_session() as sess:
    tf.saved_model.simple_save(
        sess,
        INF_EXPORT_PATH + str(version),
        inputs={'input': model.input},
        outputs={'output': model.output})
print("Model saved, version = ", version)