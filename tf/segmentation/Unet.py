import tensorflow.keras
import tensorflow.keras.backend as K
import tensorflow.keras.layers as klayers
from tensorflow.keras.models import Model

import pandas as pd
import cv2, os
import numpy as np
import tensorflow as tf
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
        
class Unet:
    def __init__(self, input_shape = (256,800,1), output_units = 1):
        
        self.input_shape = input_shape
        self.output_units = output_units
        
    def _swish(self,x):
        return K.sigmoid(x)*x
    
    def _SEBlock(self, se_ratio=16, activation = "relu"):
        
        def f(input_x):

            input_channels = 1 # input_x._keras_shape[-1]
            reduced_channels = max(input_channels // se_ratio, 8)
            
            x = klayers.GlobalAveragePooling2D()(input_x)
            x = klayers.Dense(units = reduced_channels, kernel_initializer = "he_normal")(x)
            x = klayers.Activation(activation)(x)
            x = klayers.Dense(units = input_channels, activation = 'sigmoid', kernel_initializer = "he_normal")(x)
            
            return klayers.multiply([input_x, x])
        
        return f    
    
    def _cn_bn_act(self, filters = 64, kernel_size = (3,3), bn_flag = False, activation = "relu"):
        
        def f(input_x):
            
            x = input_x
            x = klayers.Conv2D(filters = filters, kernel_size = kernel_size, strides = (1,1), padding = "same", kernel_initializer = "he_normal")(x)
            x = klayers.BatchNormalization()(x) if bn_flag == True else x
            x = klayers.Activation(activation)(x)
            
            return x
        
        return f
    
    def _UpSamplingBlock(self, filters = 64, kernel_size = (3,3), upsize = (2,2), bn_flag = True, up_flag = False, se_flag = True):
        
        def f(up_c, con_c):
            
            if up_flag:
                x = klayers.UpSampling2D(size = upsize, interpolation = 'bilinear')(up_c)
            else:
                x = klayers.Conv2DTranspose(filters = filters, kernel_size = (2,2), strides = upsize, padding = "same", kernel_initializer = "he_normal")(up_c)
            
            x = klayers.concatenate([x,con_c])
            x = self._cn_bn_act(filters = filters, kernel_size = kernel_size, bn_flag = bn_flag, activation = self._swish)(x)
            x = self._SEBlock(activation = self._swish)(x) if se_flag == True else x
            x = self._cn_bn_act(filters = filters, kernel_size = kernel_size, bn_flag = bn_flag, activation = self._swish)(x)
            
            return x
        return f
    
    def _DownSamplingBlock(self, filters = 64, kernel_size = (3,3), downsize = (2,2), bn_flag = True, is_bottom = False, se_flag = True):
        
        def f(input_x):
            
            x = self._cn_bn_act(filters = filters, kernel_size = kernel_size, bn_flag = bn_flag, activation = self._swish)(input_x)
            x = self._SEBlock(activation = self._swish)(x) if se_flag == True else x
            c = self._cn_bn_act(filters = filters, kernel_size = kernel_size, bn_flag = bn_flag, activation = self._swish)(x)
            return c if (is_bottom == True) else (c,klayers.MaxPooling2D(pool_size = downsize)(c))
            
        return f
    
    def build_unet(self):     
        input_x = klayers.Input(shape = self.input_shape)
        #encoder region
        c1,p1 = self._DownSamplingBlock(filters = 16)(input_x)
        c2,p2 = self._DownSamplingBlock(filters = 32)(p1)
        c3,p3 = self._DownSamplingBlock(filters = 32)(p2)
        c4,p4 = self._DownSamplingBlock(filters = 64)(p3)
        c5,p5 = self._DownSamplingBlock(filters = 128)(p4)
        
        c6 = self._DownSamplingBlock(filters = 256, is_bottom = True)(p5)
        
        #decoder region
        u7 = self._UpSamplingBlock(filters = 128)(c6,c5)
        u8 = self._UpSamplingBlock(filters = 64)(u7,c4)
        u9 = self._UpSamplingBlock(filters = 32)(u8,c3)
        u10 = self._UpSamplingBlock(filters = 32)(u9,c2)
        u11 = self._UpSamplingBlock(filters = 16)(u10,c1)
        output_x = klayers.Conv2D(filters = self.output_units, kernel_size = (1,1), padding = "same", activation = "sigmoid", kernel_initializer = "he_normal")(u11)
        model = Model(inputs = [input_x], outputs = [output_x])
        return model
    
def Tversky_Loss(y_true, y_pred, smooth = 1, alpha = 0.3, beta = 0.7, flatten = False):
    
    if flatten:
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
    
    TP = K.sum(y_true * y_pred)
    FP = K.sum((1-y_true) * y_pred)
    FN = K.sum(y_true * (1-y_pred))
    
    tversky_coef = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    
    return 1 - tversky_coef

def Focal_Loss(y_true, y_pred, alpha = 0.8, gamma = 2.0, flatten = False):
    
    if flatten:
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)    
    
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce_exp = K.exp(-bce)
    
    loss = K.mean(alpha * K.pow((1-bce_exp), gamma) * bce)
    return loss

def weighted_bce(weight = 0.6):
    
    def convert_2_logits(y_pred):
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
        return tf.log(y_pred / (1-y_pred))
    
    def weighted_binary_crossentropy(y_true, y_pred):
        y_pred = convert_2_logits(y_pred)
        loss = tf.nn.weighted_cross_entropy_with_logits(logits = y_pred, targets = y_true, pos_weight = weight)
        return loss
    
    return weighted_binary_crossentropy

def Combo_Loss(y_true, y_pred, a = 0.4, b = 0.2, c= 0.4):
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    return a*weighted_bce()(y_true, y_pred) + b*Focal_Loss(y_true_f, y_pred_f) + c*Tversky_Loss(y_true_f, y_pred_f)

def Dice_coef(y_true, y_pred, smooth = 1):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
 
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def Dice_loss(y_true, y_pred):   
    return  1.0 - Dice_coef(y_true, y_pred)
    
def get_batch_data(batch_df, inp_dir, img_h, img_w):
    b_size = len(batch_df)
    X = np.empty((b_size, img_h, img_w, 1))
    Y = np.empty((b_size, img_h, img_w, 1))
    for i in range(b_size):
        filename = batch_df.iloc[i]['ImageId']
        x = cv2.imread(inp_dir + 'images/' + filename, 0)
        x = np.array(x, dtype=np.float64)
        x -= x.mean()
        x /= x.std()
        y = cv2.imread(inp_dir + 'masks/' + filename, 0)
        y = np.array(y, dtype=np.float64)
        y -= y.mean()
        y /= y.std()
        X[i,] = x.reshape(img_h, img_w, 1)
        Y[i,] = y.reshape(img_h, img_w, 1)
    return X, Y

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", dest = 'epochs', type = int, default = 1, help="no. of epochs")
args = parser.parse_args()

if tf.test.is_gpu_available():
    print("****************Training on GPU****************")
else:
    print("****************Training on CPU****************")

# DATA_PATH = '/opt/dkube/input/'
# OUT_DIR = '/opt/dkube/output/'

DATA_PATH = '../../../splits/train/'
OUT_DIR = '../../../'
MODEL_DIR = OUT_DIR + 'model/'
LOG_DIR = MODEL_DIR + 'logs/'
METRIC_PATH = OUT_DIR + 'metrics/'
INF_EXPORT_PATH = MODEL_DIR + 'inf_model/'

batch_size = 8
epochs = args.epochs

print("Creating model")
unet_builder = Unet(input_shape = (256,800,1))
model = unet_builder.build_unet()
optimizer = tf.keras.optimizers.Adam(lr = 0.001, decay = 1e-6)
model.compile(loss = Combo_Loss, optimizer = optimizer, metrics = [Dice_coef])

print("Loading data")
train_df = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
no_of_samples = len(train_df)
no_of_pass = int(no_of_samples/batch_size)

callback = TensorBoard(LOG_DIR)
callback.set_model(model)
train_names = ['train_loss', 'train_tversky']
print("Starting training, no of batches per epoch are ", no_of_pass)
for each_epoch in range(epochs):
    idx = 0
    pass_count = 1
    train_metrics = []
    val_metrics = []
    for each_pass in range(1, no_of_pass+1):
        x, y = get_batch_data(train_df[idx:each_pass*batch_size], DATA_PATH, 256, 800)
        logs = model.train_on_batch(x=x, y= y)
        write_log(callback, train_names, logs, each_epoch)
        
        idx = each_pass*batch_size
        train_metrics.append(logs)
        print(logs)
        print('Epoch = ', each_epoch+1, ', step = ', each_pass, ', loss = ',logs[0], 'Dice_coef = ', logs[1])
    train_metrics = np.asarray(train_metrics)
    train_metrics = np.average(train_metrics, axis=0)

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