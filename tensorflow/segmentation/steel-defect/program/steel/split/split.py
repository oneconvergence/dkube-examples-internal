import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import os, shutil
import numpy as np
import cv2
from random import shuffle


DATA_DIR = "/opt/dkube/input/"
OUT_DIR = "/opt/dkube/output/"

# DATA_DIR = "../../../output/"
# OUT_DIR = "../../../splits/"
TRAIN_DATA = OUT_DIR + 'train/'
TEST_DATA = OUT_DIR + 'test/'


df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
train_df, test_df = train_test_split(df, test_size=.10)

if not os.path.exists(TRAIN_DATA + 'images/'):
    os.makedirs(TRAIN_DATA + 'images/')

for i in range(len(train_df)):
    filename = train_df.iloc[i]['ImageId']
    shutil.copyfile(DATA_DIR + 'images/' + filename , TRAIN_DATA + 'images/' + filename) 
train_df.to_csv(TRAIN_DATA + 'train.csv', index=False, sep=',')
print("Train Data Created")

if not os.path.exists(TEST_DATA + 'images/'):
    os.makedirs(TEST_DATA + 'images/')

for i in range(len(test_df)):
    filename = test_df.iloc[i]['ImageId']
    shutil.copyfile(DATA_DIR + 'images/' + filename , TEST_DATA + 'images/' + filename) 
test_df.to_csv(TEST_DATA + 'test.csv', index=False, sep=',')
print("Test Data Created")