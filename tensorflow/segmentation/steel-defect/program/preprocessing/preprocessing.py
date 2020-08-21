import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
import shutil

def resize_image(image_dir, row, img_w, img_h):
    img_id = row['ImageId']
#     print(img_id)
    file_path =  os.path.join(image_dir, img_id)
#     print(file_path)
    image = cv2.imread(file_path, 0)
    image_resized = cv2.resize(image, (img_w, img_h))
    image_resized = np.array(image_resized, dtype=np.float64)
    # standardization of the image
    image_resized -= image_resized.mean()
    image_resized /= image_resized.std()
    X = np.expand_dims(image_resized, axis=2)
    return X

input_dir = '/opt/dkube/input/'
# out_dir = '../../../output/'
out_dir = '/opt/dkube/output/'
print("Loading raw data...")
data_dir = input_dir + 'severstal-steel-defect-detection/' 
image_dir = os.path.join(data_dir, 'train_images')

img_w = 800 # resized weidth
img_h = 256 # resized height
train_df = pd.read_csv(os.path.join(data_dir, 'train.csv')).fillna(-1)


out_img_dir = out_dir + 'images/'
if not os.path.exists(out_img_dir):
    os.makedirs(out_img_dir)
for i in range(len(train_df)):
    X = resize_image(image_dir, train_df.iloc[i], img_w, img_h)
    filename = train_df.iloc[i]['ImageId']
#     print(X.shape, filename)
    plt.imsave(out_img_dir + filename, X.reshape(img_h,img_w))
    if i%100==0:
        print('Finished processing ' + str(i) + ' images') 
shutil.copyfile(data_dir + 'train.csv', out_dir + 'train.csv') 
print("Data prprocessing finished")