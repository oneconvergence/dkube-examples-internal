import argparse
import json

import numpy as np
import requests
import cv2
import pandas as pd

DATA_DIR = 'data/'

test_df = pd.read_csv(DATA_DIR + "cli_inp.csv")
csv = test_df.drop(['days_to_death','bcr_patient_barcode'], axis = 1)
csv = np.asarray(csv)
csv = csv.reshape(csv.shape[0],csv.shape[1],1)

img = cv2.imread(DATA_DIR + 'img_inp.png', cv2.IMREAD_GRAYSCALE)
img = img.reshape(1,img.shape[0],img.shape[1],1)


payload = {
    "inputs": {'csv_input:0': csv.tolist(),'img_input:0': img.tolist()}
}

URL = 'http://35.223.183.1:31380/v1/models/mnist-s3'

r = requests.post(URL + ':predict', json=payload, headers={"host":"mnist-s3.oc.dkube.ai"})

pred = json.loads(r.content.decode('utf-8'))

print('Days to death:',int(pred['outputs'][0][0]*1000))