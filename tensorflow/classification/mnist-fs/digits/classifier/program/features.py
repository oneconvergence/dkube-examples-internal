import numpy as np
import struct
import os
import pandas as pd

from dkube.sdk import *

inp_path = '/opt/dkube/input/'
out_path = '/opt/dkube/output/'
filename = 'featureset.parquet'

def read_idx(dataset = "training", path = "../data"):
    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)
    return img, lbl

if __name__ == "__main__":


    dkubeURL = 'https://172.16.146.128:32222'
    authToken = 'eyJhbGciOiJSUzI1NiIsImtpZCI6Ijc0YmNkZjBmZWJmNDRiOGRhZGQxZWIyOGM2MjhkYWYxIn0.eyJ1c2VybmFtZSI6Im9zbSIsInJvbGUiOiJkYXRhc2NpZW50aXN0LG1sZSxwZSxvcGVyYXRvciIsImV4cCI6NDg0NTY0NDMyMywiaWF0IjoxNjA1NjQ0MzIzLCJpc3MiOiJES3ViZSJ9.xMd-OpVe4-RdVClyvzX4bhibWNNB4nDI9WY4qbMGyYyJzrUMjb4xa-6hf0EbOXUh7ePYjKwFyuJJa-tXTj-L_pF_QlyycKF66WPIAl6-soMFIhmZXWlGCUqXHF9-pUkBqlpLLCwz07Z4nUzksMPkoIbHwWf_MUNll-lbaCNgd-z_X97ADu6lTBfhm3bV2eXnDBV5oBoxjF0Fg4fJxl_dNBzw926lswaWwndElr3VyDtpxV5hcsVKFFufunIn6r4leP6K_fYnT6E449IWOaAxYoxyTkmhQDkHBAJ3Odpd19wry1xgi9erRHB059oSr8wP3JqcPqP6hRTdpIEA9id03g'

    img, lbl = read_idx(path = inp_path)
    dataset = pd.DataFrame(data = img.reshape(60000, 784))/255
    dataset['label'] = lbl
    featureset = DkubeFeatureSet()
    featureset.update_features_path(path=out_path)
    featureset.write(dataset)

    api = DkubeApi(URL=dkubeURL, token=authToken)
    api.commit_features()


