import numpy as np
import struct
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

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

img, lbl = read_idx(path = inp_path)
dataset = pd.DataFrame(data = img.reshape(img.shape[0], 784))/255
dataset['label'] = lbl
table = pa.Table.from_pandas(dataset)
pq.write_table(table, os.path.join(out_path, filename))