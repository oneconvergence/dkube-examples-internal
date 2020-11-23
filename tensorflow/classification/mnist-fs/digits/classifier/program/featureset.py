import numpy as np
import struct
import os
import pandas as pd
from dkube.sdk import *
import argparse
import json

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

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", dest = 'url', type = str, help="setup URL", required=True)
    global FLAGS
    FLAGS, unparsed = parser.parse_known_args()
    dkubeURL = FLAGS.url
    authToken = os.getenv('DKUBE_USER_ACCESS_TOKEN')

    img, lbl = read_idx(path = inp_path)
    dataset  = pd.DataFrame(data = img.reshape(img.shape[0], 784))/255
    dataset['label'] = lbl
    ####### Featureset metadata #########
    keys   = dataset.keys()
    schema = dataset.dtypes.to_list()
    featureset_metadata = []
    for i in range(len(keys)):
        metadata = {}
        metadata["name"] = str(keys[i])
        metadata["description"] = None
        metadata["schema"] = str(schema[i])
        featureset_metadata.append(metadata)
    
#     with open('data.json', 'w') as f:
#         json.dump(json.dumps(featureset_metadata), f)
    featureset = DkubeFeatureSet()
    featureset.update_features_path(path=out_path)
    featureset.write(dataset)
    
    api = DkubeApi(URL=dkubeURL, token=authToken)
#     api.upload_featurespec(featureset = 'mnist-fs',filepath = "./data.json")
    api.commit_features()
