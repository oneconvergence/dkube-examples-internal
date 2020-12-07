import numpy as np
import struct
import os
import pandas as pd
import sys
import argparse
import yaml

sys.path.insert(0, os.path.abspath("/usr/local/lib/python3.6/dist-packages"))
from dkube.sdk import *

inp_path = '/opt/dkube/input/'
out_path = '/opt/dkube/output/'
filename = 'featureset.parquet'

def read_idx(dataset = "training", path = "../data"):
    # Fucntion to convert ubyte files to numpy arrays
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
    parser.add_argument("--url", dest = 'url', default=None, type = str, help="setup URL")
    global FLAGS
    FLAGS, unparsed = parser.parse_known_args()
    dkubeURL = FLAGS.url
    authToken = os.getenv('DKUBE_USER_ACCESS_TOKEN') # Dkube access token

    # Converting data from ubyte to numpy arrays
    img, lbl = read_idx(path = inp_path)
    dataset  = pd.DataFrame(data = img.reshape(img.shape[0], 784))/255
    dataset['label'] = lbl
    # Featureset API call
    featureset = DkubeFeatureSet()	
    #updating featureset path for output
    featureset.update_features_path(path=out_path)	
    # writing featureset, input:dataframe
    featureset.write(dataset)
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
    # DkubeApi calling
    api = DkubeApi(URL=dkubeURL, token=authToken)
    # Writing featureset metadata to yaml file
    featureset_metadata = yaml.dump(featureset_metadata, default_flow_style=False)
    with open("fspec.yaml", 'w') as f:
         f.write(featureset_metadata)
    # Uploading featureset metadata
    resp = api.upload_featurespec(featureset = 'mnist-fs',filepath = "fspec.yaml")
    print("featurespec upload response:", resp)
    # Commiting featureset before exiting the program.
    resp = api.commit_features()
    print("featureset commit response:", resp)
