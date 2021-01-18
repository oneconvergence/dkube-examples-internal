import numpy as np
import struct
import os
import pandas as pd

import argparse
import yaml

from dkube.sdk import *

inp_path = "/opt/dkube/input/"
out_path = "/opt/dkube/output/"

# Read dataset
def read_idx(dataset="training", path="../data"):
    # Fucntion to convert ubyte files to numpy arrays
    if dataset == "training":
        fname_img = os.path.join(path, "train-images-idx3-ubyte")
        fname_lbl = os.path.join(path, "train-labels-idx1-ubyte")
    elif dataset == "testing":
        fname_img = os.path.join(path, "t10k-images-idx3-ubyte")
        fname_lbl = os.path.join(path, "t10k-labels-idx1-ubyte")

    # Load everything in some numpy arrays
    with open(fname_lbl, "rb") as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, "rb") as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)
    return img, lbl


if __name__ == "__main__":

    ########--- Parse for parameters ---########

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", dest="url", default=None, type=str, help="setup URL")
    parser.add_argument("--fs", dest="fs", required=True, type=str, help="featureset")

    global FLAGS
    FLAGS, unparsed = parser.parse_known_args()

    ########--- Get DKube client handle ---########

    # DKube API EndPoint
    dkubeURL = FLAGS.url
    # Dkube access token
    authToken = os.getenv("DKUBE_USER_ACCESS_TOKEN")
    # client handle
    api = DkubeApi(URL=dkubeURL, token=authToken)

    ########--- Load mnist data  ---########

    # Convert data from ubyte to numpy arrays
    img, lbl = read_idx(path=inp_path)
    df = pd.DataFrame(data=img.reshape(img.shape[0], 784)) / 255
    df["label"] = lbl

    ########--- Upload Featureset metadata ---########

    # Featureset to use
    fs = FLAGS.fs

    # Prepare featurespec - Name, Description, Schema for each feature
    keys = df.keys()
    schema = df.dtypes.to_list()
    featureset_metadata = []
    for i in range(len(keys)):
        metadata = {}
        metadata["name"] = str(keys[i])
        metadata["description"] = None
        metadata["schema"] = str(schema[i])
        featureset_metadata.append(metadata)

    # Commit featureset
    resp = api.commit_featureset(name=fs, df=df, metadata=featureset_metadata)
    print("featureset commit response:", resp)
