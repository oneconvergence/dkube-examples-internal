import json
import numpy as np
import requests
import kfserving
from typing import List, Dict
import logging
import io
import base64
import sys,json
import os
import pandas as pd

filename = 'temp.csv'
img = ""

def preprocess(inputs: Dict) -> Dict:
    # inputs is a json file, inside that data, using the data value form a image
    # write into jpeg file
    del inputs['instances']
    logging.info("prep =======> %s",str(type(inputs)))
    try:
        json_data = inputs
    except ValueError:
        return json.dumps({ "error": "Recieved invalid json" })
    data = json_data["signatures"]["inputs"][0][0]["data"]
    with open(filename,'w') as f:
        f.write(data)
    data = pd.read_csv(filename)
    dates = data['Date']
    dates = [date.split('-')[0] for date in dates]
    l = len(dates)
    dates = np.asarray(dates).reshape(l,1)
    payload = {"instances": dates.tolist() , "token":inputs["token"]}
    logging.info("token =======> %s",str(inputs["token"]))
    return payload
    
def postprocess(predictions: List) -> List:
    logging.info("prep =======> %s",str(type(predictions)))
    preds = predictions["predictions"]
    l = len(preds)
    if l == 1:
        st = 'Stock value is, '
    else:
        st = 'Stock values are, '
    for i in range(l):
        st += str(round(preds[i],3))
        if i != l-1:
            st += ',  '
    st += '.'
    return {"result": st}
