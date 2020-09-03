from typing import List, Dict
import json
import logging
import numpy as np
import requests
import cv2
import pandas as pd
import base64


def preprocess(inputs: Dict) -> Dict:
    logging.info("inputs %s", str(inputs))
    del inputs['instances']
    try:
        json_data = inputs
    except ValueError:
        return json.dumps({ "error": "Recieved invalid json" })
    data = json_data["signatures"]["inputs"][0][0]["data"].encode()
    with open("image.png", "wb") as fh:
        fh.write(base64.decodebytes(data))
    img = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28,28))
    img = img.reshape(1,1,img.shape[0],img.shape[1])
    img = img.astype('float32')
    payload = {"instances": img.tolist(), "token":inputs["token"]}
    logging.info("token =======> %s",str(inputs["token"]))
    return payload

def postprocess(inputs: List) -> List:
    return inputs
