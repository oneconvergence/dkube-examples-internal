import kfserving
from typing import List, Dict
from PIL import Image
import logging
import io
import numpy as np
import base64
import argparse

import sys,json
import requests
import os
import logging
import cv2

DEFAULT_MODEL_NAME = "model"

parser = argparse.ArgumentParser(parents=[kfserving.kfserver.parser])
parser.add_argument('--model_name', default=DEFAULT_MODEL_NAME,
                    help='The name that the model is served under.')
parser.add_argument('--predictor_host', help='The URL for the model predict function', required=True)

args, _ = parser.parse_known_args()

filename = 'temp.jpg'
img_w = 298
img_h = 298


def b64_filewriter(filename, content):
    string = content.encode('utf8')
    b64_decode = base64.decodebytes(string)
    fp = open(filename, "wb")
    fp.write(b64_decode)
    fp.close()


class ImageTransformer(kfserving.KFModel):
    def __init__(self, name: str, predictor_host: str):
        super().__init__(name)
        self.predictor_host = predictor_host

    def preprocess(self, inputs: Dict) -> Dict:
        logging.info("preprocess1")
        del inputs['instances']
        try:
            json_data = inputs
        except ValueError:
            return json.dumps({ "error": "Recieved invalid json" })
        data = json_data["signatures"]["inputs"][0][0]["data"]
        b64_filewriter(filename, data)
        image = cv2.imread(filename)
        img = image
        x = image
        x = cv2.resize(x, (img_w, img_h))
        x = np.array(x, dtype=np.float64)
        x = x.reshape(1,img_h,img_w,3)
        payload = {"inputs": {'input': x.tolist()}, 'token':inputs['token']}
        return payload

    def postprocess(self, inputs: List) -> List:
        pred = inputs["outputs"][0][0]
        result = ''
        if pred < 0.5:
            result = "Cat"
        else:
            result = "Dog"
        return result

if __name__ == "__main__":
    transformer = ImageTransformer(args.model_name, predictor_host=args.predictor_host)
    kfserver = kfserving.KFServer()
    kfserver.start(models=[transformer])

