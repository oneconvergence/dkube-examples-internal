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
import rpy2.robjects as robjects

DEFAULT_MODEL_NAME = "model"

parser = argparse.ArgumentParser(parents=[kfserving.kfserver.parser])
parser.add_argument('--model_name', default=DEFAULT_MODEL_NAME,
                    help='The name that the model is served under.')
parser.add_argument('--predictor_host', help='The URL for the model predict function', required=True)

args, _ = parser.parse_known_args()

rstring = '''
library(png)
library(base64enc)
library(readr)
function(base64string){
    outconn <- file("inp.png","wb")
    base64decode(what=base64string, output=outconn)
    close(outconn)
    x <- readPNG("inp.png")
    y <- array(x, c(1,784))
    y
}
'''

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
        ###### Loading R transformer code to convert base64 string to array ########
        # with open('transformer.R','r') as f:
        #     rstring = f.read()
        rfunc=robjects.r(rstring)
        x = rfunc(data)
        x = np.asarray(x)
        payload = {"inputs": x.tolist(), 'token':inputs['token']}
        return payload

    def postprocess(self, inputs: List) -> List:
        return inputs

if __name__ == "__main__":
    transformer = ImageTransformer(args.model_name, predictor_host=args.predictor_host)
    kfserver = kfserving.KFServer()
    kfserver.start(models=[transformer])