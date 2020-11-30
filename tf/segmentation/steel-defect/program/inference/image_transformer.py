import json
import numpy as np
import requests
import cv2
from PIL import Image
import kfserving
from typing import List, Dict
import logging
import io
import base64
import sys,json
import os

img_w = 800 # resized weidth
img_h = 256 # resized height
filename = 'temp.jpg'
img = ""

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
        # inputs is a json file, inside that data, using the data value form a image
        # write into jpeg file
        del inputs['instances']
        logging.info("prep =======> %s",str(type(inputs)))
        try:
            json_data = inputs
        except ValueError:
            return json.dumps({ "error": "Recieved invalid json" })
        data = json_data["signatures"]["inputs"][0][0]["data"]
        b64_filewriter(filename, data)
        image = cv2.imread(filename, 0)
        img = image
        x = image
        x = cv2.resize(x, (img_w, img_h))
        x = np.array(x, dtype=np.float64)
        x -= x.mean()
        x /= x.std()
        x = x.reshape(1,img_h,img_w,1)
        payload = {"inputs": {'input': x.tolist()}, 'token':inputs['token']}
        return payload

    def postprocess(self, predictions: List) -> List:
        logging.info("prep =======> %s",str(type(predictions)))
        image = cv2.imread(filename, 0)
        inp_img = cv2.resize(image, (img_w, img_h))
        count = 0
        class_viz_count = [0,0,0,0]
        class_iou_score = [0, 0, 0, 0]
        class_mask_sum = [0, 0, 0, 0]
        class_pred_sum = [0, 0, 0, 0]
        pred = np.asarray(predictions['outputs'])
        t_img = cv2.resize(inp_img, (img_w, img_h))
        p_mask = 0
        p_class = 0
        for idx, val in enumerate(pred):
            if pred[idx].sum() > 0: 
                preds_temp = [pred[idx][...,i] for i in range(0,4)]
                preds_temp = [p > .5 for p in preds_temp]
                for i, pred in enumerate(preds_temp):
                    image_class = i + 1
                    class_pred_sum[i] += pred.sum()
                    if pred.sum() > 0 and class_viz_count[i] < 5:
                        class_viz_count[i] += 1
                        p_mask = pred
                        p_class = image_class
                        logging.info("prep =======> got the class")
        im1 = Image.fromarray(t_img, mode = 'L')
        im2 = Image.fromarray(np.uint8((p_mask)*255), mode = 'L')
        alphaBlended1 = Image.blend(im1, im2, alpha=0.6)
        output = np.asarray(alphaBlended1)
        cv2.imwrite('out.png', output)
        with open('out.png', 'rb') as open_file:
            byte_content = open_file.read()
        base64_bytes = base64.b64encode(byte_content)
        base64_string = base64_bytes.decode('utf-8')
        logging.info("prep =======> %s",str(output.shape))
        return {'image':base64_string, 'p_class': p_class}
