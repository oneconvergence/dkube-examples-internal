import json
import numpy as np
import requests
import cv2
from PIL import Image
from typing import List, Dict
import logging
import io
import base64

img_w = 800 # resized weidth
img_h = 256 # resized height
filename = 'temp.png'

def b64_filewriter(filename, content):
    string = content.encode('utf8')
    b64_decode = base64.decodebytes(string)
    fp = open(filename, "wb")
    fp.write(b64_decode)
    fp.close()

def preprocess(inputs: Dict) -> Dict:
    del inputs['instances']
    logging.info("prep =======> %s",str(inputs['token']))
    try:
        json_data = inputs
    except ValueError:
        return json.dumps({ "error": "Recieved invalid json" })
    data = json_data["signatures"]["inputs"][0][0]["data"]
    b64_filewriter(filename, data)
    image = cv2.imread(filename, 0)
    x = image
    x = cv2.resize(x, (img_w, img_h))
    x = np.array(x, dtype=np.float64)
    x -= x.mean()
    x /= x.std()
    x = x.reshape(1,img_h,img_w,1)
    payload = {"inputs": {'input': x.tolist()}}
    return payload

def postprocess(predictions: List) -> List:
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
    im1 = Image.fromarray(t_img, mode = 'L')
    im2 = Image.fromarray(np.uint8((p_mask)*255), mode = 'L')
    alphaBlended1 = Image.blend(im1, im2, alpha=0.6)
    output = np.asarray(alphaBlended1)
    return output, p_class

with open('image_string.txt', 'r') as f:
    data = f.read()
# filename = 'temp.png'
# b64_filewriter(filename, data)  
# filename = '005d86c25.jpg'
# data = json_data["signatures"]["inputs"][0][0]["data"].encode()

inputs = {
            'signatures':{
                'inputs':[[{'data':data}]]
            },
            'instances': [],
            'token': 'Dumy token'
}
# img = cv2.imread(filename, 0)
payload = preprocess(inputs)
r = requests.post('http://35.184.20.234:9005/v1/models/steel:predict', json=payload)
preds = json.loads(r.content.decode('utf-8'))
out_img , out_class = postprocess(preds)
cv2.imwrite('out.png',out_img)