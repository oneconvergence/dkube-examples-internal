import sys,json
import requests
import base64
import os
import logging
import torch
import torchvision.transforms as transforms
from PIL import Image
from collections import OrderedDict


def convert(input_file):
    try:
        f = open(input_file, "rb").read()
        data = base64.encodestring(f)
        data = data.decode('utf-8')
    except Exception as err:
        msg = "Failed to convert input image. " + str(err)
        logging.error(msg)
        return "", msg
    return data, ""

def cleanup(input_file, convertor_file=''):
    if input_file != None and os.path.exists(input_file):
        os.remove(input_file)
    if convertor_file != None and os.path.exists(base_dir + convertor_file + ".py"):
        os.remove(base_dir + convertor_file + ".py")


def b64_filewriter(filename, content):
    string = content.encode('utf8')
    b64_decode = base64.decodebytes(string)
    fp = open(filename, "wb")
    fp.write(b64_decode)
    fp.close()

def preprocess(json_data):
        model_inputs  = model_method = script = None
        input_file = ""
        input_type_mapping = {}
        in_signature = json_data["signatures"]["name"]
        in_data = json_data["signatures"]["inputs"]
        rqst_list = []
        batch_list = {}
        logging.info("preprpcess2")
        for batch in in_data:
            for element in batch:
                batch_list.clear()
                key = "inputs"
                data = element["data"]
                if data:
                    input_file = "input_" +key
                    b64_filewriter(input_file, data)
                process = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
                print("input_file: %s" % input_file)
                input_image = Image.open(input_file)
                #input_tensor = process(input_image)
                input_tensor = transforms.ToTensor()(input_image)
                input_batch = input_tensor.unsqueeze(0)
        logging.info("preprpcess3")
        res = {"signature_name":in_signature,"instances": input_batch[0:1].tolist()}
        return res


def postprocess(output):
   import json
   with open('/workspace/imagenet_class_index.json', 'r') as fp:
      class_idx = json.load(fp)
      labels = [class_idx[str(k)][1] for k in range(len(class_idx))]

   # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
   _, indices = torch.sort(output, descending=True)
   percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100
   #print(labels[index[0]], percentage[index[0]].item())
   res1 = OrderedDict()
   inds = indices[0][:10]
   for idx in inds:
      #print(labels[idx], percentage[idx].item())
      res1[labels[idx]] = percentage[idx].item()
   keys = list(res1.keys())
   values = list(res1.values())
   #keys = ['Labrador_retriever', 'golden_retriever', 'English_foxhound', 'kuvasz', 'curly-coated_retriever', 'redbone', 'clumber', 'bloodhound', 'Sussex_spaniel', 'otterhound']
   #values = [68.79701232910156, 23.091894149780273, 2.264239549636841, 1.1032527685165405, 0.8432762622833252, 0.5348328351974487, 0.41480228304862976, 0.32325422763824463, 0.3198412358760834, 0.2750047743320465]
   #data = [[[keys[:5], values[:5]], [keys[5:], values[5:]]]]
   data = [keys, values]
   #data = {'results': [[[['Labrador_retriever', 'golden_retriever', 'English_foxhound', 'kuvasz', 'curly-coated_retriever', 'redbone', 'clumber', 'bloodhound', 'Sussex_spaniel', 'otterhound'], [68.79701232910156, 23.091894149780273, 2.264239549636841, 1.1032527685165405, 0.8432762622833252, 0.5348328351974487, 0.41480228304862976, 0.32325422763824463, 0.3198412358760834, 0.2750047743320465]]]]} 
   return {'results': data}

