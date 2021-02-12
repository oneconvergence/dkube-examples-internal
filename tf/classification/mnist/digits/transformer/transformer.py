# Copyright 2019 kubeflow.org.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

DEFAULT_MODEL_NAME = "model"

parser = argparse.ArgumentParser(parents=[kfserving.kfserver.parser])
parser.add_argument('--model_name', default=DEFAULT_MODEL_NAME,
                    help='The name that the model is served under.')
parser.add_argument('--predictor_host', help='The URL for the model predict function', required=True)

args, _ = parser.parse_known_args()

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
def convert(input_file):
    try:
        image = np.asarray(Image.open(input_file))
        image = np.true_divide(image,[255.0],out=None)
        image = np.reshape(image, (1, 784))
        image = image.astype(np.float32)
        data = json.dumps(image, cls=NumpyEncoder)
        data = data[1:-1]
        data = eval(data)
    except Exception as err:
        msg = "Failed to convert input image. " + str(err)
        logging.error(msg)
        return "", msg
    return data, ""



def cleanup(input_file, convertor_file):
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


class ImageTransformer(kfserving.KFModel):
    def __init__(self, name: str, predictor_host: str):
        super().__init__(name)
        self.predictor_host = predictor_host

    def preprocess(self, inputs: Dict) -> Dict:
        #return {'instances': [image_transform(instance) for instance in inputs['instances']]}        
        del inputs['instances']
        logging.info("prep =======> %s",str(inputs['token']))
        try:
            json_data = inputs
        except ValueError:
            return json.dumps({ "error": "Recieved invalid json" })
        model_inputs  = model_method = script = None
        input_file = ""
        input_type_mapping = {}
        in_signature = json_data["signatures"]["name"]
        in_data = json_data["signatures"]["inputs"]
        rqst_list = []
        batch_list = {}        
        for batch in in_data:
            for element in batch:
                batch_list.clear()
                key = "image"
                data = element["data"]
                if data:
                    input_file = "/tmp/input_" +key
                    b64_filewriter(input_file, data)
                output, err = convert(input_file)
                if err != "":
                    return json.dumps({ "error": err })
                if output == '' or output == b'':
                    cleanup(input_file, script)
                    return json.dumps({ "error": "Script did not execute successfuly" })
                if isinstance(output, (bytes, bytearray)):
                    output = output.decode('utf-8')
                batch_list[key] = output
                cleanup(input_file, script)
            rqst_list.append(batch_list)
        res = {"signature_name":in_signature,"instances":rqst_list,"token":inputs['token']}
        return res

    def postprocess(self, inputs: List) -> List:
        return inputs

if __name__ == "__main__":
    transformer = ImageTransformer(args.model_name, predictor_host=args.predictor_host)
    kfserver = kfserving.KFServer()
    kfserver.start(models=[transformer])


