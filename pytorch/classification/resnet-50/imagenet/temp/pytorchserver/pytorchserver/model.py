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
import os
from typing import Dict
import torch
import importlib
import sys
import torch

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from collections import OrderedDict
#from preprocess import preprocess
import pytorchserver.preprocess as preprocess
#from torchvision import datasets, transforms

PYTORCH_FILE = "resnet50-dkube.pth"


class PyTorchModel(kfserving.KFModel):
    def __init__(self, name: str, model_class_name: str, model_dir: str):
        super().__init__(name)
        self.name = name
        #self.model_class_name = model_class_name
        self.model_dir = model_dir
        self.ready = False
        self.model = None
        #self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        print("Model name: %s, Model Dir: %s" % (name, model_dir))

    def load(self):
        #model_file_dir = kfserving.Storage.download(self.model_dir)
        model_file_dir = kfserving.Storage.download(self.model_dir)
        model_file = os.path.join(model_file_dir, PYTORCH_FILE)
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.__dict__["resnet50"]()
        checkpoint = torch.load(model_file, map_location=self.device)
        new_state_dict = OrderedDict()

        for k, v in checkpoint['state_dict'].items():
           name = k[7:] # remove `module.`
           new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
        self.model.eval()
        self.ready = True
        print("Model is ready")

    def predict(self, in_request: Dict) -> Dict:
        inputs = []
        with torch.no_grad():
            try:
                request = preprocess.preprocess(in_request)
                inputs = torch.tensor(request["instances"]).to(self.device)
            except Exception as e:
                raise TypeError(
                    "Failed to initialize Torch Tensor from inputs: %s, %s" % (e, inputs))
            try:
                output = self.model(inputs)
                res = preprocess.postprocess(output)
                print(res)
                return res
            except Exception as e:
                raise Exception("Failed to predict %s" % e)
