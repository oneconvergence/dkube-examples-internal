from flask import Flask, request
import requests
import numpy as np
import os
import json
from PIL import Image
import sys

app = Flask(__name__, static_url_path='/tmp')
UPLOAD_FOLDER = '/tmp'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET'])
def keras_mnist_inference_server():
    return "Started server"

@app.route('/predict', methods=['POST'])
def start_inference():
    file = request.files['file']
    filename = file.filename
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    image = np.asarray(Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename)))
    image_data = np.reshape(image, (28, 28, 1))
    payload = {
            "signature_name":"predict", "instances": [{'images': image_data.tolist()}]
            }
    r = requests.post(app.config['TF_SERVING_URL'] + ':predict', json=payload)
    output = json.loads(r.content)['predictions']
    output_dict = {}
    for i in range (5):
        output_dict[i+5] = output[0][i]
    return str(output_dict)

if __name__ == "__main__":
    app.config['TF_SERVING_URL'] = sys.argv[1]
    app.run(host='0.0.0.0')
