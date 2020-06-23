README explains how to write a code & build image for transformer stage of serving of model trained by the below DKube example.

*https://github.com/oneconvergence/dkube-examples/tree/2.0.7/tensorflow/classification/resnet/catsdogs/classifier*

Why is this required?
====================
Transfomer stage is necessary when there is certain preprocessing involved on the inputs before calling the predict method which will pass the processed input to model for inference.

How to write & build transformer
================================
Implement a transformer.py and place in this project directory.
The below function is mandatory to be implemented.

*def preprocess(inputs: Dict) -> Dict:*

Please see the sample *transformer.py*

Update the python package requirements in *requirements.txt*

Build the container using - docker build -t <registry/image:tag> .

Push in the container registry and make the image **public**

NOTE - Its very important to return *token* in the returned response. See below line in the example.

*res = {"signature_name":in_signature,"examples":rqst_list,"token":inputs['token']}*

Use in DKube
============
Use the transformer image in *Test Inference* functionality of DKube.
Supply the image built as input while deploying the model for inference.


How to test
===========
This package shows the *transformer.py* for DKube cats-dogs example which is built on resnetv2 model.
This package has an example image file which can be used with the below *curl* command.

*example/cat.png* - Actual cat image to be used for inference testing
*example/image.json* - JSON input expected by *transformer.py* - Image is sent as B64 encoded bytes (standard way)
    - Please update the *serving_url* and *token* fields in *image.json*


curl -v --insecure -H "Authorization: Bearer eyJhbGciOiJSUzI1NiIsImtpZCI6Ijc0YmNkZjBmZWJmNDRiOGRhZGQxZWIyOGM2MjhkYWYxIn0.eyJ1c2VybmFtZSI6Im9jZGt1YmUiLCJyb2xlIjoib3BlcmF0b3IiLCJleHAiOjQ4MzI1NTA3MTAsImlhdCI6MTU5MjU1MDcxMCwiaXNzIjoiREt1YmUifQ.IADw32O4Y5eX_PxYB7vXIU583U0XsEMovETOvI3_xrg26JACKr7CpquUem-OxZWqcB-aplfyJmiKKcQkw4llX_gheBM6LFkj0IDKNmoG7BhYg2g7rko90b8D2-O8aLkZw9XoiRHoSFW5ZJ5HzVSWaqFeRe-vZpNUYbHJpNX3C8eLWkAJlUAJ-H-jFmnqw5vMyUJedwXFcZgHM_XiCNUNOR5hOkcbRq16lH10uwifLFHFp9fQT9y9kKaTcZYEkpAwtXX272tIk3JegcsKFCS58zC1jsOWH6UF7t-4DX-sOLyG83tXTywWAzeq9hwLQZUFjhCa2J2KKZSEcmLS7TO35A" https://34.72.97.235:32222/dkube/inf/v1/models/d3-inf-job1-ocdkube:predict -d @image.json
