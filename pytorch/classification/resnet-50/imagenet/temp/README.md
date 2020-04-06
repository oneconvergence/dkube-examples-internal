Instructions to run pytorch job in dkube and inference in kfserving:

1. Create workspace/project using from below github link:
https://github.com/oneconvergence/dkube-examples/tree/pytorch-examples/pytorch/classification/resnet-50/imagenet/classification/program

2. Add dataset (tiny-imagenet) from following link and extrack it:
http://cs231n.stanford.edu/tiny-imagenet-200.zip

3. Create custom job using pytorch image (docker.io/ocdr/pytorch:v1).
   docker.io/ocdr/pytorch:v1 has pytorch 1.4 with cuda 10.1.
   Note that cuda version on GPU should be same as that of in pytorch image.

4. Create custom job with above details.

Inference steps:
1. Use prebuild kfserving docker image (docker.io/ocdr/pytorchserver:v2)
2. Build kfserving pytorchserver image if one want to modify:
   a. Clone kfserving repo as:
      https://github.com/kubeflow/kfserving.git
   b. Replace kfserving/python/pytorchserver with pytorchserver code in this repo as:
      rm -rf kfserving/python/pytorchserver
      cp -r pytorchserver kfserving/python/.
      cp pytorch.Dockerfile kfserving/python/.
   c. Build pytorch docker image as:
      cd kfserving/python
      docker build -t ocdr/pytorchserver:v2 -f pytorch.Dockerfile .
   
3. Edit kfserving config install/0.2.2/kfserving.yaml with pytorchserver image.
   Following is the diff of changes:
            "pytorch": {
-            "image": "gcr.io/kfserving/pytorchserver",
-            "defaultImageVersion": "0.2.2",
+            "image": "docker.io/ocdr/pytorchserver",
+            "defaultImageVersion": "v2",
             "allowedImageVersions": [
-               "0.2.2"
+               "v2"

4. Install/Update kfserving as:
   kubectl apply -f install/0.2.2/kfserving.yaml

5. Update s3/minio creads in s3_secret.yaml and apply the conf as:
   kubectl -f s3_secret.yaml

6. Deploy pytorch application by updating storageUri (model path) in kf_pytorch.yaml and deploy it as:
   kubectl -f kf_pytorch.yaml

7. Build & Deploy inference server as:
   sudo docker build -t inference:v1 -f inference.dockerfile .   
   kubectl apply -f inference.yaml

8. Open https://<server-ip>:31123/inference and select
   Model Serving URL as:  http://<istio-ingressgateway-ip>/v1/models/pytorch-resnet50:predict
   Program: catsdogs and upload image for inference.
   istio-ingressgatewayip is CLUSTER-IP from following command:
   kubectl -n istio-system get service istio-ingressgateway
