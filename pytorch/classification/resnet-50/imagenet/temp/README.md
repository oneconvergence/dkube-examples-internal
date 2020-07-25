# Instructions to run pytorch job in dkube and inference in kfserving:

1. Create workspace/project using from below github link:
https://github.com/oneconvergence/dkube-examples/tree/pytorch-examples/pytorch/classification/resnet-50/imagenet/classification/program

2. Add dataset (tiny-imagenet) from following link and extrack it manually:
http://cs231n.stanford.edu/tiny-imagenet-200.zip

3. Create custom job using pytorch image (docker.io/ocdr/pytorch:v1).
   docker.io/ocdr/pytorch:v1 has pytorch 1.4 with cuda 10.1.
   Note that cuda version on GPU should be same as that of in pytorch image.

4. Create custom job with above details.

# Inference steps
  Install kfserving on different cluster where dkube is not installed.

## Install Istio and knative-serving if not installed already.

### Istio installation:
   
   ```
   sudo apt-get update
   export ISTIO_VERSION=1.1.7
   curl -L https://git.io/getLatestIstio | ISTIO_VERSION=1.2.2 sh -
   cd istio-1.2.2/
   for i in install/kubernetes/helm/istio-init/files/crd*yaml; do kubectl apply -f $i; done

   Add following in istio-ns.yaml
   {
   apiVersion: v1
   kind: Namespace
   metadata:
     name: istio-system
     labels:
       istio-injection: disabled
   }

   kubectl apply -f istio-ns.yaml

   sudo snap install helm --classic
   helm template --namespace=istio-system \
     --set sidecarInjectorWebhook.enabled=true \
     --set sidecarInjectorWebhook.enableNamespacesByDefault=true \
     --set global.proxy.autoInject=disabled \
     --set global.disablePolicyChecks=true \
     --set prometheus.enabled=false \
     `# Disable mixer prometheus adapter to remove istio default metrics.` \
     --set mixer.adapters.prometheus.enabled=false \
     `# Disable mixer policy check, since in our template we set no policy.` \
     --set global.disablePolicyChecks=true \
     `# Set gateway pods to 1 to sidestep eventual consistency / readiness problems.` \
     --set gateways.istio-ingressgateway.autoscaleMin=1 \
     --set gateways.istio-ingressgateway.autoscaleMax=1 \
     --set gateways.istio-ingressgateway.resources.requests.cpu=500m \
     --set gateways.istio-ingressgateway.resources.requests.memory=256Mi \
     `# More pilot replicas for better scale` \
     --set pilot.autoscaleMin=2 \
     `# Set pilot trace sampling to 100%` \
     --set pilot.traceSampling=100 \
     install/kubernetes/helm/istio \
     > ./istio.yaml

   kubectl apply -f istio.yaml
   ```

### Knative-serving installation:
   
   ```
   kubectl apply --selector knative.dev/crd-install=true --filename https://github.com/knative/serving/releases/download/v0.8.0/serving.yaml
   kubectl apply --filename https://github.com/knative/serving/releases/download/v0.8.0/serving.yaml
   ```
## kfserving installation
   1. Clone kfserving repo
   ```https://github.com/kubeflow/kfserving.git```
   
   2. Update install/0.2.2/kfserving.yaml with following to use kustomized pytorch server docker image for deployment.
   ```
            "pytorch": {
-            "image": "gcr.io/kfserving/pytorchserver",
-            "defaultImageVersion": "0.2.2",
+            "image": "docker.io/ocdr/pytorchserver",
+            "defaultImageVersion": "v2",
             "allowedImageVersions": [
-               "0.2.2"
+               "v2"
   ```
   
   3. Install kfserving as:
   
   ```kubectl apply -f install/0.2.2/kfserving.yaml```

### Follow below steps to update pytorch server docker image and repeat above step 2 & 3. (Optional)

   1. Replace kfserving/python/pytorchserver with [pytorchserver](./pytorchserver) code in this repo as:
      ```
      rm -rf kfserving/python/pytorchserver
      cp -r pytorchserver kfserving/python/.
      cp pytorch.Dockerfile kfserving/python/.
      ```
   2. Build pytorch docker image as:
      ```
      cd kfserving/python
      docker build -t ocdr/pytorchserver:v2 -f pytorch.Dockerfile .
      ```

## Update s3/minio creads
   Edit [s3_secret.yaml](./s3_secret.yaml) with s3/minio creds where trained pytorch model is located and apply the conf as:
   
   ```kubectl apply -f s3_secret.yaml```

## Deploy pytorch application
   Update storageUri (model path) in [kf_pytorch.yaml](./kf_pytorch.yaml) and deploy it as:
   
   ```kubectl apply -f kf_pytorch.yaml```

## Build & Deploy inference server as:

   ```
   sudo docker build -t inference:v1 -f inference.dockerfile .   
   kubectl apply -f inference.yaml
   ```
   
## Run inference

   1. Browse ```https://<server-ip>:31123/inference```
   
   2. Get istio-ingressgateway ip as CLUSTER-IP from output following command:
   
      ```kubectl -n istio-system get service istio-ingressgateway```

   3. Select Model Serving URL as: ```http://<istio-ingressgateway-ip>/v1/models/pytorch-resnet50:predict```
      Program: catsdogs and upload image for inference.
