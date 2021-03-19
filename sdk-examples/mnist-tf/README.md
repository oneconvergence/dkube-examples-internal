### SDK-Example (MNIST-DIGIT-CLASSIFICATION (TF))

1. Create a Dkube-IDE with framework as tensorflow and version as 1.14.
2. Download the pipeline.ipynb file from "https://github.com/oneconvergence/dkube-examples-internal/tree/sdk-eg-mnist/sdk-examples/mnist-tf/pipeline.ipynb" and upload.
3. Run all the cells of pipeline.ipynb file, it will automatically create a pipeline, a DKube run and a test inference.
4. Dkube-Webapp-Inference:
   - Go to https://:32222/inference
   - Copy the model serving URL from the test inference tab.
   - Copy the auth token from developer settings
   - Select model type mnist
   - Upload the digit image ("https://github.com/oneconvergence/dkube-examples-internal/tree/sdk-eg-mnist/sdk-examples/mnist-tf/1.png") and click on predict.


