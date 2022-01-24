# Steel Defect Detetion Example

## Adding repos

### 1. Add Code Repo:
1. Provide name: **Steel**
2. Source URL: https://github.com/oneconvergence/dkube-examples-internal/tree/master/tf/segmentation/steel-defect
3. Submit

### 2. Add Dataset repo 
1. Provide name: **steel-data**
2. Source: Other
3. URL: https://storage.googleapis.com/steeldata/severstal/severstal-steel-defect-detection.zip

### 3. Add dataset repo
1. Provide name: **steel-prep**
2. Versioning: DVS, Source: None

### 4. Add dataset repo
1. Provide name: **steel-train**
2. Versioning: DVS, Source: None

### 5. Add dataset repo
1. Provide name: **steel-test**
2. Versioning: DVS, Source: None

### 6. Add model repo
1. Provide name: **steel-model**
2. Versioning: DVS, Source: None

## Create and launch Jupyterlab with project steel
1. Go to workspace/steel
2. Run pipeline.ipynb to create the pipeline run.
3. The pipeline run will create a test inference at the end.

## Test Inference
1. open {setup-IP/URL}:32222/inference
2. Give Serving URL, and auth token from developer settings.
3. Choose model steel
4. Download and upload sample image https://github.com/oneconvergence/dkube-examples-internal/blob/master/tf/segmentation/test_sample.jpeg 
5. Click Submit
6. Expected Output
![](steel_output.png?raw=true)
