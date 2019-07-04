### Anjani-Tiles Pipeline

#### Install the Kubeflow Pipelines SDK
sudo apt install jq -y                
latest_version=$(curl --silent https://api.github.com/repos/kubeflow/pipelines/releases/latest | jq -r .tag_name)      
pip3 install https://storage.googleapis.com/ml-pipeline/release/${latest_version}/kfp.tar.gz --upgrade

#### DSL Compile
dsl-compile --py anjani-tiles.py --output anjani-tiles.tar.gz

#### Upload this generated tarball into the Pipelines UI

### Prerequisite

1. Create Workspace with name: **Anjani-tiles-videos**
> github link: https://github.com/oneconvergence/dkube-examples/tree/ai-hub/pipeline/tiles_detection

2. Create AWS S3 Dataset with name: **Anjani-tiles-s3-bucket**

  > Bucket: anjani-tiles-dataset  
  > prefix: /videos

3. Run the Pipeline
