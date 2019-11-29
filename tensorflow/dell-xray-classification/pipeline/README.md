## Tiles Pipeline Notebook with videos as input

### Prerequisite

1. Create Workspace with name: **Anjani-tiles-videos**
> github link: https://github.com/oneconvergence/dkube-apps/tree/tiles_defect_detection_video_processing/tiles_detection

2. Create AWS S3 Dataset with name: **Anjani-tiles-s3-bucket**

  > Bucket: anjani-tiles-dataset  
  > prefix: /videos

3. Launch Notebook with workspace as **Anjani-tiles-videos**         
4. change directory to **program/pipeline/notebooks** and open Notebook **dkube-tiles-pipeline.ipynb**           

5. Update access_url in **Create and Run pipeline** cell with URL at which dkube is accessible, copy paste from the browser of this window.

6. Run the notebook
