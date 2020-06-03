# Pet Detector
This example is derived from [tensorflow object detection example](https://github.com/tensorflow/models/tree/master/research/object_detection) and modified to run on Dkube Platform.

 - This program detects pet breeds from images. It uses transfer learning to train the model. The pre trained model used for transfer learning is "faster_rcnn_resnet101_coco_11_06_2017".
 - Modified program is configurable and takes Hyperparameters like steps, batchsize etc from Dkube UI and update the pipeline config file. 
 - Program is modified to export the trained model for inference.

# Directories

 - **program/preprocessing** : This directory has data prprocessing code files.
 - **program/training**: This directory has training code files implemented on top of Tensorflow framework. 
 - **inference**: This directory has compatible test data images which can be used for inference.

# How to Preprocess Data
Tensorflow object detection API expects the input dataset to be in TFRecord format. But the pet dataset available is in .jpg and .xml format. We need some preprocessing to convert this into TFRecord format.
## Step 1: Create a Project
 1. Click *Repos* in side menu under *WORKFLOW* section.
 2. Click *+Project* button under *Projects* section.
 3. Enter a unique name say *pets-detector-preprocessing* .
 4. Select *Project Source* as *Git*.
 5. Paste link *[https://github.com/oneconvergence/dkube-examples/tree/master/tensorflow/object-detection/pets/program/preprocessing
 ](https://github.com/oneconvergence/dkube-examples/tree/master/tensorflow/object-detection/pets/program/preprocessing)* in the URL text box.
 6. Click *Add Project* button.
 7. Enter branch name in *Branch* text-box.
 8. Project will be created and imported in Dkube. Progress of import can be seen.
 9. Please wait till status turns to *ready*.
## Step 2. Create Download Dataset DVS
This step is to create a dvs dataset which will hold the downloaded dataset. This will act as the output dataset for download preprocess run.
 1. Click *Repos* in side menu under *WORKFLOW* section.
 2. Click *+Dataset* button under *Datasets*.
 3. Enter a unique name say *pets-download-dataset* .
 4. Select *Dataset Source* as *None*.
 5. Click *Add Dataset* button.
 6. Dataset will be created in Dkube. 
 7. Please wait till status turns to *ready*.
## Step 3: Download the Dataset
This step will download the images.tar.gz and annotations.tar.gz for Oxford IIT Pets dataset and create a dataset named "pets" in dkube.
 1. Click *Runs* side menu under *WORKFLOW* section.
 2. Click *+Run* and select *Preprocessing* button.
 3. Fill the fields in Job form and click *Submit* button. See below for sample values to be given in the form, for advanced usage please refer to **Dkube User Guide**.
    - **Basic** tab
	  - Enter a unique name say *download-pets-dataset* 
	  - Docker Image URL : docker.io/ocdr/dkube-datascience-preprocess:1.2
	  - Private : If image is private, select private and provide dockerhub username and password
	  - Start-up script : `python download.py`
	  - Click *Next*.
	- **Repos** tab
	  - *Inputs* section
	    - Project: Click on **+** button and select *pets-detector-preprocessing* project.
	  - *Outputs* section
	    - Dataset: Click on **+** button and select *pets-download-dataset*.
	    - Mount path: Enter path say */opt/dkube/output*.
	    - Click *Next*.
4. Click *Submit* button.
5. Check the *Status* field for lifecycle of Preprocessing run under *All Runs* section, wait till it shows *complete*.

## Step 4. Create TF-record Dataset DVS
This step is to prepare a *DVS* dataset for storing output of preprocess job which will convert downloaded dataset into *tf-record*.
 1. Click *Repos* in side menu under *WORKFLOW* section.
 2. Click *+Dataset* button under *Datasets*.
 3. Enter a unique name say *preprocess-pets-dataset* .
 4. Select *Dataset Source* as *None*.
 5. Click *Add Dataset* button.
 6. Dataset will be created in Dkube. 
 7. Please wait till status turns to *ready*.
 
## Step 5: Preprocess Data (Conversion to TFRecord format)
This step converts the downloaded dataset to TFRecords, the format expected by tensorflow object detection API.
 1. Click *Runs* side menu under *WORKFLOW* section.
 2. Click *+Run* and select *Preprocessing* button.
 3. Fill the fields in Job form and click *Submit* button. See below for sample values to be given in the form, for advanced usage please refer to **Dkube User Guide**.
    - **Basic** tab
	  - Enter a unique name say *preprocess-pets-dataset* 
	  - Docker Image URL :  docker.io/ocdr/dkube-datascience-preprocess:1.2
	  - Private : If image is private, select private and provide dockerhub username and password
	  - Start-up script : `python extract.py; python create_pet_tf_record.py --label_map_path=pet_label_map.pbtxt`
	  - Click *Next*.
	- **Repos** tab
	  - *Inputs* section
	    - Project: Click on **+** button and select *pets-detector-preprocessing* project.
	    - Dataset: Click on **+** button and select *pets-download-dataset*.
	    - Mount path: Enter mount path say */opt/dkube/input*
	  - *Outputs* section
	    - Dataset: Click on **+** button and select *preprocess-pets-dataset*.
	    - Enter mount path: Enter path say */opt/dkube/output*.
	    - Click *Next*.
4. Click *Submit* button.
5. Check the *Status* field for lifecycle of Preprocessing run under *All Runs* section, wait till it shows *complete*.

# How to Train
## Step 1: Create a Project

1. Click *Repos* in side menu under *WORKFLOW* section.
 2. Click *+Project* button under *Projects* section.
 3. Enter a unique name say *pets-detector-training* .
 4. Select *Project Source* as *Git*.
 5. Paste link *[https://github.com/oneconvergence/dkube-examples/tree/master/tensorflow/object-detection/pets/program/training](https://github.com/oneconvergence/dkube-examples/tree/master/tensorflow/object-detection/pets/program/training)* in the URL text box.
 6. Click *Add Project* button.
 7. Enter branch name in *Branch* text-box.
 8. Project will be created and imported in Dkube. Progress of import can be seen.
 9. Please wait till status turns to *ready*.

## Step 2: Add model for transfer learning
This step will download *faster-rcnn* object detection model which we will use to perform transfer learning.
 1. Click *Repos* in side menu under *WORKFLOW* section.
 2. Click *+Model* button under *Models*.
 3. Enter a unique name say *faster-rcnn* .
 4. Select *Model Source* as *Other*.
 5. Paste link *[http://storage.googleapis.com/download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz](http://storage.googleapis.com/download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz)* in the URL text box.
 6. Select extract uploaded file checkbox.
 7. Click *Add Model* button.
 8. Model will be created in Dkube. 
 9. Please wait till status turns to *ready*.

## Step 3. Create Output Model DVS
This step is to create a dvs model which will hold the trained output model. 
 1. Click *Repos* in side menu under *WORKFLOW* section.
 2. Click *+Model* button under *Models*.
 3. Enter a unique name say *pets-detector* .
 4. Select *Model Source* as *None*.
 5. Click *Add Model* button.
 6. Model will be created in Dkube. 
 7. Please wait till status turns to *ready*.

## Step 4: Start a training job
 1. Click *Runs* side menu under *WORKFLOW* section.
 2. Click *+Run* and select *Training* button.
 3. Fill the fields in Job form and click *Submit* button. See below for sample values to be given in the form, for advanced usage please refer to **Dkube User Guide**.
    - **Basic** tab
	  - Enter a unique name say *training-pets-detector* 
	  - Start-up script : `bash process.sh; python model_main.py`
	  - Click *Next*.
	- **Repos** tab
	  - *Inputs* section
	    - Project: Click on **+** button and select *pets-detector-training* project.
	    - Dataset: Click on **+** button and select *preprocess-pets-dataset*.
	    - Mount path: Enter mount path say */opt/dkube/input/dataset*
	    - Models: Click on **+** button and select *faster-rcnn*.
	    - Mount path: Enter mount path say */opt/dkube/input/model*
	  - *Outputs* section
	    - Models: Click on **+** button and select *pets-detector*.
	    - Enter mount path: Enter path say */opt/dkube/output*.
	    - Click *Next*.
	  - *Configuration* section
	    - Parameters upload configuration: Select the pipeline.config file which is stored locally(Download from [https://github.com/oneconvergence/dkube-examples/tree/master/tensorflow/object-detection/pets/program/training/pipeline.config](https://github.com/oneconvergence/dkube-examples/tree/master/tensorflow/object-detection/pets/program/training/pipeline.config))
4. Click *Submit* button.
5. Check the *Status* field for lifecycle of Training run under *All Runs* section, wait till it shows *complete*.

# How to Serve
After the job is *complete* from above step. The trained model will get generated inside *Dkube*.
 1. Click on the training run named *training-pets-detector*.
 2. Select *Lineage* tab beside the *Summary* details page.
 3. Click on *OUTPUTS* model *pets-detector*.
 4. Click on *Test Inference* button. 
 5. Enter a meaningful name for inference job say *pets-inference*.
 6. Select CPU/GPU. A button named *Test Inference* appear.
 7. Click on *Test inference* button.
 8. Click *Test Inferences* in side menu under *WORKFLOW* section.
 9. Wait till *status* field shows *running*.
 10. Copy the *URL* shown in *Endpoint* field of the serving job.
 
# How to test Inference
1. To test inference open a new tab with link < DKUBE_URL:port/inference >
2. Paste the *URL* shown in *Endpoint* field of the serving job.
3. Copy and Paste *Dkube OAuth token* from *Developer Settings* present in menu on top right to *Authorization Token* present in Inference page.
4. Select Model type as *objdetect*.
5. Set the Number of classes to 37.
6. Upload an image for inference, images in **inference** folder can be used.
7. Upload the labe map file in the file upload section. The pet_label_map.pbtxt file in **inference** cab be used.
8. Click _Predict_ button and the image is displayed with detection boxes returned by the model.
