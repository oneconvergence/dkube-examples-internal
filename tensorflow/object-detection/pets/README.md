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
## Step1: Create a workspace
 1. Click *Workspaces* side menu option.
 2. Click *+Workspace* button.
 3. Select *Github* option.
 4. Enter a unique name say *pets-detector-preprocessing*
 5. Paste link *[https://github.com/oneconvergence/dkube-examples/tree/obj-det-1.2/tensorflow/object-detection/pets/program/preprocessing
 ](https://github.com/oneconvergence/dkube-examples/tree/obj-det-1.2/tensorflow/object-detection/pets/program/preprocessing)* in the URL text box.
 6. Click *Add Workspace* button.
 7. Workspace will be created and imported in Dkube. Progress of import can be seen.
 8. Please wait till status turns to *ready*.
## Step2: Download the Dataset
This step will download the images.tar.gz and annotations.tar.gz for Oxford IIT Pets dataset and create a dataset named "pets" in dkube.
1. Click *Jobs* side menu option.
 2. Click *+Data Preprocessing* button.
 3. Fill the fields in Job form and click *Submit* button. Toggle *Expand All* button to auto expand the form. See below for sample values to be given in the form, for advanced usage please refer to **Dkube User Guide**.
	- Enter a unique name say *download-pets-dataset*
	- Enter target dataset name for data being downloaded by the job say *pets*.
	- **Container** section - 
	    - Docker Image URL : docker.io/ocdr/dkube-datascience-preprocess:1.1
	    - Private : If image is private, select private and provide dockerhub username and password
	    - Start-up script : `python download.py`
	-  **Parameters** section - Leave it to default.
	- **Workspace** section - Please select the workspace *pet-detector-preprocessing* created in *Step1*.
4. Click *Submit* button.
5. A new entry with name *download-pets-dataset* will be created in *Data Preprocessing* table.
6. Check the *Status* field for lifecycle of job, wait till it shows *complete*.

## Step3: Preprocess Data (Conversion to TFRecord format)
This step converts the downloaded dataset to TFRecords, the format expected by tensorflow object detection API.
1. Click *Jobs* side menu option.
 2. Click *+Data Preprocessing* button.
 3. Fill the fields in Job form and click *Submit* button. Toggle *Expand All* button to auto expand the form. See below for sample values to be given in the form, for advanced usage please refer to **Dkube User Guide**.
	- Enter a unique name say *preprocess-pets-dataset*
	- Enter target dataset name for the preprocessed data say *tf-records*.
	- **Container** section - 
	    - Docker Image URL : docker.io/ocdr/dkube-datascience-preprocess:1.1
	    - Private : If image is private, select private and provide dockerhub username and password
	    - Start-up script : `python extract.py; python create_pet_tf_record.py --data_dir=/tmp/dataset/ --output_dir=$OUT_DIR --label_map_path=pet_label_map.pbtxt`
	-  **Parameters** section - Leave it to default.
	- **Workspace** section - Please select the workspace *pet-detector-preprocessing* created in *Step1*.
	- **Dataset** section - Please select the dataset *pets* created in *Step2*.
4. Click *Submit* button.
5. A new entry with name *preprocess-pets-dataset* will be created in *Data Preprocessing* table.
6. Check the *Status* field for lifecycle of job, wait till it shows *complete*.
# How to Train
## Step1: Create a workspace

 1. Click *Workspaces* side menu option.
 2. Click *+Workspace* button.
 3. Select *Github* option.
 4. Enter a unique name say *pet-detector*
 5. Paste link *[https://github.com/oneconvergence/dkube-examples/tree/obj-det-1.2/tensorflow/object-detection/pets/program/training 
 ](https://github.com/oneconvergence/dkube-examples/tree/obj-det-1.2/tensorflow/object-detection/pets/program/training )* in the URL text box.
 6. Click *Add Workspace* button.
 7. Workspace will be created and imported in Dkube. Progress of import can be seen.
 8. Please wait till status turns to *ready*.

## Step2: Add model for transfer learning
 1. Click *Models* side menu option.
 2. Click *+Model* button.
 3. Select *Other* option.
 4. Enter a unique name say *faster-rcnn*
 5. Paste link *[http://storage.googleapis.com/download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz](http://storage.googleapis.com/download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz)* in the URL text box.
 6. Click *Add Model* button.
 7. Model will be created and imported in Dkube. Progress of import can be seen.
 8. Please wait till status turns to *ready*.
## Step3: Start a training job
 1. Click *Jobs* side menu option.
 2. Click *+Training Job* button.
 3. Fill the fields in Job form and click *Submit* button. Toggle *Expand All* button to auto expand the form. See below for sample values to be given in the form, for advanced usage please refer to **Dkube User Guide**.
	- Enter a unique name say *pet-detector*
	- **Container** section
	  - Tensorflow Version - Leave with default options selected.
	  - Startup script - `bash process.sh; python model_main.py --model_dir=$OUT_DIR`
	- **GPUs** section - Provide the required number of GPUs. This field is optional, if not provided network will train on CPU.
	-  **Parameters** section
		- Select the pipeline.config file which is stored locally(Download from https://github.com/oneconvergence/dkube-examples/blob/1.2/tensorflow/object-detection/pets/program/training/pipeline.config)
		- Set the number of steps
	- **Workspace** section - Please select the workspace *pet-detector* created in *Step1(How to train)*.
	- **Model** section - Please select the workspace *faster-rcnn* created in *Step2(How to train)*.
	- **Dataset** section - Please select the dataset *tf-records* created in *Step3(How to Preprocess Data)*.
4. Click *Submit* button.
5. A new entry with name *pet-detector* will be created in *Jobs* table.
6. Check the *Status* field for lifecycle of job, wait till it shows *complete*.

# How to Serve

 1. After the job is *complete* from above step. The trained model will get generated inside *Dkube*. Link to which is reflected in the *Model* field of a job in *Job* table.
 2. Click the link to see the trained model details.
 3. Click the *Deploy* button to deploy the trained model for serving. A form will display.
 4. Input the unique name say *pet-detector-serving*
 5. Select *CPU* or *GPU* to deploy model on specific device. Unless specifically required, model can be served on CPU.
 6. Click *Deploy* button.
 7. Click *Inferences* side menu and check that a serving job is created with the name given i.e, *digits-serving*.
 8. Wait till *status* field shows *running*.
 9. Copy the *URL* shown in *Endpoint* field of the serving job.

# How to test Inference
1. To test inference **dkubectl** binary is needed.
2. Please use *dkube-notebook* for testing inference.
3. Create a file *pet-detector.ini* with below contents, Only field to be filled in is *modelurl*. Paste the *URL* copied in previous step.

    ```
    [INFAPP]
    #Name of the program to run, choices are - digits,catsdogs,cifar,objdetect,bolts
    program="objdetect"
    #Serving URL of the model in dkube
    modelurl=""
    #Container image to be used for inference app
    #image=""
    #IP to make inference app available on
    #accessip=""
    
    ################################################################################################################
    #    Following fields need not be filled in when launching inference application from inside dkube notebook    #
    ################################################################################################################
    #Name of the dkube user
    #user=""
    #Path to the kubeconfig of the cluster ex: "~/.dkube/kubeconfig"
    #kubeconfig=""
    ```
  4. Execute the command `dkubectl infapp launch --config pet-detector.ini -n pet-detector`
  5. The above command will output a URL, please click the URL to see the UI which can be used for testing inference.
  6. Upload an image for inference, images in **inference** folder can be used.
  7. Upload the labe map file in the file upload section. The pet_label_map.pbtxt file in **inference** cab be used.
  8. Set the Number of classes to 32
  7. Click *Detect* button and the image is displayed with detection boxes returned by the model.
