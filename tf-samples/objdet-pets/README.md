# Steps to train object detection in dkube
This program is to detect pet breeds from images. It uses transfer learning to train the model. The pre trained model used for transfer learning is "faster_rcnn_resnet101_coco_11_06_2017".
## Add workspace
This workspace contains the code for downloading the dataset, extarcting the datset, preprocessing, and training. 
1. Select the git repository where the training pgm is stored as the workspace
- Select Github
- Name : object-detection
- URL : https://github.com/oneconvergence/dkube-examples/tree/master/tf-samples/objdet-pets
## Add dataset 
Tensorflow object detection API expects the input dataset to be in TFRecord format. But the pet dataset available is in .jpg and .xml format. We need some preprocessing to convert this into TFRecord format. 
### Download dataset
Use the below command to download the images and annotations for pets dataset. The resultant dataset <Pets> will be available in the dataset tab in dkube.
- dkubectl data download start --config <download.ini>
```bash
[DATA_DOWNLOAD]
dkubeURL=""
token=""
name="Pets"
script="python /home/dkube/objdet-pets/download.py"
image="ocdr/dkube-datascience-preprocess:1.1"
tags=["something"]
```
- This will download the images.tar.gz and annotations.tar.gz for Oxford IIT  Pets dataset and create a dataset named "Pets" in dkube.
### Preprocess data
Use the below command to convert the downloaded dataset to TFRecord format. The resultant dataset <TFRecords> will be available in the dataset tab in dkube.
- dkubectl data preprocess start --config <preprocess.ini>
```bash
[DATA_PREPROCESS]
dkubeURL=""
token=""
name="TFRecords"
script="python /home/dkube/objdet-pets/extract.py; python /home/dkube/objdet-pets/create_pet_tf_record.py --data_dir=/tmp/dataset/ --output_dir=$OUT_DIR --label_map_path=/home/dkube/objdet-pets/pet_label_map.pbtxt"
datasets=["Pets"]
image="ocdr/dkube-datascience-preprocess:1.1"
tags=["sometag"]

```
## Add model for transfer learning
- Select the pretrained model for transfer learning as model
- Select other
- Name : faster rcnn
- URL : http://storage.googleapis.com/download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz
## Prepare config file
Tensorflow object detection API expects the model parameters in a pipeline configuration file. This file needs to be updated with the dataset path and pretrained model path inside dkube. To facilitate that the user has to modify the pipeline config file as below:
- Set fine_tune_checkpoint: "MODEL_PATH/faster_rcnn_resnet101_coco_11_06_2017/model.ckpt"
- train_input_reader.input_path: "DATA_PATH/pet_faces_train.record-?????-of-00010"
- eval_input_reader.input_path: "DATA_PATH/pet_faces_val.record-?????-of-00010"
- The MODEL_PATH and DATA_PATH will be replaced with appropriate values inside Dkube. This configuration file needs to be available in the host machine. Sample config file for faster RCNN model is available here.
- https://github.com/oneconvergence/dkube-examples/blob/master/tf-samples/objdet-pets/pipeline.config

## Start a training job
1. Name : obj-det
2. Framework : v1.12
3. Start-up script :
```bash
bash process.sh
python model_main.py --model_dir=$OUT_DIR
```
4. Parameters
- Upload file : select the pipeline.config file here if file is stored loacally.
- Set the number of steps
5. Workspace : select "object-detection"
6. Model : select "faster-rcnn"
7. Dataset : select "TFRecords"
8. Start training
## Deploy the model
1. Select models tab
2. Select the model generated in the previous step
3. Deploy
The sering URL will be available in the inference tab in dkube UI
## Inference
1. Start the inference server on the machine where dkube is launched using the following command.
```bash
dkubectl infapp launch --config infapp.ini -n objdet
```
```
[INFAPP]
#Path to the kubeconfig of the cluster
kubeconfig=<>
#Name of the program to run, choices are - mnist,catsdogs
program="objdet"
#Name of the dkube user
user=<>
#Serving URL of the model in dkube
modelurl=<>
#Container image to be used for inference app
image="ocdr/dkube-d3inf:1.1"
#IP to make inference app available on
accessip=<>
```
2. This command return the url where the UI can be accessed.
3. Go to this url and select image, label map file and provide the number of classes.
4. Select detect.
