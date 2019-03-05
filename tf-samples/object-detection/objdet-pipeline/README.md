# Steps to train object detection in dkube
## Prerequisites
### Prepare the dataset
The object detection datasets are available usually as annotations + images. Tensorflow object detection API requires the dataset to be in TFRecords format. Dkube does not support this conversion as of now. So the conversion needs to be done outside Dkube and the TFRecords can be stored in S3 or as K8s volume for use in Dkube.
### Prepare config file
Tensorflow object detection API expects the model parameters in a pipeline configuration file. This file needs to be updated with the dataset path and pretrained model path inside dkube. To facilitate that the user has to modify the pipeline config file as below:
- Set fine_tune_checkpoint: "MODEL_PATH/faster_rcnn_resnet101_coco_11_06_2017/model.ckpt"
- train_input_reader.input_path: "DATA_PATH/TFRecords/pet_faces_train.record-?????-of-00010"
- eval_input_reader.input_path: "DATA_PATH/TFRecords/pet_faces_val.record-?????-of-00010"
- The MODEL_PATH and DATA_PATH will be replaced with appropriate values inside Dkube. This configuration file needs to be available in the host machine. Sample config file for faster RCNN model is available here.
- https://github.com/oneconvergence/dkube-examples/blob/object-detection/tf-samples/object-detection/pipeline.config
### Prepare training job workspace
The workspace should contain the training program and the label map file. Alos for updating the config file, the user has to copy the processing script provided here into the workspace. This script will extract the dataset and model if they are compressed and update the config file accordingly. The training program should store the checkpoints generated in the path specified by $OUT_DIR. 
### Prepare export job workspace
The export workspace should contain the export program that accepts the model checkpoint directory as input and generates the saved_model.pb. This should store the generated model in the path specified by $OUT_DIR.
## Setup the environment
### Add training workspace
Select the repository where the training pgm is stored as the workspace
- Select Github
- Name : object_detection_tarining
- URL : path of the training program in github
### Add export workspace
Select the repository where the export pgm is stored as the workspace
- Select Github
- Name : object_detection_export
- URL : path of the export program in github
### Add dataset
Select the s3 bucket where the dataset is stored as TFRecords
- Select s3FileSystem
- Select AWS
- Name : pets
- Access Key :
- Secret key :
- Bucket : aws-dkube
- Prefix : obj-det-test
### Add pretrained model
Select the pretrained model for transfer learning as model
- Select other
- Name : faster rcnn
- URL : Public URL where the pretrained model is available
## Train the model
Select training jobs in dkube UI, and fill the following:
1. Name : object-detection-train
2. Framework : v1.12
3. Start-up script :
```bash
export S3_REQUEST_TIMEOUT_MSEC=60000
bash process.sh <c/n> (c- if data/model is compressed, n - otherwise)
python model_main.py --pipeline_config_path=$HOME/pipeline.config --model_dir=$OUT_DIR
```
4. Parameters
Upload file : select the pipeline.config file
5. Workspace : select the previously saved taring workspace
6. Model : select the pretrained model added
7. Dataset : select the loaded dataset
8. Start training
## Export the model
Select training jobs in dkube UI, and fill the following:
1. Name : object-detection-export
2. Framework : v1.12
3. Start-up script :
```bash
export S3_REQUEST_TIMEOUT_MSEC=60000
python export_inference_graph.py --input_type=image_tensor --pipeline_config_path=$HYPERPARAMS_JSON_FILEPATH --trained_checkpoint_prefix=$MODEL_PATH/$MODEL_NAME/out/model.ckpt-20000  --output_directory=$OUT_DIR
```
4. Parameters
- Upload file : select the pipeline.config file
5. Workspace : select the previously saved export workspace
6. Model : select the model generated during training
7. Start training
## Deploy the model
1. Select models tab
2. Select the model generated in the previous step
3. Deploy
The sering URL will be available in the inference tab in dkube UI
## Inference
1. Start the inference server on the machine where dkube is launched using the following command.
```bash
./run.sh create <unique name> <user name> ocdr/dkube-d3inf:<version> <serving url from dkube> <tag>
```
2. This command return the url where the UI can be accessed.
3. Go to this url and select image, label map file and provide the number of classes.
4. Select detect.
