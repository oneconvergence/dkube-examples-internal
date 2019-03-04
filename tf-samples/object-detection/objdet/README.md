#Steps to train object detection in dkube
##Step 1 : Stet up  the environment
1. Copy the below script and add it along with the training program in the workspace.
https://github.com/oneconvergence/dkube-examples/blob/object-detection/tf-samples/object-detection/objdet/process.sh

2. Select the repository where the training pgm is stored as the workspace
Select Github
Name : object_detection
URL : https://github.com/tijithomas/Object-Detection/tree/with_script/obj-det
3. Select the s3 bucket where the dataset is stored as TFRecords
Select s3FileSystem
Select AWS
Name : pets
Access Key :
Secret key :
Bucket : aws-dkube
Prefix : obj-det-test
4. Select the pretrained model for transfer learning as model
Select other
Name : faster rcnn
URL : http://storage.googleapis.com/download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz
##Step 2 : Start a training job
1. Name : obj-det
2. Framework : v1.12-objdet
3. Start-up script :
```export S3_REQUEST_TIMEOUT_MSEC=60000
bash process.sh <c/n> (c- if data/model is compressed, n - otherwise)
python model_main.py --pipeline_config_path=$HOME/pipeline.config --model_dir=$OUT_DIR```
4. Parameters
Upload file : select the pipeline.config file
In the pipeline.config file 
Set fine_tune_checkpoint: "MODEL_PATH/faster_rcnn_resnet101_coco_11_06_2017/model.ckpt"
train_input_reader.input_path: "DATA_PATH/TFRecords/pet_faces_train.record-?????-of-00010"
eval_input_reader.input_path: "DATA_PATH/TFRecords/pet_faces_val.record-?????-of-00010"
5. Workspace : select the previously saved workspace
6. Model : select the model downloaded
7. Dataset : select the loaded dataset
8. Start training

