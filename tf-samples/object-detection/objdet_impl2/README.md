# Steps to train object detection in dkube
## Set up  the environment
### Add workspace
1. Copy the below script and add it along with the training program in the workspace.
- https://github.com/oneconvergence/dkube-examples/blob/object-detection/tf-samples/object-detection/objdet/process.sh
- The training program should store the checkpoints generated in the path specified by $OUT_DIR.
- The workspace should contain the pipeline.config file
2. Select the repository where the training pgm is stored as the workspace
- Select Github
- Name : object_detection
- URL : https://github.com/oneconvergence/dkube-examples/tree/object-detection/tf-samples/object-detection/objdet
### Add dataset 
- Select the s3 bucket where the dataset is stored as TFRecords
- Select s3FileSystem
- Select AWS
- Name : pets
- Access Key :
- Secret key :
- Bucket : aws-dkube
- Prefix : obj-det-test
### Add model for transfer learning
- Select the pretrained model for transfer learning as model
- Select other
- Name : faster rcnn
- URL : http://storage.googleapis.com/download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz
### Prepare config file
Tensorflow object detection API expects the model parameters in a pipeline configuration file. This file needs to be updated with the dataset path and pretrained model path inside dkube. To facilitate that the user has to modify the pipeline config file as below:
- Set fine_tune_checkpoint: "MODEL_PATH/faster_rcnn_resnet101_coco_11_06_2017/model.ckpt"
- train_input_reader.input_path: "DATA_PATH/TFRecords/pet_faces_train.record-?????-of-00010"
- eval_input_reader.input_path: "DATA_PATH/TFRecords/pet_faces_val.record-?????-of-00010"
- The MODEL_PATH and DATA_PATH will be replaced with appropriate values inside Dkube. This configuration file needs to be available in the host machine. Sample config file for faster RCNN model is available here.
- https://github.com/oneconvergence/dkube-examples/blob/object-detection/tf-samples/object-detection/pipeline.config
## Start a training job
1. Name : obj-det
2. Framework : v1.12-objdet
3. Start-up script :
```bash
bash process.sh <c/n> (c- if data/model is compressed, n - otherwise)
python model_main.py --pipeline_config_path=$HOME/pipeline.config --model_dir=$OUT_DIR
```
4. Workspace : select the previously saved workspace
5. Model : select the model downloaded
6. Dataset : select the loaded dataset
7. Start training
## Deploy the model
1. Select models tab
2. Select the model generated in the previous step
3. Deploy
The sering URL will be available in the inference tab in dkube UI
## Inference
1. Start the inference server on the machine where dkube is launched using the following command.
```bash
./run.sh create name=<name> user=<user name> program=objdet model_serving_url=<model serving url> image=<image> access_ip=<access ip>
```
2. This command return the url where the UI can be accessed.
3. Go to this url and select image, label map file and provide the number of classes.
4. Select detect.
