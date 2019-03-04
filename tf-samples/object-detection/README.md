Object Detection Model
----------------------
This model is to detect various breeds of cats and dogs.
It uses the Oxford IIIT-pets dataset.

The output
of the detector will look like the following:

![](img/oxford_pet.png)

There are two verson of this detector.
1. Non pipelined implementation: The training program does the training as well as exporting of the model.

2. Pipelined implementation: It has two components. The training component will train the model and save the checkpoints. The export component will generate the saved_model fom the saved checkpoints.

Object detection is about not only finding the class of object but also localising the extent of an object in the image. The TensorFlow Object Detection API is an open source framework built on top of TensorFlow that makes it easy to construct, train and deploy object detection models. 

DKube supports the tensorflow object detection APIs in all the Tensorflow versions(1.10, 1.11, 1.12) for both gpu and cpu.

The Tensorflow object detection API expects the dataset to be in TFRecord format. So the user has to convert the Image and annotation dataset to TFRecord format and use the TFRecord format data as dataset when training an object detection model in Dkube. This conversion needs to be done outside of Dkube as the current version of Dkube does not support data preprocessing.

In order to train an object detection model in dkube, the user needs the following artifacts:
Main program - given in workspace as a python file
Label map file - given in workspace as .pbtxt file
Pipeline config file - stored in the host machine
Dataset - in TFRecord format
Pretrained model
Processing script - in the workspace as .sh file

A preprocessing script is used in dkube to update the MODEL and DATA directory paths in configuration file. This script will extract the dataset or model if it is in compressed format and update the corresponding paths in the config file. The user has to add this script in the workspace to make it work in dkube.

