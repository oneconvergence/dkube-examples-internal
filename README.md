# dkube-examples

This repository contains docker file examples to build Docker image for DKube jobs.

Docker files:

- **docker-images/dkube-tf-141.Dockerfile** is an example with using base image as DKube default tf image.
- **docker-images/arvados-tfjl-141.Dockerfile** is an example dockerfile for building DKube job image using other image, here tensorflow 1.14 image has been taken for example. 


## Build image

> `sudo docker build . -t <repo/image-name:tag> -f <docker-filename>`

## Push image to dockerhub        

> `sudo docker push <repo/image-name:tag>`
