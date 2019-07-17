FROM tensorflow/tensorflow:1.12.0-gpu-py3
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-pip python3-dev

RUN apt-get update && \
    apt install -y vim
RUN pip3 install keras

