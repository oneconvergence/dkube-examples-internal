FROM ocdr/dkube-datascience-tf-cpu:v1.12
USER root

#python2 set up required to start notebook in python2 env
RUN apt-get update
RUN apt-get install -y --no-install-recommends python-pip python-dev python-setuptools
RUN apt-get install -qq libfluidsynth1 build-essential libasound2-dev libjack-dev libsm6 libxext6 libxrender-dev gpg-agent
RUN pip2 install --upgrade pip
RUN pip2 install numpy==1.16.2
RUN pip2 install tensorflow
RUN pip2 install ipykernel

#Notebook specific dependency installation
#Python2 dependencies
RUN pip2 install lucid svgwrite matplotlib sklearn pandas gsutil xgboost
RUN pip2 install witwidget
RUN pip2 install magenta pyfluidsynth

#python3 dependencies
RUN pip3 install lucid dopamine-rl cmake atari_py opencv-python svgwrite matplotlib gsutil xgboost
RUN pip3 install tornado==4.5.3 anchor_exp==0.0.0.5
RUN pip3 install https://storage.googleapis.com/ml-explainability-solution/anchor_ai_platform-0.1.tar.gz
RUN pip3 install witwidget 
RUN pip3 install -q moviepy
RUN pip3 install librosa
RUN pip3 install adanet
RUN pip3 install -q pandas==0.22.0
RUN pip3 install kf
RUN pip3 -q install umap-learn
RUN pip3 install cmake
RUN pip3 install atari_py
RUN pip3 install sonnet
RUN pip3 install -q dm-sonnet tfp-nightly
#RUN pip3 install vega_dataset
RUN pip3 install -q gensim==3.2.0
RUN pip3 install -q git+https://github.com/conversationai/unintended-ml-bias-analysis

#other dependencies
RUN curl -sL https://deb.nodesource.com/setup_11.x  | bash -
RUN apt-get -y install nodejs
RUN npm install -g svelte-cli@2.2.0

USER dkube
#install ipykernel to start notebook in python2 env
WORKDIR /home/dkube/
RUN python2 -m ipykernel install --user

