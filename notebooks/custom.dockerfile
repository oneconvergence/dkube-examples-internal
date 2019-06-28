FROM ocdr/dkube-datascience-tf-cpu:v1.12
USER root
#python2 set up required to start notebook in python2 env
RUN apt-get update
RUN apt-get install -y --no-install-recommends python-pip python-dev python-setuptools
RUN pip2 install --upgrade pip
RUN pip2 install tensorflow
RUN pip2 install ipykernel
  
#Notebook specific dependency installation
#Python2 dependencies
RUN pip2 install lucid svgwrite matplotlib sklearn pandas

#python3 dependencies
RUN pip3 install lucid dopamine-rl cmake atari_py opencv-python svgwrite matplotlib
RUN pip3 install tornado==4.5.3 anchor_exp==0.0.0.5
RUN pip3 install https://storage.googleapis.com/ml-explainability-solution/anchor_ai_platform-0.1.tar.gz

#other dependencies
RUN curl -sL https://deb.nodesource.com/setup_11.x  | bash -
RUN apt-get -y install nodejs
RUN npm install -g svelte-cli@2.2.0

USER dkube
#install ipykernel to start notebook in python2 env
WORKDIR /home/dkube/
RUN python2 -m ipykernel install --user
