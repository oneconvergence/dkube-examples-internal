FROM tensorflow/tensorflow:1.14.0-py3-jupyter

RUN apt-get update && apt-get upgrade -y && \
        apt-get install -y --no-install-recommends \
        sudo \
        git \
        apt-utils \
        python3-pip \
        python3-dev \
        libsm6 \
        libxext6 

RUN apt-get -y install curl gnupg
RUN curl -sL https://deb.nodesource.com/setup_12.x  | bash -
RUN apt-get -y install nodejs
RUN npm install
# Arvados 
RUN apt-get install -y nano build-essential git python3-dev \
    libssl1.0-dev pkg-config

RUN apt-get install -y libcurl4-openssl-dev libssl-dev \
    python3-llfuse libfuse-dev ruby ruby-dev bundler

#RUN apt autoremove -y

RUN apt-get -y --no-install-recommends install gnupg

RUN /usr/bin/apt-key adv --keyserver pool.sks-keyservers.net --recv 1078ECD7

RUN echo "deb http://apt.arvados.org/ bionic main" | tee /etc/apt/sources.list.d/arvados.list

RUN apt update

RUN gem install arvados-cli

RUN pip3 install arvados-python-client


RUN export DEBIAN_FRONTEND=noninteractive

RUN ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime

RUN apt install git-el git-email git-gui gitk gitweb git-cvs git-mediawiki git-svn -y

RUN dpkg-reconfigure --frontend noninteractive tzdata


#PROCESSING
RUN pip3 install scoop && \
    pip3 install multiprocessing_generator


#GRAPHING
RUN pip3 install plotly && \
#    pip install python-igraph && \
    pip3 install seaborn && \
    pip3 install altair && \
    pip3 install git+https://github.com/jakevdp/JSAnimation.git && \
    pip3 install bokeh

#TEXT PROCESSING
RUN pip3 install textblob && \
    pip3 install git+git://github.com/amueller/word_cloud.git && \
    pip3 install toolz cytoolz && \
    pip3 install gensim && \
    pip3 install bs4


#DATA
RUN pip3 install h5py && \
    pip3 install pyexcel-ods && \
    pip3 install pandas-profiling

#IMAGE
RUN apt-get install --no-install-recommends -y  \
    ffmpeg \
    imagemagick

RUN pip3 install pydicom && \
    pip3 install scikit-image && \
    pip3 install opencv-python && \
    pip3 install ImageHash

#LEARNING
RUN apt-get install pandoc --no-install-recommends -y && \
    pip3 install pypandoc && pip3 install deap && pip3 install scikit-learn && \
    pip3 install git+https://github.com/tflearn/tflearn.git

RUN pip3 install Keras==2.3.1

RUN pip3 uninstall -y enum34
#MISC
RUN pip3 install tensorflow_hub==0.3.0 &&\
    pip3 install Cython &&\
    pip3 install tensornets &&\
    pip3 install requests

#AUDIO
RUN apt-get install -y --no-install-recommends libsndfile1 && \
    pip3 install librosa

#Object detection
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y --no-install-recommends tzdata unzip && \
    apt-get -y install --no-install-recommends  python3-tk wget && \
    pip3 install absl-py && \
    pip3 install matplotlib && \
    pip3 install pillow && \
    pip3 install pycocotools 

# Installing jupyterlab
RUN pip3 install jupyterlab && \
    pip3 install jupyterlab[extras] && \
    pip3 install jupyterlab-github && \
    pip3 install jupyterlab_github 

# Installing git extension for jupyterlab
RUN pip3 install jupyterlab-git  && \
    jupyter lab build --dev-build=False --minimize=False

RUN jupyter labextension install @jupyterlab/github
RUN jupyter labextension install @jupyterlab/google-drive

RUN jupyter serverextension enable --py jupyterlab_git --system

RUN jupyter serverextension enable --py jupyterlab_github --system


#CLEANUP
RUN rm -rf /root/.cache/pip/* && \
    rm -rf /root/.cache/pip3/* && \
    apt-get autoremove -y && \
    apt-get clean

#RUN apt-get -y install --no-install-recommends ipython3
RUN pip3 install ipykernel
RUN useradd -m dkube && echo "dkube:dkube" | chpasswd && adduser dkube sudo
RUN usermod -aG sudo,root dkube
RUN echo 'dkube ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER dkube

WORKDIR /home/dkube/
LABEL heritage="dkube"
RUN python3 -m ipykernel install --user
RUN mkdir /home/dkube/.jupyter/
RUN mkdir /home/dkube/.config/
RUN mkdir /home/dkube/.config/arvados/
COPY bashrc .bashrc
RUN jupyter notebook --generate-config
RUN git config --global user.email "dkube@oneconvergence.com"
RUN git config --global user.name "dkube"
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]
