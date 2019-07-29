FROM ubuntu:18.04

# Add a few needed packages to the base Ubuntu 18.04
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        wget \
        git \
        sudo \
        libcurl3-dev \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng-dev \
        libzmq3-dev \
        pkg-config \
        python3-pip \
        python3-dev \
        python3-setuptools \
        rsync \
        software-properties-common \
        unzip \
        zip \
        zlib1g-dev \
        libfluidsynth1 \
        libasound2-dev \
        libjack-dev \
        libsm6 \
        libxext6 \
        libxrender-dev \
        gpg-agent \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

#NB specific dependencies which may not in installed in nb cells
RUN curl -sL https://deb.nodesource.com/setup_11.x  | bash -
RUN apt-get -y install nodejs
RUN ln -s /usr/bin/pip3 /usr/bin/pip
RUN pip install --upgrade pip

RUN pip install \
        ipykernel \
        jupyter \
        setuptools \
        wheel \
        && \
    python3 -m ipykernel.kernelspec

RUN pip install \
        numpy==1.16.2 \
        tensorflow \
        lucid \
        svgwrite \
        matplotlib \
        sklearn \
        pandas \
        gsutil \
        xgboost \
        witwidget \
        magenta \
        pyfluidsynth


# Set up our notebook config.
COPY scripts/jupyter/jupyter_notebook_config.py /root/.jupyter/


# Jupyter has issues with being run directly:
#   https://github.com/ipython/ipython/issues/7062
# We just add a little wrapper script.
COPY scripts/jupyter/run_jupyter.sh /

# IPython
EXPOSE 8888
#RUN useradd -m dkube && echo "dkube:dkube" | chpasswd && adduser dkube sudo
#USER dkube

WORKDIR /root
LABEL heritage="dkube"
#RUN python -m ipykernel install --user
#RUN mkdir /home/dkube/.jupyter/
RUN bash -c "echo c.NotebookApp.ip = \'*\' > /root/.jupyter/jupyter_notebook_config.py"
CMD ["/run_jupyter.sh", "--allow-root"]
