# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.7-slim

ENV APP_HOME /app
WORKDIR $APP_HOME

RUN apt-get update
RUN apt-get install software-properties-common -y
RUN apt-get install libgtk2.0-dev -y
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
RUN add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran35/'
RUN apt install r-base -y
RUN Rscript -e "install.packages('png')"
RUN Rscript -e "install.packages('base64enc')"
RUN Rscript -e "install.packages('readr')"
# Install production dependencies.
RUN pip3 install rpy2
COPY . .
RUN pip3 install --no-cache-dir -r ./requirements.txt

ENTRYPOINT ["python3", "model.py"]