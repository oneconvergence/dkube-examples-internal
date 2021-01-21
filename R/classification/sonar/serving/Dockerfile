FROM rocker/r-apt:bionic

RUN apt-get update && \
    apt-get install -y -qq \
        r-cran-plumber \
        r-cran-jsonlite \
        r-cran-optparse \
        r-cran-stringr \
        r-cran-urltools \
        r-cran-pls \
        curl

RUN Rscript -e "install.packages('caret')"
RUN Rscript -e "install.packages('mlbench')"
RUN Rscript -e "install.packages('randomForest')"
RUN Rscript -e "install.packages('doMC')"
RUN Rscript -e "install.packages('e1071')"

RUN mkdir microservice
COPY . /microservice
WORKDIR /microservice

#dkube-kfserving port must be 8080
EXPOSE 8080

#dkube-kfserving use ENTRYPOINT in the below format instead of CMD
ENTRYPOINT ["Rscript", "microservice.R", "--model=sonar.R", "--api=REST", "--service=MODEL", "--persistence=0", "--model_name=sonar", "--model_base_path=/mnt/models/"]