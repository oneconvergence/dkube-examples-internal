FROM rocker/r-apt:bionic

RUN apt-get update && \
    apt-get install -y -qq \
    	r-cran-plumber \
    	r-cran-jsonlite \
    	r-cran-optparse \
    	r-cran-stringr \
    	r-cran-urltools \
    	r-cran-caret \
		r-cran-randomforest \
    	r-cran-pls \
    	curl

RUN R -e 'install.packages("OpenImageR")'
RUN R -e 'install.packages("hash")'
RUN R -e 'install.packages("reticulate")'

RUN mkdir microservice
COPY . /microservice
COPY https://github.com/oneconvergence/dkube-examples/blob/2.1.5/R/classification/mnist/mnist.R /microservice
WORKDIR /microservice

#dkube-kfserving port must be 8080
EXPOSE 8080

#dkube-kfserving use ENTRYPOINT in the below format instead of CMD
ENTRYPOINT ["Rscript", "microservice.R","--model=mnist.R", "--transformer=transformer.R", "--api=REST", "--service=TRANSFORMER", "--persistence=0"]
