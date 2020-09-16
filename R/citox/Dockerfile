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

RUN mkdir microservice
COPY . /microservice
WORKDIR /microservice

#dkube-kfserving port must be 8080
EXPOSE 8080

#dkube-kfserving use ENTRYPOINT in the below format instead of CMD
ENTRYPOINT ["Rscript", "microservice.R", "--model=mnist.R", "--api=REST", "--service=MODEL", "--persistence=0"]
