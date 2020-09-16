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


EXPOSE 8080

ENTRYPOINT ["Rscript", "microservice.R", "--model=mnist.R", "--api=REST", "--service=MODEL", "--persistence=0"]
