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

ENV MODEL mnist.R
ENV API_TYPE REST
ENV SERVICE_TYPE MODEL
ENV PERSISTENCE 0

RUN mkdir microservice
COPY . /microservice
WORKDIR /microservice
COPY  mnist.R /microservice/
COPY  ./model.Rds /microservice/
COPY  ./mnist.Rds /microservice/



#RUN curl -OL https://raw.githubusercontent.com/SeldonIO/seldon-core/v0.2.7/wrappers/s2i/R/microservice.R > /microservice/microservice.R

EXPOSE 8080

#CMD Rscript microservice.R --model $MODEL --api $API_TYPE --service $SERVICE_TYPE --persistence $PERSISTENCE
#ENTRYPOINT ["Rscript", "microservice.R", "--model=$MODEL", "--api=$API_TYPE", "--service=$SERVICE_TYPE", "--persistence=$PERSISTENCE", "--modelname=rinfcitox"]
#ENTRYPOINT ["Rscript", "microservice.R", "--model=mnist.R", "--api=REST", "--service=MODEL", "--persistence=0", "--model_name=r-citox", "--model_base_path=/microservice"]
ENTRYPOINT ["Rscript", "microservice.R", "--model=mnist.R", "--api=REST", "--service=MODEL", "--persistence=0"]
